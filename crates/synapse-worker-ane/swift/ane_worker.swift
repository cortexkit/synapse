import CoreML
import CryptoKit
import Darwin
import Foundation

private let protocolVersion = 1
private let maxFrameBytes = 64 * 1024 * 1024
private let engineVersion = "coreml-swift-v1"

private enum WorkerError: Error, CustomStringConvertible {
    case invalid(String)
    case io(String)

    var description: String {
        switch self {
        case .invalid(let message): return message
        case .io(let message): return message
        }
    }
}

private struct Args {
    let socket: String
    let nonce: String
    let testAbort: Bool
    let testAbortOnRequest: Bool
}

private struct LoadedBucket {
    let model: MLModel
    let modelURL: URL
    let outputName: String
    let bucket: Int
    let dims: Int
    let temporaryRoot: URL?
}

private struct LoadedModel {
    let buckets: [LoadedBucket]
}

private struct WorkerState {
    var models: [String: LoadedModel] = [:]
    var nextModel: UInt64 = 0
    var lastPlacementShare: Double? = nil
}

@main
private struct Main {
    static func main() async {
        do {
            let args = try parseArgs()
            let fd = try connectUnixSocket(path: args.socket)
            defer { close(fd) }
            try writeJSONFrame(fd: fd, value: [
                "v": protocolVersion,
                "nonce": args.nonce,
                "engine": [
                    "engine": "ane-coreml",
                    "version": engineVersion,
                    "build_flags": [
                        "risk_class": "abort_capable",
                        "backend": "coreml",
                        "placement": "neural-engine"
                    ]
                ],
                "pid": Int(ProcessInfo.processInfo.processIdentifier),
                "max_frame": maxFrameBytes
            ])
            let ack = try readJSONFrame(fd: fd)
            guard (ack["v"] as? Int) == protocolVersion, (ack["accept"] as? Bool) == true else {
                throw WorkerError.invalid("module rejected worker handshake")
            }
            let acceptedFrame = min(maxFrameBytes, ack["max_frame"] as? Int ?? maxFrameBytes)
            var state = WorkerState()

            while true {
                let requestData: Data
                do {
                    requestData = try readFrame(fd: fd, maxFrame: acceptedFrame)
                } catch WorkerError.io(let message) where message == "eof" {
                    break
                }
                let request = try jsonObject(requestData)
                if args.testAbortOnRequest || (args.testAbort && (request["type"] as? String) != "LOAD") {
                    abort()
                }
                let type = request["type"] as? String ?? ""
                switch type {
                case "LOAD":
                    let reqId = stringField(request, "req_id")
                    do {
                        let response = try await handleLoad(state: &state, request: request, reqId: reqId)
                        try writeJSONFrame(fd: fd, value: response)
                    } catch {
                        try writeJSONFrame(fd: fd, value: err(reqId: reqId, code: classifyLoadError(error), msg: String(describing: error)))
                    }
                case "EMBED_BATCH":
                    let reqId = stringField(request, "req_id")
                    let raw = try readFrame(fd: fd, maxFrame: acceptedFrame)
                    do {
                        let (response, vectors) = try handleEmbedBatch(state: state, request: request, reqId: reqId, raw: raw)
                        try writeJSONFrame(fd: fd, value: response)
                        try writeFrame(fd: fd, data: vectors, maxFrame: acceptedFrame)
                    } catch {
                        try writeJSONFrame(fd: fd, value: err(reqId: reqId, code: "inference_failed", msg: String(describing: error)))
                    }
                case "RERANK", "GENERATE":
                    let reqId = request["req_id"] as? String
                    _ = try readFrame(fd: fd, maxFrame: acceptedFrame)
                    try writeJSONFrame(fd: fd, value: err(
                        reqId: reqId,
                        code: "unknown_type",
                        msg: "synapse-worker-ane v1 supports LOAD, EMBED_BATCH, PING, UNLOAD, and SHUTDOWN only"
                    ))
                case "UNLOAD":
                    let reqId = stringField(request, "req_id")
                    let modelRef = stringField(request, "model_ref")
                    if let loaded = state.models.removeValue(forKey: modelRef) {
                        removeTemporaryRoots(loaded.buckets)
                    }
                    try writeJSONFrame(fd: fd, value: ["type": "UNLOADED", "req_id": reqId])
                case "PING":
                    let reqId = stringField(request, "req_id")
                    var response: [String: Any] = [
                        "type": "PONG",
                        "req_id": reqId,
                        "rss_mb": 0,
                        "models_loaded": state.models.count
                    ]
                    if let share = state.lastPlacementShare {
                        response["placement_share"] = share
                    }
                    try writeJSONFrame(fd: fd, value: response)
                case "SHUTDOWN":
                    return
                default:
                    try writeJSONFrame(fd: fd, value: err(reqId: request["req_id"] as? String, code: "unknown_type", msg: "unknown worker request type \(type)"))
                }
            }
        } catch {
            fputs("synapse-worker-ane: \(error)\n", stderr)
            Foundation.exit(1)
        }
    }
}

private func parseArgs() throws -> Args {
    var socket: String?
    var nonce: String?
    var testAbort = false
    var testAbortOnRequest = false
    var iterator = CommandLine.arguments.dropFirst().makeIterator()
    while let arg = iterator.next() {
        switch arg {
        case "--socket": socket = iterator.next()
        case "--nonce": nonce = iterator.next()
        case "--test-abort": testAbort = true
        case "--test-abort-on-request": testAbortOnRequest = true
        default: throw WorkerError.invalid("unknown argument \(arg)")
        }
    }
    guard let socket, let nonce else {
        throw WorkerError.invalid("usage: synapse-worker-ane --socket <path> --nonce <hex16>")
    }
    return Args(socket: socket, nonce: nonce, testAbort: testAbort, testAbortOnRequest: testAbortOnRequest)
}

private func handleLoad(state: inout WorkerState, request: [String: Any], reqId: String) async throws -> [String: Any] {
    let artifactPath = stringField(request, "artifact_path")
    let artifactDigest = request["artifact_digest"] as? String ?? ""
    let format = stringField(request, "format")
    guard format == "mlmodelc" || format == "coreml" else {
        throw WorkerError.invalid("ANE worker only loads compiled .mlmodelc artifacts, got \(format)")
    }

    let runtimeConfig = request["runtime_config"] as? [String: Any] ?? [:]
    let paths = jsonStringArray(runtimeConfig["artifact_paths"] as? String) ?? [artifactPath]
    let digests = jsonStringArray(runtimeConfig["artifact_digests"] as? String) ?? [artifactDigest]
    guard paths.count == digests.count else {
        throw WorkerError.invalid("ANE artifact_paths and artifact_digests must have the same length")
    }

    let started = monotonicTime()
    let configuration = MLModelConfiguration()
    configuration.computeUnits = .cpuAndNeuralEngine
    if #available(macOS 14.4, *) {
        configuration.optimizationHints.reshapeFrequency = .infrequent
    }

    var temporaryRoots: [URL] = []
    var buckets: [LoadedBucket] = []
    do {
        for (path, digest) in zip(paths, digests) {
            let sourceURL = URL(fileURLWithPath: path).standardizedFileURL
            try verifyDigest(url: sourceURL, digest: digest)
            let (modelURL, temporaryRoot) = try materializeCoreMLArtifact(
                sourceURL: sourceURL,
                format: format
            )
            if let temporaryRoot {
                temporaryRoots.append(temporaryRoot)
            }
            let model = try MLModel(contentsOf: modelURL, configuration: configuration)
            let outputName = try selectOutputName(model: model)
            let bucket = try inferBucket(model: model)
            let dims = try inferOutputDims(model: model, outputName: outputName, bucket: bucket)
            buckets.append(LoadedBucket(
                model: model,
                modelURL: modelURL,
                outputName: outputName,
                bucket: bucket,
                dims: dims,
                temporaryRoot: temporaryRoot
            ))
        }
    } catch {
        removeTemporaryRoots(temporaryRoots)
        throw error
    }
    guard let dims = buckets.first?.dims else {
        throw WorkerError.invalid("ANE model load received no Core ML packages")
    }
    buckets.sort { $0.bucket < $1.bucket }
    var placements: [Double?] = []
    for bucket in buckets {
        placements.append(await placementShare(modelURL: bucket.modelURL, configuration: configuration))
    }
    state.lastPlacementShare = placements.compactMap { $0 }.min()

    let modelRef = "ane:\(state.nextModel)"
    state.nextModel += 1
    state.models[modelRef] = LoadedModel(buckets: buckets)
    return [
        "type": "LOADED",
        "req_id": reqId,
        "model_ref": modelRef,
        "dims": dims,
        "cold_load_ms": UInt64((monotonicTime() - started) * 1000)
    ]
}

private func handleEmbedBatch(state: WorkerState, request: [String: Any], reqId: String, raw: Data) throws -> ([String: Any], Data) {
    let modelRef = stringField(request, "model_ref")
    guard let loaded = state.models[modelRef] else {
        throw WorkerError.invalid("unknown model_ref '\(modelRef)'")
    }
    let normalize = request["normalize"] as? Bool ?? true
    let pooling = request["pooling"] as? String ?? "mean"
    guard pooling == "mean" else {
        throw WorkerError.invalid("ANE worker supports mean pooling only, got \(pooling)")
    }
    guard let itemDicts = request["items"] as? [[String: Any]], !itemDicts.isEmpty else {
        throw WorkerError.invalid("EMBED_BATCH requires at least one item")
    }
    let ids = try decodeI32Frame(raw)
    let tokenCounts = itemDicts.map { $0["n_tokens"] as? Int ?? 0 }
    let expected = tokenCounts.reduce(0, +)
    guard ids.count == expected else {
        throw WorkerError.invalid("raw id frame has \(ids.count) tokens, expected \(expected)")
    }
    guard let maxTokens = tokenCounts.max(),
          let bucket = loaded.buckets.first(where: { $0.bucket >= maxTokens }) else {
        let available = loaded.buckets.map { $0.bucket }.map(String.init).joined(separator: ",")
        throw WorkerError.invalid("batch requires \(tokenCounts.max() ?? 0) tokens but ANE buckets are [\(available)]")
    }

    var rows: [(ids: [Int], mask: [Int])] = []
    rows.reserveCapacity(itemDicts.count)
    var offset = 0
    for (index, nTokens) in tokenCounts.enumerated() {
        guard nTokens > 0 else { throw WorkerError.invalid("item \(index) has zero tokens") }
        var rowIds = ids[offset ..< offset + nTokens].map(Int.init)
        var mask = [Int](repeating: 1, count: nTokens)
        if nTokens < bucket.bucket {
            rowIds.append(contentsOf: repeatElement(0, count: bucket.bucket - nTokens))
            mask.append(contentsOf: repeatElement(0, count: bucket.bucket - nTokens))
        }
        rows.append((rowIds, mask))
        offset += nTokens
    }

    let providers = try rows.map { row in try featureProvider(inputIds: row.ids, attentionMask: row.mask) }
    let predictions = try bucket.model.predictions(fromBatch: MLArrayBatchProvider(array: providers))
    var flat: [Float] = []
    flat.reserveCapacity(rows.count * bucket.dims)
    for index in 0 ..< predictions.count {
        let output = predictions.features(at: index)
        guard let value = output.featureValue(for: bucket.outputName)?.multiArrayValue else {
            throw WorkerError.invalid("prediction output is missing feature \(bucket.outputName)")
        }
        var vector = try poolAndNormalize(hiddenStates: value, attentionMask: rows[index].mask)
        if !normalize {
            vector = try poolOnly(hiddenStates: value, attentionMask: rows[index].mask)
        }
        guard vector.count == bucket.dims else {
            throw WorkerError.invalid("vector dims \(vector.count) did not match loaded dims \(bucket.dims)")
        }
        flat.append(contentsOf: vector)
    }

    return ([
        "type": "VECTORS",
        "req_id": reqId,
        "dims": bucket.dims,
        "n": rows.count
    ], encodeF32Frame(flat))
}

private func jsonStringArray(_ encoded: String?) -> [String]? {
    guard let encoded, let data = encoded.data(using: .utf8) else { return nil }
    guard let value = try? JSONSerialization.jsonObject(with: data),
          let values = value as? [Any] else { return nil }
    let strings = values.compactMap { $0 as? String }
    return strings.count == values.count ? strings : nil
}

private func materializeCoreMLArtifact(sourceURL: URL, format: String) throws -> (URL, URL?) {
    var isDirectory: ObjCBool = false
    guard FileManager.default.fileExists(atPath: sourceURL.path, isDirectory: &isDirectory) else {
        throw WorkerError.invalid("artifact path does not exist: \(sourceURL.path)")
    }
    if isDirectory.boolValue {
        return (sourceURL, nil)
    }
    guard format == "mlmodelc" || format == "coreml" else {
        throw WorkerError.invalid("unsupported Core ML artifact format \(format)")
    }
    let root = FileManager.default.temporaryDirectory
        .appendingPathComponent("synapse-ane-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    let process = Process()
    process.executableURL = URL(fileURLWithPath: "/usr/bin/unzip")
    process.arguments = ["-q", "-o", sourceURL.path, "-d", root.path]
    let errorPipe = Pipe()
    process.standardError = errorPipe
    try process.run()
    process.waitUntilExit()
    guard process.terminationStatus == 0 else {
        let message = String(data: errorPipe.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8) ?? "unknown unzip error"
        try? FileManager.default.removeItem(at: root)
        throw WorkerError.invalid("unzip Core ML artifact failed: \(message.trimmingCharacters(in: .whitespacesAndNewlines))")
    }
    let candidates = ([root] + (FileManager.default.enumerator(
        at: root,
        includingPropertiesForKeys: [.isDirectoryKey],
        options: [.skipsHiddenFiles]
    )?.compactMap { $0 as? URL } ?? [])).filter { url in
        var candidateIsDirectory: ObjCBool = false
        return FileManager.default.fileExists(atPath: url.path, isDirectory: &candidateIsDirectory)
            && candidateIsDirectory.boolValue
            && FileManager.default.fileExists(atPath: url.appendingPathComponent("model.mlmodel").path)
    }
    guard let modelURL = candidates.first else {
        try? FileManager.default.removeItem(at: root)
        throw WorkerError.invalid("unzipped Core ML artifact did not contain a compiled .mlmodelc bundle")
    }
    return (modelURL, root)
}

private func removeTemporaryRoots(_ roots: [URL]) {
    for root in roots {
        try? FileManager.default.removeItem(at: root)
    }
}

private func removeTemporaryRoots(_ buckets: [LoadedBucket]) {
    removeTemporaryRoots(buckets.compactMap { $0.temporaryRoot })
}

private func selectOutputName(model: MLModel) throws -> String {
    let outputs = Array(model.modelDescription.outputDescriptionsByName.keys).sorted()
    if outputs.count == 1, let only = outputs.first { return only }
    if outputs.contains("last_hidden_state") { return "last_hidden_state" }
    throw WorkerError.invalid("model outputs require an explicit known last_hidden_state output, got \(outputs)")
}

private func inferBucket(model: MLModel) throws -> Int {
    guard let input = model.modelDescription.inputDescriptionsByName["input_ids"],
          let shape = input.multiArrayConstraint?.shape.map({ $0.intValue }),
          let bucket = shape.last,
          bucket > 0 else {
        throw WorkerError.invalid("Core ML model must expose fixed input_ids multiarray shape")
    }
    return bucket
}

private func inferOutputDims(model: MLModel, outputName: String, bucket: Int) throws -> Int {
    if let output = model.modelDescription.outputDescriptionsByName[outputName],
       let shape = output.multiArrayConstraint?.shape.map({ $0.intValue }),
       shape.count == 3,
       shape[1] == bucket,
       shape[2] > 0 {
        return shape[2]
    }
    let zeroIds = [Int](repeating: 0, count: bucket)
    var mask = [Int](repeating: 0, count: bucket)
    mask[0] = 1
    let output = try model.prediction(from: featureProvider(inputIds: zeroIds, attentionMask: mask))
    guard let value = output.featureValue(for: outputName)?.multiArrayValue else {
        throw WorkerError.invalid("prediction output is missing feature \(outputName)")
    }
    let shape = value.shape.map { $0.intValue }
    guard shape.count == 3, shape[2] > 0 else {
        throw WorkerError.invalid("expected output shape [1, bucket, hidden], got \(shape)")
    }
    return shape[2]
}

private func featureProvider(inputIds: [Int], attentionMask: [Int]) throws -> MLDictionaryFeatureProvider {
    try MLDictionaryFeatureProvider(dictionary: [
        "input_ids": MLFeatureValue(multiArray: makeMultiArray(values: inputIds)),
        "attention_mask": MLFeatureValue(multiArray: makeMultiArray(values: attentionMask)),
    ])
}

private func makeMultiArray(values: [Int]) throws -> MLMultiArray {
    let array = try MLMultiArray(shape: [1, NSNumber(value: values.count)], dataType: .int32)
    try array.withUnsafeMutableBufferPointer(ofType: Int32.self) { buffer, _ in
        guard buffer.count == values.count else {
            throw WorkerError.invalid("unexpected MLMultiArray backing length: \(buffer.count)")
        }
        for (index, value) in values.enumerated() {
            guard let cast = Int32(exactly: value) else {
                throw WorkerError.invalid("input value \(value) does not fit in Int32")
            }
            buffer[index] = cast
        }
    }
    return array
}

private func poolOnly(hiddenStates: MLMultiArray, attentionMask: [Int]) throws -> [Float] {
    try pool(hiddenStates: hiddenStates, attentionMask: attentionMask, normalize: false)
}

private func poolAndNormalize(hiddenStates: MLMultiArray, attentionMask: [Int]) throws -> [Float] {
    try pool(hiddenStates: hiddenStates, attentionMask: attentionMask, normalize: true)
}

private func pool(hiddenStates: MLMultiArray, attentionMask: [Int], normalize: Bool) throws -> [Float] {
    let shape = hiddenStates.shape.map { $0.intValue }
    guard shape.count == 3, shape[0] == 1 else {
        throw WorkerError.invalid("expected output shape [1, bucket, hidden], got \(shape)")
    }
    let bucket = shape[1]
    let hidden = shape[2]
    guard attentionMask.count == bucket else {
        throw WorkerError.invalid("attention mask length \(attentionMask.count) does not match output bucket \(bucket)")
    }
    var pooled = [Float](repeating: 0, count: hidden)
    var denominator: Float = 0
    try readFloatArray(hiddenStates) { buffer in
        guard buffer.count == bucket * hidden else {
            throw WorkerError.invalid("output element count \(buffer.count) does not match bucket*hidden \(bucket * hidden)")
        }
        for tokenIndex in 0 ..< bucket {
            let weight = Float(attentionMask[tokenIndex])
            if weight == 0 { continue }
            denominator += weight
            let base = tokenIndex * hidden
            for hiddenIndex in 0 ..< hidden {
                pooled[hiddenIndex] += buffer[base + hiddenIndex] * weight
            }
        }
    }
    guard denominator > 0 else { throw WorkerError.invalid("attention mask is all zeros") }
    for index in pooled.indices { pooled[index] /= denominator }
    if normalize {
        var normSquared: Float = 0
        for value in pooled { normSquared += value * value }
        let norm = sqrt(normSquared)
        if norm > 0 {
            for index in pooled.indices { pooled[index] /= norm }
        }
    }
    return pooled
}

private func readFloatArray(_ array: MLMultiArray, body: ([Float]) throws -> Void) throws {
    switch array.dataType {
    case .float16:
        try array.withUnsafeBytes { rawBuffer in
            let values = rawBuffer.bindMemory(to: UInt16.self).map { Float(Float16(bitPattern: $0)) }
            try body(values)
        }
    case .float32:
        try array.withUnsafeBytes { rawBuffer in
            let values = rawBuffer.bindMemory(to: Float32.self).map { Float($0) }
            try body(values)
        }
    case .double:
        try array.withUnsafeBytes { rawBuffer in
            let values = rawBuffer.bindMemory(to: Double.self).map { Float($0) }
            try body(values)
        }
    default:
        throw WorkerError.invalid("unsupported MLMultiArray output data type: \(array.dataType.rawValue)")
    }
}

private func placementShare(modelURL: URL, configuration: MLModelConfiguration) async -> Double? {
    guard #available(macOS 14.4, *) else { return nil }
    do {
        let report = try await buildPlacementReport(modelURL: modelURL, configuration: configuration)
        return report
    } catch {
        fputs("ANE placement check failed: \(error)\n", stderr)
        return nil
    }
}

@available(macOS 14.4, *)
private func buildPlacementReport(modelURL: URL, configuration: MLModelConfiguration) async throws -> Double? {
    let plan = try await MLComputePlan.load(contentsOf: modelURL, configuration: configuration)
    var preferred: [String] = []
    switch plan.modelStructure {
    case .program(let program):
        let function = program.functions["main"] ?? program.functions.values.sorted { $0.inputs.count < $1.inputs.count }.first
        guard let function else { return nil }
        collectProgramOperations(plan: plan, block: function.block, into: &preferred)
    case .neuralNetwork(let network):
        for layer in network.layers {
            if let usage = plan.deviceUsage(for: layer) {
                preferred.append(deviceLabel(usage.preferred))
            }
        }
    case .pipeline, .unsupported:
        return nil
    @unknown default:
        return nil
    }
    let dispatchable = preferred.filter { $0 != "unknown" }
    guard !dispatchable.isEmpty else { return nil }
    let neural = dispatchable.filter { $0 == "neuralEngine" }.count
    return Double(neural) / Double(dispatchable.count)
}

@available(macOS 14.4, *)
private func collectProgramOperations(plan: MLComputePlan, block: MLModelStructure.Program.Block, into preferred: inout [String]) {
    for operation in block.operations {
        if let usage = plan.deviceUsage(for: operation) {
            preferred.append(deviceLabel(usage.preferred))
        } else {
            preferred.append("unknown")
        }
        for nestedBlock in operation.blocks {
            collectProgramOperations(plan: plan, block: nestedBlock, into: &preferred)
        }
    }
}

private func deviceLabel(_ device: MLComputeDevice) -> String {
    switch device {
    case .cpu: return "cpu"
    case .gpu: return "gpu"
    case .neuralEngine: return "neuralEngine"
    @unknown default: return "unknown"
    }
}

private func verifyDigest(url: URL, digest: String) throws {
    let trimmed = digest.trimmingCharacters(in: .whitespacesAndNewlines)
    if trimmed.isEmpty { return }
    let expected = trimmed.hasPrefix("sha256:") ? String(trimmed.dropFirst("sha256:".count)) : trimmed
    let actual = try sha256Path(url: url)
    guard expected == actual else {
        throw WorkerError.invalid("artifact digest mismatch for \(url.path): expected \(expected), got \(actual)")
    }
}

private func sha256Path(url: URL) throws -> String {
    var isDirectory: ObjCBool = false
    guard FileManager.default.fileExists(atPath: url.path, isDirectory: &isDirectory) else {
        throw WorkerError.invalid("artifact path does not exist: \(url.path)")
    }
    var hasher = SHA256()
    if isDirectory.boolValue {
        let files = try FileManager.default.subpathsOfDirectory(atPath: url.path)
            .map { url.appendingPathComponent($0) }
            .filter { path in
                var isDir: ObjCBool = false
                return FileManager.default.fileExists(atPath: path.path, isDirectory: &isDir) && !isDir.boolValue
            }
            .sorted { $0.path < $1.path }
        for file in files {
            let rel = file.path.replacingOccurrences(of: url.path + "/", with: "")
            hasher.update(data: Data(rel.utf8))
            hasher.update(data: Data([0]))
            try updateHasher(&hasher, from: file)
        }
    } else {
        try updateHasher(&hasher, from: url)
    }
    return SHA256DigestToHex(hasher.finalize())
}

private func updateHasher(_ hasher: inout SHA256, from url: URL) throws {
    let handle = try FileHandle(forReadingFrom: url)
    defer { try? handle.close() }
    while let chunk = try handle.read(upToCount: 4 * 1024 * 1024), !chunk.isEmpty {
        hasher.update(data: chunk)
    }
}

private func SHA256DigestToHex(_ digest: SHA256.Digest) -> String {
    digest.map { String(format: "%02x", $0) }.joined()
}

private func decodeI32Frame(_ data: Data) throws -> [Int32] {
    guard data.count % 4 == 0 else { throw WorkerError.invalid("i32 frame length is not divisible by 4") }
    var values: [Int32] = []
    values.reserveCapacity(data.count / 4)
    for offset in stride(from: 0, to: data.count, by: 4) {
        let chunk = data[offset ..< offset + 4]
        let value = chunk.withUnsafeBytes { raw -> Int32 in
            let bytes = raw.bindMemory(to: UInt8.self)
            let rawValue = UInt32(bytes[0]) | (UInt32(bytes[1]) << 8) | (UInt32(bytes[2]) << 16) | (UInt32(bytes[3]) << 24)
            return Int32(bitPattern: rawValue)
        }
        values.append(value)
    }
    return values
}

private func encodeF32Frame(_ values: [Float]) -> Data {
    var data = Data(capacity: values.count * 4)
    for value in values {
        var bits = value.bitPattern.littleEndian
        data.append(Data(bytes: &bits, count: 4))
    }
    return data
}

private func classifyLoadError(_ error: Error) -> String {
    let message = String(describing: error)
    if message.contains("compute") || message.contains("configuration") {
        return "config_invalid"
    }
    return "artifact_invalid"
}

private func err(reqId: String?, code: String, msg: String) -> [String: Any] {
    var value: [String: Any] = ["type": "ERR", "code": code, "msg": msg]
    if let reqId { value["req_id"] = reqId }
    return value
}

private func stringField(_ object: [String: Any], _ name: String) -> String {
    object[name] as? String ?? ""
}

private func jsonObject(_ data: Data) throws -> [String: Any] {
    guard let object = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
        throw WorkerError.invalid("frame is not a JSON object")
    }
    return object
}

private func readJSONFrame(fd: Int32) throws -> [String: Any] {
    try jsonObject(readFrame(fd: fd, maxFrame: maxFrameBytes))
}

private func writeJSONFrame(fd: Int32, value: [String: Any]) throws {
    let data = try JSONSerialization.data(withJSONObject: value, options: [])
    try writeFrame(fd: fd, data: data, maxFrame: maxFrameBytes)
}

private func readFrame(fd: Int32, maxFrame: Int) throws -> Data {
    let lenBytes = try readExactly(fd: fd, count: 4)
    let len = lenBytes.withUnsafeBytes { raw -> UInt32 in
        let bytes = raw.bindMemory(to: UInt8.self)
        return UInt32(bytes[0]) | (UInt32(bytes[1]) << 8) | (UInt32(bytes[2]) << 16) | (UInt32(bytes[3]) << 24)
    }
    guard len <= UInt32(maxFrame) else {
        throw WorkerError.io("frame length \(len) exceeds max \(maxFrame)")
    }
    return try readExactly(fd: fd, count: Int(len))
}

private func writeFrame(fd: Int32, data: Data, maxFrame: Int) throws {
    guard data.count <= maxFrame else {
        throw WorkerError.io("frame length \(data.count) exceeds max \(maxFrame)")
    }
    var len = UInt32(data.count).littleEndian
    try withUnsafeBytes(of: &len) { raw in
        try writeExactly(fd: fd, bytes: raw)
    }
    try data.withUnsafeBytes { raw in
        try writeExactly(fd: fd, bytes: raw)
    }
}

private func readExactly(fd: Int32, count: Int) throws -> Data {
    var data = Data(count: count)
    var offset = 0
    while offset < count {
        let readCount = data.withUnsafeMutableBytes { raw -> Int in
            let base = raw.baseAddress!.advanced(by: offset)
            return Darwin.read(fd, base, count - offset)
        }
        if readCount == 0 {
            if offset == 0 { throw WorkerError.io("eof") }
            throw WorkerError.io("unexpected EOF")
        }
        if readCount < 0 {
            if errno == EINTR { continue }
            throw WorkerError.io(String(cString: strerror(errno)))
        }
        offset += readCount
    }
    return data
}

private func writeExactly(fd: Int32, bytes: UnsafeRawBufferPointer) throws {
    var offset = 0
    while offset < bytes.count {
        let wrote = Darwin.write(fd, bytes.baseAddress!.advanced(by: offset), bytes.count - offset)
        if wrote < 0 {
            if errno == EINTR { continue }
            throw WorkerError.io(String(cString: strerror(errno)))
        }
        offset += wrote
    }
}

private func connectUnixSocket(path: String) throws -> Int32 {
    let fd = socket(AF_UNIX, SOCK_STREAM, 0)
    guard fd >= 0 else { throw WorkerError.io(String(cString: strerror(errno))) }
    var addr = sockaddr_un()
    addr.sun_family = sa_family_t(AF_UNIX)
    let maxPath = MemoryLayout.size(ofValue: addr.sun_path)
    guard path.utf8.count < maxPath else {
        close(fd)
        throw WorkerError.invalid("socket path is too long")
    }
    path.withCString { cString in
        withUnsafeMutablePointer(to: &addr.sun_path.0) { pointer in
            _ = strncpy(pointer, cString, maxPath)
        }
    }
    let status = withUnsafePointer(to: &addr) { pointer in
        pointer.withMemoryRebound(to: sockaddr.self, capacity: 1) { sockaddrPointer in
            Darwin.connect(fd, sockaddrPointer, socklen_t(MemoryLayout<sockaddr_un>.size))
        }
    }
    guard status == 0 else {
        let message = String(cString: strerror(errno))
        close(fd)
        throw WorkerError.io(message)
    }
    return fd
}

private func monotonicTime() -> Double {
    var time = timespec()
    clock_gettime(CLOCK_MONOTONIC_RAW, &time)
    return Double(time.tv_sec) + Double(time.tv_nsec) / 1_000_000_000
}
