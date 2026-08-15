import AnePrefillSidecar
import CoreML
import CryptoKit
import Darwin
import Foundation

private struct Arguments {
    let socket: String
    let nonce: String
}

private struct ModelShape: Equatable {
    let window: Int
    let layers: Int
    let kvHeads: Int
    let headDimension: Int
    let vocabularySize: Int

    init(_ request: InstallRequest) {
        window = request.window
        layers = request.layers
        kvHeads = request.kvHeads
        headDimension = request.headDimension
        vocabularySize = request.vocabularySize
    }
}

private final class InstalledModel: @unchecked Sendable {
    let model: MLModel
    let artifactSHA256: String
    let shape: ModelShape
    let readinessMilliseconds: Double

    init(model: MLModel, artifactSHA256: String, shape: ModelShape, readinessMilliseconds: Double) {
        self.model = model
        self.artifactSHA256 = artifactSHA256
        self.shape = shape
        self.readinessMilliseconds = readinessMilliseconds
    }
}

private struct ExecutionLease: @unchecked Sendable {
    let request: ExecuteRequest
    let model: InstalledModel
    let ticket: ExecutionTicket
    let publication: SharedMemoryPublication
}

private struct ExecutionResult {
    let digest: String
}

private final class CoreMLStage: @unchecked Sendable {
    private let modelLock = NSLock()
    private var models: [String: InstalledModel] = [:]
    private let executions = ExecutionRegistry()
    private let timings = TimingLedger()

    func install(_ request: InstallRequest) async throws -> InstalledModel {
        let shape = ModelShape(request)
        if let existing = modelLock.withLock({ models[request.modelRef] }) {
            guard existing.artifactSHA256 == normalizedDigest(request.artifactSHA256),
                existing.shape == shape
            else {
                throw SidecarError.invalid(
                    "model_ref is already installed with a different artifact or shape")
            }
            return existing
        }

        let started = monotonicMilliseconds()
        let loaded: InstalledModel = try await withinReadinessBudget(
            milliseconds: request.readinessTimeoutMilliseconds
        ) {
            let artifact = URL(fileURLWithPath: request.artifactPath).standardizedFileURL
            try verifyDigest(at: artifact, expected: request.artifactSHA256)
            let configuration = MLModelConfiguration()
            configuration.computeUnits = .cpuAndNeuralEngine
            let model = try MLModel(contentsOf: artifact, configuration: configuration)
            return InstalledModel(
                model: model,
                artifactSHA256: normalizedDigest(request.artifactSHA256),
                shape: shape,
                readinessMilliseconds: monotonicMilliseconds() - started
            )
        }

        return try modelLock.withLock {
            if let existing = models[request.modelRef] {
                guard existing.artifactSHA256 == loaded.artifactSHA256, existing.shape == shape
                else {
                    throw SidecarError.invalid(
                        "model_ref is already installed with a different artifact or shape")
                }
                return existing
            }
            models[request.modelRef] = loaded
            return loaded
        }
    }

    func reserve(_ request: ExecuteRequest) throws -> ExecutionLease {
        modelLock.lock()
        let model = models[request.modelRef]
        modelLock.unlock()
        guard let model else { throw SidecarError.notFound(request.modelRef) }
        try validateFixedWindowRequest(request, shape: model.shape)
        try validateSharedMemoryHandoff(request.handoff, request: request, shape: model.shape)
        let publication = try SharedMemoryPublication(descriptor: request.handoff)
        let ticket = try executions.begin(executionID: request.requestID)
        return ExecutionLease(
            request: request, model: model, ticket: ticket, publication: publication)
    }

    func execute(_ lease: ExecutionLease) async throws -> ExecutionResult {
        let executionStarted = monotonicMilliseconds()
        var predictionMilliseconds = 0.0
        var kvLayoutMilliseconds = 0.0
        var logitsCopyMilliseconds = 0.0
        var integrityMilliseconds = 0.0

        do {
            try throwIfCancelled(lease.ticket)
            let provider = try featureProvider(for: lease.request)
            let shape = lease.model.shape
            let kvPreparationStarted = monotonicMilliseconds()
            try lease.publication.begin()
            try lease.publication.zeroKV()
            let kvBackings = try lease.publication.withMutableBytes(
                offset: lease.request.handoff.kvOffset,
                count: lease.request.handoff.kvBytes
            ) { destination in
                try MappedKVOutputBackings(
                    destination: destination,
                    window: shape.window,
                    layers: shape.layers,
                    kvHeads: shape.kvHeads,
                    cacheTokens: lease.request.handoff.cacheTokens,
                    headDimension: shape.headDimension
                )
            }
            kvLayoutMilliseconds = monotonicMilliseconds() - kvPreparationStarted

            let options = MLPredictionOptions()
            options.outputBackings = kvBackings.outputBackings
            let predictionStarted = monotonicMilliseconds()
            let prediction: MLFeatureProvider
            do {
                prediction = try await lease.model.model.prediction(
                    from: provider,
                    options: options
                )
            } catch {
                throw SidecarError.kvConversion(
                    "CoreML could not write K/V outputs into shared memory: \(error)")
            }
            predictionMilliseconds = monotonicMilliseconds() - predictionStarted
            try throwIfCancelled(lease.ticket)

            let kvValidationStarted = monotonicMilliseconds()
            try kvBackings.validateAndClearInactive(
                prediction: prediction,
                activeTokens: lease.request.activeTokens
            )
            kvLayoutMilliseconds += monotonicMilliseconds() - kvValidationStarted
            try throwIfCancelled(lease.ticket)

            let logitsStarted = monotonicMilliseconds()
            guard let logitsArray = prediction.featureValue(for: "logits")?.multiArrayValue else {
                throw SidecarError.invalid("CoreML prediction has no logits output")
            }
            try lease.publication.withMutableBytes(
                offset: lease.request.handoff.logitsOffset,
                count: lease.request.handoff.logitsBytes
            ) { destination in
                try copyActiveLogitsToSharedMemory(
                    logitsArray,
                    to: destination,
                    activeTokens: lease.request.activeTokens,
                    window: shape.window,
                    vocabularySize: shape.vocabularySize
                )
            }
            logitsCopyMilliseconds = monotonicMilliseconds() - logitsStarted
            try throwIfCancelled(lease.ticket)

            let integrityStarted = monotonicMilliseconds()
            let digest = try lease.publication.finish()
            integrityMilliseconds = monotonicMilliseconds() - integrityStarted
            let wasCancelled = executions.finish(lease.ticket)
            if wasCancelled { throw SidecarError.cancelled }
            let timing = StageTimings(
                executionID: lease.request.requestID,
                readinessMilliseconds: lease.model.readinessMilliseconds,
                predictionMilliseconds: predictionMilliseconds,
                kvLayoutMilliseconds: kvLayoutMilliseconds,
                logitsCopyMilliseconds: logitsCopyMilliseconds,
                integrityMilliseconds: integrityMilliseconds,
                totalMilliseconds: monotonicMilliseconds() - executionStarted,
                cancelled: false
            )
            timings.record(timing)
            return ExecutionResult(digest: digest)
        } catch {
            let wasCancelled = executions.finish(lease.ticket)
            let timing = StageTimings(
                executionID: lease.request.requestID,
                readinessMilliseconds: lease.model.readinessMilliseconds,
                predictionMilliseconds: predictionMilliseconds,
                kvLayoutMilliseconds: kvLayoutMilliseconds,
                logitsCopyMilliseconds: logitsCopyMilliseconds,
                integrityMilliseconds: integrityMilliseconds,
                totalMilliseconds: monotonicMilliseconds() - executionStarted,
                cancelled: wasCancelled || error as? SidecarError == .cancelled
            )
            timings.record(timing)
            throw error
        }
    }

    func abort(executionID: String) -> Bool {
        executions.abort(executionID: executionID)
    }

    func timing(executionID: String) -> StageTimings? {
        timings.read(executionID: executionID)
    }

    func shutdown() {
        executions.abortAll()
    }
}

@main
private struct Main {
    static func main() async {
        do {
            let arguments = try parseArguments()
            let connection = try UnixConnection(path: arguments.socket)
            defer { connection.close() }

            // No CoreML stage object exists before this exchange. A version, nonce, or
            // engine mismatch ends the connection before any request can acquire
            // provenance or health accounting in the host.
            try connection.writeJSON([
                "type": "HELLO",
                "protocol": sidecarProtocolVersion,
                "engine": ["name": sidecarEngine.name, "version": sidecarEngine.version],
                "nonce": arguments.nonce,
                "max_frame_bytes": defaultMaxFrameBytes,
            ])
            let acceptedFrameBytes = try negotiateHelloAck(
                connection.readFrame(), nonce: arguments.nonce)
            connection.maxFrameBytes = acceptedFrameBytes
            let stage = CoreMLStage()

            while true {
                let frame: Data
                do {
                    frame = try connection.readFrame()
                } catch SidecarError.io(let message) where message == "eof" {
                    return
                }
                do {
                    switch try parseCommand(frame) {
                    case .install(let request):
                        let installed = try await stage.install(request)
                        try connection.writeJSON([
                            "type": "INSTALLED",
                            "request_id": request.requestID,
                            "model_ref": request.modelRef,
                            "readiness_ms": installed.readinessMilliseconds,
                            "engine": [
                                "name": sidecarEngine.name, "version": sidecarEngine.version,
                            ],
                        ])
                    case .execute(let request):
                        let lease = try stage.reserve(request)
                        Task.detached {
                            do {
                                let result = try await stage.execute(lease)
                                try connection.writeJSON([
                                    "type": "EXECUTED",
                                    "request_id": request.requestID,
                                    "execution_id": request.requestID,
                                    "model_ref": request.modelRef,
                                    "active_tokens": request.activeTokens,
                                    "logits_bytes": request.handoff.logitsBytes,
                                    "kv_bytes": request.handoff.kvBytes,
                                    "kv_layout": [
                                        "kind": "f16_le",
                                        "order": [
                                            "layer", "key_or_value", "head", "position",
                                            "dimension",
                                        ],
                                        "layers": lease.model.shape.layers,
                                        "key_value_count": 2,
                                        "kv_heads": lease.model.shape.kvHeads,
                                        "cache_tokens": request.handoff.cacheTokens,
                                        "active_tokens": request.activeTokens,
                                        "head_dimension": lease.model.shape.headDimension,
                                    ],
                                    "handoff": [
                                        "kind": request.handoff.kind,
                                        "path": request.handoff.path,
                                        "capacity_bytes": request.handoff.capacityBytes,
                                        "generation": request.handoff.generation,
                                        "logits_offset": request.handoff.logitsOffset,
                                        "logits_bytes": request.handoff.logitsBytes,
                                        "kv_offset": request.handoff.kvOffset,
                                        "kv_bytes": request.handoff.kvBytes,
                                        "sha256": result.digest,
                                    ],
                                ])
                            } catch {
                                try? connection.writeJSON(
                                    errorResponse(error, requestID: request.requestID))
                            }
                        }
                    case .abort(let request):
                        try connection.writeJSON([
                            "type": "ABORTED",
                            "request_id": request.requestID,
                            "execution_id": request.executionID,
                            "was_active": stage.abort(executionID: request.executionID),
                            "cancellation_owner": "sidecar",
                        ])
                    case .timingReadback(let request):
                        guard let timing = stage.timing(executionID: request.executionID) else {
                            throw SidecarError.notFound(request.executionID)
                        }
                        try connection.writeJSON(
                            timingResponse(requestID: request.requestID, timing: timing))
                    case .shutdown(let request):
                        stage.shutdown()
                        try connection.writeJSON([
                            "type": "SHUTDOWN_ACK", "request_id": request.requestID,
                        ])
                        return
                    }
                } catch {
                    try connection.writeJSON(errorResponse(error, requestID: requestID(in: frame)))
                }
            }
        } catch {
            fputs("ane-prefill-sidecar: \(error)\n", stderr)
            Foundation.exit(1)
        }
    }
}

private func validateFixedWindowRequest(_ request: ExecuteRequest, shape: ModelShape) throws {
    guard request.inputIDs.count == shape.window, request.attentionMask.count == shape.window else {
        throw SidecarError.invalid(
            "EXECUTE inputs must be padded by the module to the installed fixed window")
    }
    guard request.activeTokens <= shape.window else {
        throw SidecarError.invalid("active_tokens exceeds the installed fixed window")
    }
    for index in 0..<shape.window {
        let expectedMask: Int32 = index < request.activeTokens ? 1 : 0
        guard request.attentionMask[index] == expectedMask else {
            throw SidecarError.invalid(
                "EXECUTE must use right padding: active tokens first and no padded token in the decode cache"
            )
        }
    }
}

private func validateSharedMemoryHandoff(
    _ handoff: SharedMemoryHandoffDescriptor,
    request: ExecuteRequest,
    shape: ModelShape
) throws {
    let expectedLogitsBytes = shape.vocabularySize * MemoryLayout<Float>.size
    let expectedKVBytes =
        shape.layers * 2 * shape.kvHeads * handoff.cacheTokens * shape.headDimension
        * MemoryLayout<UInt16>.size
    guard handoff.logitsOffset == sharedMemoryHeaderBytes,
        handoff.logitsBytes == expectedLogitsBytes,
        handoff.kvOffset.isMultiple(of: 64),
        handoff.kvBytes == expectedKVBytes,
        handoff.cacheTokens >= request.activeTokens,
        handoff.capacityBytes == handoff.kvOffset + handoff.kvBytes
    else {
        throw SidecarError.invalid(
            "EXECUTE shared-memory layout does not match the installed model")
    }
}

private func copyActiveLogitsToSharedMemory(
    _ array: MLMultiArray,
    to destination: UnsafeMutableRawBufferPointer,
    activeTokens: Int,
    window: Int,
    vocabularySize: Int
) throws {
    let shape = array.shape.map(\.intValue)
    let strides = array.strides.map(\.intValue)
    let sequenceAxes = shape.indices.filter { shape[$0] == window }
    let vocabularyAxes = shape.indices.filter { shape[$0] == vocabularySize }
    guard vocabularyAxes.count == 1,
        let vocabularyAxis = vocabularyAxes.first,
        sequenceAxes.count <= 1,
        sequenceAxes.first != vocabularyAxis,
        strides.count == shape.count,
        strides.allSatisfy({ $0 > 0 }),
        shape.indices.allSatisfy({ axis in
            axis == sequenceAxes.first || axis == vocabularyAxis || shape[axis] == 1
        }),
        destination.count == vocabularySize * MemoryLayout<UInt32>.size
    else {
        throw SidecarError.invalid(
            "logits output shape \(shape) is incompatible with the installed model")
    }
    let storageCount = requiredStorageCount(shape: shape, strides: strides)
    var index = [Int](repeating: 0, count: shape.count)
    if let sequenceAxis = sequenceAxes.first {
        index[sequenceAxis] = activeTokens - 1
    }
    let target = destination.bindMemory(to: UInt32.self)
    for vocabularyIndex in 0..<vocabularySize {
        index[vocabularyAxis] = vocabularyIndex
        let offset = zip(index, strides).reduce(0) { $0 + $1.0 * $1.1 }
        let value: Float
        switch array.dataType {
        case .float16:
            let source = array.dataPointer.bindMemory(to: UInt16.self, capacity: storageCount)
            value = Float(Float16(bitPattern: source[offset]))
        case .float32:
            let source = array.dataPointer.bindMemory(to: Float.self, capacity: storageCount)
            value = source[offset]
        case .double:
            let source = array.dataPointer.bindMemory(to: Double.self, capacity: storageCount)
            value = Float(source[offset])
        default:
            throw SidecarError.invalid(
                "unsupported CoreML logits data type \(array.dataType.rawValue)")
        }
        target[vocabularyIndex] = value.bitPattern.littleEndian
    }
}

private func throwIfCancelled(_ ticket: ExecutionTicket) throws {
    if ticket.isCancelled { throw SidecarError.cancelled }
}

private func featureProvider(for request: ExecuteRequest) throws -> MLDictionaryFeatureProvider {
    let ids = try int32Array(request.inputIDs)
    let mask = try int32Array(request.attentionMask)
    return try MLDictionaryFeatureProvider(dictionary: ["input_ids": ids, "attention_mask": mask])
}

private func int32Array(_ values: [Int32]) throws -> MLMultiArray {
    let array = try MLMultiArray(shape: [1, NSNumber(value: values.count)], dataType: .int32)
    let target = array.dataPointer.bindMemory(to: Int32.self, capacity: values.count)
    for (index, value) in values.enumerated() {
        target[index] = value
    }
    return array
}

private func requiredStorageCount(shape: [Int], strides: [Int]) -> Int {
    zip(shape, strides).reduce(1) { partial, pair in partial + (pair.0 - 1) * pair.1 }
}

private func normalizedDigest(_ raw: String) -> String {
    raw.lowercased().hasPrefix("sha256:")
        ? String(raw.dropFirst("sha256:".count)).lowercased() : raw.lowercased()
}

private func verifyDigest(at url: URL, expected: String) throws {
    guard FileManager.default.fileExists(atPath: url.path) else {
        throw SidecarError.notFound(url.path)
    }
    let actual = try sha256(of: url)
    guard actual == normalizedDigest(expected) else {
        throw SidecarError.artifactMismatch
    }
}

private func sha256(of url: URL) throws -> String {
    var isDirectory: ObjCBool = false
    guard FileManager.default.fileExists(atPath: url.path, isDirectory: &isDirectory) else {
        throw SidecarError.notFound(url.path)
    }
    var hasher = SHA256()
    if isDirectory.boolValue {
        let relativePaths = try FileManager.default.subpathsOfDirectory(atPath: url.path).sorted()
        for relativePath in relativePaths {
            let child = url.appendingPathComponent(relativePath)
            var childIsDirectory: ObjCBool = false
            guard
                FileManager.default.fileExists(atPath: child.path, isDirectory: &childIsDirectory),
                !childIsDirectory.boolValue
            else {
                continue
            }
            hasher.update(data: Data(relativePath.utf8))
            hasher.update(data: Data([0]))
            hasher.update(data: try Data(contentsOf: child))
        }
    } else {
        hasher.update(data: try Data(contentsOf: url))
    }
    return hasher.finalize().map { String(format: "%02x", $0) }.joined()
}

private func timingResponse(requestID: String, timing: StageTimings) -> [String: Any] {
    var response: [String: Any] = [
        "type": "TIMING",
        "request_id": requestID,
        "execution_id": timing.executionID,
        "prediction_ms": timing.predictionMilliseconds,
        "kv_layout_ms": timing.kvLayoutMilliseconds,
        "logits_copy_ms": timing.logitsCopyMilliseconds,
        "integrity_ms": timing.integrityMilliseconds,
        "total_ms": timing.totalMilliseconds,
        "cancelled": timing.cancelled,
    ]
    if let readinessMilliseconds = timing.readinessMilliseconds {
        response["readiness_ms"] = readinessMilliseconds
    }
    return response
}

private func errorResponse(_ error: Error, requestID: String?) -> [String: Any] {
    let typed = error as? SidecarError
    var response: [String: Any] = [
        "type": "ERROR",
        "code": typed?.code ?? "internal_failure",
        "message": typed?.description ?? String(describing: error),
    ]
    if let requestID { response["request_id"] = requestID }
    return response
}

private func requestID(in frame: Data) -> String? {
    guard let object = try? JSONSerialization.jsonObject(with: frame) as? [String: Any] else {
        return nil
    }
    return object["request_id"] as? String
}

private func parseArguments() throws -> Arguments {
    var socket: String?
    var nonce: String?
    var arguments = CommandLine.arguments.dropFirst().makeIterator()
    while let argument = arguments.next() {
        switch argument {
        case "--socket": socket = arguments.next()
        case "--nonce": nonce = arguments.next()
        default: throw SidecarError.invalid("unknown argument \(argument)")
        }
    }
    guard let socket, !socket.isEmpty, let nonce, !nonce.isEmpty else {
        throw SidecarError.invalid(
            "usage: ane-prefill-sidecar --socket <unix socket> --nonce <nonce>")
    }
    return Arguments(socket: socket, nonce: nonce)
}

private final class UnixConnection: @unchecked Sendable {
    private let fd: Int32
    private let writeLock = NSLock()
    var maxFrameBytes: Int = defaultMaxFrameBytes

    init(path: String) throws {
        fd = try connectUnixSocket(path: path)
    }

    func close() {
        Darwin.close(fd)
    }

    func readFrame() throws -> Data {
        let lengthBytes = try readExactly(fd: fd, count: 4)
        let length = lengthBytes.withUnsafeBytes { raw -> UInt32 in
            let bytes = raw.bindMemory(to: UInt8.self)
            return UInt32(bytes[0]) | (UInt32(bytes[1]) << 8) | (UInt32(bytes[2]) << 16)
                | (UInt32(bytes[3]) << 24)
        }
        guard length <= UInt32(maxFrameBytes) else {
            throw SidecarError.io("frame exceeds negotiated maximum")
        }
        return try readExactly(fd: fd, count: Int(length))
    }

    func writeJSON(_ object: [String: Any]) throws {
        let data = try JSONSerialization.data(withJSONObject: object, options: [])
        try writeFrame(data)
    }

    func writeFrame(_ data: Data) throws {
        guard data.count <= maxFrameBytes else {
            throw SidecarError.io("response frame exceeds negotiated maximum")
        }
        writeLock.lock()
        defer { writeLock.unlock() }
        var length = UInt32(data.count).littleEndian
        try withUnsafeBytes(of: &length) { bytes in try writeExactly(fd: fd, bytes: bytes) }
        try data.withUnsafeBytes { bytes in try writeExactly(fd: fd, bytes: bytes) }
    }
}

private func readExactly(fd: Int32, count: Int) throws -> Data {
    var data = Data(count: count)
    var offset = 0
    while offset < count {
        let readCount = data.withUnsafeMutableBytes { raw -> Int in
            Darwin.read(fd, raw.baseAddress!.advanced(by: offset), count - offset)
        }
        if readCount == 0 {
            if offset == 0 { throw SidecarError.io("eof") }
            throw SidecarError.io("unexpected eof")
        }
        if readCount < 0 {
            if errno == EINTR { continue }
            throw SidecarError.io("socket read failed: \(String(cString: strerror(errno)))")
        }
        offset += readCount
    }
    return data
}

private func writeExactly(fd: Int32, bytes: UnsafeRawBufferPointer) throws {
    var offset = 0
    while offset < bytes.count {
        let written = Darwin.write(
            fd, bytes.baseAddress!.advanced(by: offset), bytes.count - offset)
        if written < 0 {
            if errno == EINTR { continue }
            throw SidecarError.io("socket write failed: \(String(cString: strerror(errno)))")
        }
        offset += written
    }
}

private func connectUnixSocket(path: String) throws -> Int32 {
    let fd = socket(AF_UNIX, SOCK_STREAM, 0)
    guard fd >= 0 else {
        throw SidecarError.io("socket creation failed: \(String(cString: strerror(errno)))")
    }
    var address = sockaddr_un()
    address.sun_family = sa_family_t(AF_UNIX)
    let maximumPath = MemoryLayout.size(ofValue: address.sun_path)
    let bytes = Array(path.utf8)
    guard bytes.count < maximumPath else {
        Darwin.close(fd)
        throw SidecarError.invalid("unix socket path is too long")
    }
    withUnsafeMutableBytes(of: &address.sun_path) { buffer in
        for (index, byte) in bytes.enumerated() { buffer[index] = byte }
    }
    let status = withUnsafePointer(to: &address) { pointer in
        pointer.withMemoryRebound(to: sockaddr.self, capacity: 1) {
            Darwin.connect(fd, $0, socklen_t(MemoryLayout<sockaddr_un>.size))
        }
    }
    guard status == 0 else {
        let message = String(cString: strerror(errno))
        Darwin.close(fd)
        throw SidecarError.io("socket connection failed: \(message)")
    }
    return fd
}
