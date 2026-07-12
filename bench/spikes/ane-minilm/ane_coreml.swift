import CoreML
import Foundation

private enum CLIError: LocalizedError {
    case usage(String)
    case invalid(String)
    case missingOutputName([String])
    case unsupportedOutputType(String)
    case unsupportedInputType(String)

    var errorDescription: String? {
        switch self {
        case .usage(let message), .invalid(let message):
            return message
        case .missingOutputName(let outputs):
            return "could not determine output name; model outputs are: \(outputs.joined(separator: ", "))"
        case .unsupportedOutputType(let message), .unsupportedInputType(let message):
            return message
        }
    }
}

private struct CompileOptions {
    let sourceModel: String
    let outputModelC: String
}

private enum PoolingMode: String {
    case graph
    case mean
    case cls
}

private struct EmbedOptions {
    let modelPath: String
    let inputPath: String
    let outputPath: String
    let statsPath: String?
    let placementPath: String?
    let batchSize: Int
    let outputName: String?
    let computeUnits: MLComputeUnits
    let poolingMode: PoolingMode
    let pairScoring: Bool
}

private struct TokenizedRow: Decodable {
    let id: String
    let input_ids: [Int]
    let attention_mask: [Int]
    let token_count: Int?
}

private struct VectorRow: Encodable {
    let id: String
    let vec: [Float]
}

private struct ScoreRow: Encodable {
    let id: String
    let score: Float
}

private struct RunStats: Encodable {
    let model_path: String
    let output_name: String
    let bucket: Int
    let items: Int
    let input_tokens: Int
    let output_dimension: Int
    let batch_size: Int
    let cold_load_s: Double
    let infer_wall_s: Double
    let docs_per_s: Double
    let tokens_per_s: Double
    let compute_units: String
    let pooling_mode: String
    let pair_scoring: Bool
    let request_latency_p50_ms: Double?
    let request_latency_p95_ms: Double?
}

private struct PlacementOperation: Encodable {
    let operator_name: String
    let preferred_device: String
    let supported_devices: [String]
}

private struct PlacementSummary: Encodable {
    let total_ops: Int
    let dispatchable_ops: Int
    let preferred_device_counts: [String: Int]
    let preferred_device_share: [String: Double]
    let dispatchable_device_counts: [String: Int]
    let dispatchable_device_share: [String: Double]
    let non_neural_engine_operations: [PlacementOperation]
    let unknown_operations: [PlacementOperation]
}

private struct PlacementReport: Encodable {
    let model_path: String
    let model_kind: String
    let compute_units: String
    let summary: PlacementSummary
}

private enum Command {
    case compile(CompileOptions)
    case embed(EmbedOptions)
}

@main
private enum ANEMiniLMCLI {
    static func main() async {
        do {
            let command = try parseCommand(arguments: Array(CommandLine.arguments.dropFirst()))
            switch command {
            case .compile(let options):
                try await compileModel(options)
            case .embed(let options):
                try await embedRows(options)
            }
        } catch {
            fputs("error: \(error.localizedDescription)\n", stderr)
            exit(1)
        }
    }
}

private func parseCommand(arguments: [String]) throws -> Command {
    guard let subcommand = arguments.first else {
        throw CLIError.usage(usageText)
    }

    switch subcommand {
    case "compile":
        return .compile(try parseCompileOptions(arguments: Array(arguments.dropFirst())))
    case "embed", "run":
        return .embed(try parseEmbedOptions(arguments: Array(arguments.dropFirst())))
    case "help", "--help", "-h":
        throw CLIError.usage(usageText)
    default:
        throw CLIError.usage("unknown subcommand \(subcommand)\n\n\(usageText)")
    }
}

private func parseCompileOptions(arguments: [String]) throws -> CompileOptions {
    var sourceModel: String?
    var outputModelC: String?

    var index = 0
    while index < arguments.count {
        let token = arguments[index]
        switch token {
        case "--model":
            sourceModel = try requireValue(arguments, index: &index, flag: token)
        case "--out":
            outputModelC = try requireValue(arguments, index: &index, flag: token)
        case "--help", "-h":
            throw CLIError.usage(usageText)
        default:
            throw CLIError.invalid("unexpected compile argument: \(token)")
        }
        index += 1
    }

    guard let sourceModel, let outputModelC else {
        throw CLIError.usage("compile requires --model <.mlpackage|.mlmodel> and --out <.mlmodelc>\n\n\(usageText)")
    }

    return CompileOptions(sourceModel: sourceModel, outputModelC: outputModelC)
}

private func parseEmbedOptions(arguments: [String]) throws -> EmbedOptions {
    var modelPath: String?
    var inputPath = "-"
    var outputPath = "-"
    var statsPath: String?
    var placementPath: String?
    var batchSize = 8
    var outputName: String?
    var computeUnits = MLComputeUnits.cpuAndNeuralEngine
    var poolingMode = PoolingMode.mean
    var pairScoring = false

    var index = 0
    while index < arguments.count {
        let token = arguments[index]
        switch token {
        case "--model":
            modelPath = try requireValue(arguments, index: &index, flag: token)
        case "--input":
            inputPath = try requireValue(arguments, index: &index, flag: token)
        case "--output":
            outputPath = try requireValue(arguments, index: &index, flag: token)
        case "--stats-out":
            statsPath = try requireValue(arguments, index: &index, flag: token)
        case "--placement-out":
            placementPath = try requireValue(arguments, index: &index, flag: token)
        case "--batch-size":
            let value = try requireValue(arguments, index: &index, flag: token)
            guard let parsed = Int(value), parsed > 0 else {
                throw CLIError.invalid("--batch-size must be a positive integer")
            }
            batchSize = parsed
        case "--output-name":
            outputName = try requireValue(arguments, index: &index, flag: token)
        case "--compute-units":
            let raw = try requireValue(arguments, index: &index, flag: token)
            computeUnits = try parseComputeUnits(raw)
        case "--pooling":
            let raw = try requireValue(arguments, index: &index, flag: token)
            guard let parsed = PoolingMode(rawValue: raw) else {
                throw CLIError.invalid("--pooling must be graph, mean, or cls")
            }
            poolingMode = parsed
        case "--pair-scoring":
            pairScoring = true
        case "--help", "-h":
            throw CLIError.usage(usageText)
        default:
            throw CLIError.invalid("unexpected embed argument: \(token)")
        }
        index += 1
    }

    guard let modelPath else {
        throw CLIError.usage("embed requires --model <.mlmodelc>\n\n\(usageText)")
    }

    return EmbedOptions(
        modelPath: modelPath,
        inputPath: inputPath,
        outputPath: outputPath,
        statsPath: statsPath,
        placementPath: placementPath,
        batchSize: batchSize,
        outputName: outputName,
        computeUnits: computeUnits,
        poolingMode: pairScoring ? .graph : poolingMode,
        pairScoring: pairScoring
    )
}

private func requireValue(_ arguments: [String], index: inout Int, flag: String) throws -> String {
    let next = index + 1
    guard next < arguments.count else {
        throw CLIError.invalid("missing value for \(flag)")
    }
    index = next
    return arguments[next]
}

private func parseComputeUnits(_ raw: String) throws -> MLComputeUnits {
    switch raw {
    case "cpuAndNeuralEngine", "cpu-and-ne", "CPU_AND_NE":
        return .cpuAndNeuralEngine
    case "all", "ALL":
        return .all
    case "cpuAndGPU", "cpu-and-gpu", "CPU_AND_GPU":
        return .cpuAndGPU
    case "cpuOnly", "cpu-only", "CPU_ONLY":
        return .cpuOnly
    default:
        throw CLIError.invalid("unsupported compute units: \(raw)")
    }
}

private func compileModel(_ options: CompileOptions) async throws {
    let sourceURL = URL(fileURLWithPath: options.sourceModel).standardizedFileURL
    let outputURL = URL(fileURLWithPath: options.outputModelC).standardizedFileURL
    try ensureParentDirectory(for: outputURL)
    try removeIfExists(outputURL)

    let compiledTempURL = try await MLModel.compileModel(at: sourceURL)
    try FileManager.default.copyItem(at: compiledTempURL, to: outputURL)
    fputs("compiled \(sourceURL.path) -> \(outputURL.path)\n", stderr)
}

private func embedRows(_ options: EmbedOptions) async throws {
    let rows = try loadRows(from: options.inputPath)
    guard let firstRow = rows.first else {
        throw CLIError.invalid("input is empty")
    }
    let bucket = firstRow.input_ids.count
    guard bucket > 0 else {
        throw CLIError.invalid("bucket length must be positive")
    }
    for row in rows {
        guard row.input_ids.count == bucket, row.attention_mask.count == bucket else {
            throw CLIError.invalid("all rows must have the same input_ids and attention_mask length")
        }
    }

    let coldStarted = monotonicTime()
    let configuration = MLModelConfiguration()
    configuration.computeUnits = options.computeUnits
    if #available(macOS 14.4, *) {
        configuration.optimizationHints.reshapeFrequency = .infrequent
    }
    let modelURL = URL(fileURLWithPath: options.modelPath).standardizedFileURL
    let model = try MLModel(contentsOf: modelURL, configuration: configuration)
    let resolvedOutputName = try selectOutputName(model: model, requested: options.outputName)

    _ = try predictOne(model: model, row: firstRow)
    let coldLoadS = monotonicTime() - coldStarted

    let inferStarted = monotonicTime()
    var vectors: [VectorRow] = []
    var scores: [ScoreRow] = []
    vectors.reserveCapacity(rows.count)
    scores.reserveCapacity(rows.count)
    var inputTokens = 0
    var outputDimension = 0
    var requestLatenciesMS: [Double] = []
    let groups = options.pairScoring ? groupPairRows(rows) : [rows]

    for group in groups {
        let requestStarted = monotonicTime()
        for start in stride(from: 0, to: group.count, by: options.batchSize) {
            let batch = Array(group[start ..< min(start + options.batchSize, group.count)])
        let providers = try batch.map(featureProvider)
        let batchProvider = MLArrayBatchProvider(array: providers)
        let predictions = try model.predictions(fromBatch: batchProvider)

        for batchIndex in 0 ..< predictions.count {
            let row = batch[batchIndex]
            let output = predictions.features(at: batchIndex)
            guard let value = output.featureValue(for: resolvedOutputName)?.multiArrayValue else {
                throw CLIError.invalid("prediction output is missing feature \(resolvedOutputName)")
            }
            if options.pairScoring {
                scores.append(ScoreRow(id: row.id, score: try readScalar(value)))
                outputDimension = 1
            } else {
                let vector = try poolOutput(
                    value,
                    attentionMask: row.attention_mask,
                    mode: options.poolingMode
                )
                vectors.append(VectorRow(id: row.id, vec: vector))
                outputDimension = vector.count
            }
                inputTokens += row.attention_mask.reduce(0, +)
            }
        }
        if options.pairScoring {
            requestLatenciesMS.append((monotonicTime() - requestStarted) * 1_000)
        }
    }
    let inferWallS = monotonicTime() - inferStarted

    if options.pairScoring {
        try writeJSONLines(scores, to: options.outputPath)
    } else {
        try writeJSONLines(vectors, to: options.outputPath)
    }

    let stats = RunStats(
        model_path: modelURL.path,
        output_name: resolvedOutputName,
        bucket: bucket,
        items: rows.count,
        input_tokens: inputTokens,
        output_dimension: outputDimension,
        batch_size: options.batchSize,
        cold_load_s: coldLoadS,
        infer_wall_s: inferWallS,
        docs_per_s: inferWallS > 0 ? Double(rows.count) / inferWallS : 0,
        tokens_per_s: inferWallS > 0 ? Double(inputTokens) / inferWallS : 0,
        compute_units: computeUnitsLabel(options.computeUnits),
        pooling_mode: options.poolingMode.rawValue,
        pair_scoring: options.pairScoring,
        request_latency_p50_ms: percentile(requestLatenciesMS, fraction: 0.50),
        request_latency_p95_ms: percentile(requestLatenciesMS, fraction: 0.95)
    )

    if let statsPath = options.statsPath {
        try writeJSON(stats, to: statsPath)
    }
    if let placementPath = options.placementPath {
        let report = try await buildPlacementReport(modelURL: modelURL, configuration: configuration)
        try writeJSON(report, to: placementPath)
    }

    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    let statsData = try encoder.encode(stats)
    FileHandle.standardError.write(statsData)
    FileHandle.standardError.write(Data("\n".utf8))
}

private func selectOutputName(model: MLModel, requested: String?) throws -> String {
    let outputs = Array(model.modelDescription.outputDescriptionsByName.keys).sorted()
    if let requested {
        guard outputs.contains(requested) else {
            throw CLIError.invalid("requested output \(requested) is not present in model outputs: \(outputs)")
        }
        return requested
    }
    if outputs.count == 1, let only = outputs.first {
        return only
    }
    if outputs.contains("last_hidden_state") {
        return "last_hidden_state"
    }
    throw CLIError.missingOutputName(outputs)
}

private func loadRows(from path: String) throws -> [TokenizedRow] {
    let text: String
    if path == "-" {
        let data = FileHandle.standardInput.readDataToEndOfFile()
        text = String(decoding: data, as: UTF8.self)
    } else {
        text = try String(contentsOf: URL(fileURLWithPath: path), encoding: .utf8)
    }

    let decoder = JSONDecoder()
    var rows: [TokenizedRow] = []
    for (lineNumber, rawLine) in text.split(whereSeparator: \.isNewline).enumerated() {
        let trimmed = rawLine.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.isEmpty {
            continue
        }
        do {
            rows.append(try decoder.decode(TokenizedRow.self, from: Data(trimmed.utf8)))
        } catch {
            throw CLIError.invalid("invalid JSON on line \(lineNumber + 1): \(error)")
        }
    }
    return rows
}

private func featureProvider(for row: TokenizedRow) throws -> MLDictionaryFeatureProvider {
    let ids = try makeMultiArray(values: row.input_ids)
    let mask = try makeMultiArray(values: row.attention_mask)
    return try MLDictionaryFeatureProvider(dictionary: [
        "input_ids": MLFeatureValue(multiArray: ids),
        "attention_mask": MLFeatureValue(multiArray: mask),
    ])
}

private func makeMultiArray(values: [Int]) throws -> MLMultiArray {
    let array = try MLMultiArray(shape: [1, NSNumber(value: values.count)], dataType: .int32)
    try array.withUnsafeMutableBufferPointer(ofType: Int32.self) { buffer, _ in
        guard buffer.count == values.count else {
            throw CLIError.invalid("unexpected MLMultiArray backing length: \(buffer.count)")
        }
        for (index, value) in values.enumerated() {
            guard let cast = Int32(exactly: value) else {
                throw CLIError.unsupportedInputType("input value \(value) does not fit in Int32")
            }
            buffer[index] = cast
        }
    }
    return array
}

private func predictOne(model: MLModel, row: TokenizedRow) throws -> MLFeatureProvider {
    let provider = try featureProvider(for: row)
    return try model.prediction(from: provider)
}

private func poolOutput(
    _ output: MLMultiArray,
    attentionMask: [Int],
    mode: PoolingMode
) throws -> [Float] {
    let shape = output.shape.map { $0.intValue }
    if mode == .graph {
        guard shape.count == 1 || (shape.count == 2 && shape[0] == 1) else {
            throw CLIError.unsupportedOutputType(
                "graph-pooled output must have shape [hidden] or [1, hidden], got \(shape)"
            )
        }
        var vector: [Float] = []
        try readFloatArray(output) { vector = $0 }
        return vector
    }

    guard shape.count == 3, shape[0] == 1 else {
        throw CLIError.unsupportedOutputType("expected output shape [1, bucket, hidden], got \(shape)")
    }
    let bucket = shape[1]
    let hidden = shape[2]
    guard attentionMask.count == bucket else {
        throw CLIError.unsupportedOutputType(
            "attention mask length \(attentionMask.count) does not match output bucket \(bucket)"
        )
    }

    var pooled = [Float](repeating: 0, count: hidden)
    try readFloatArray(output) { buffer in
        guard buffer.count == bucket * hidden else {
            throw CLIError.unsupportedOutputType(
                "output element count \(buffer.count) does not match bucket*hidden \(bucket * hidden)"
            )
        }
        if mode == .cls {
            pooled = Array(buffer.prefix(hidden))
            return
        }

        var denominator: Float = 0
        for tokenIndex in 0 ..< bucket {
            let weight = Float(attentionMask[tokenIndex])
            guard weight >= 0 else {
                throw CLIError.invalid("attention masks must be non-negative")
            }
            if weight == 0 {
                continue
            }
            denominator += weight
            let base = tokenIndex * hidden
            for hiddenIndex in 0 ..< hidden {
                pooled[hiddenIndex] += buffer[base + hiddenIndex] * weight
            }
        }
        guard denominator > 0 else {
            throw CLIError.invalid("attention mask is all zeros")
        }
        for index in pooled.indices {
            pooled[index] /= denominator
        }
    }

    var normSquared: Float = 0
    for value in pooled {
        normSquared += value * value
    }
    let norm = sqrt(normSquared)
    if norm > 0 {
        for index in pooled.indices {
            pooled[index] /= norm
        }
    }
    return pooled
}

private func groupPairRows(_ rows: [TokenizedRow]) -> [[TokenizedRow]] {
    var groups: [[TokenizedRow]] = []
    var currentKey: String?
    for row in rows {
        let key = row.id.components(separatedBy: "::").first ?? row.id
        if key != currentKey {
            groups.append([])
            currentKey = key
        }
        groups[groups.count - 1].append(row)
    }
    return groups
}

private func percentile(_ values: [Double], fraction: Double) -> Double? {
    guard !values.isEmpty else {
        return nil
    }
    let sorted = values.sorted()
    let index = Int(ceil(fraction * Double(sorted.count))) - 1
    return sorted[max(0, min(index, sorted.count - 1))]
}

private func readScalar(_ output: MLMultiArray) throws -> Float {
    var values: [Float] = []
    try readFloatArray(output) { values = $0 }
    guard values.count == 1, let score = values.first else {
        throw CLIError.unsupportedOutputType(
            "pair-scoring output must contain one scalar, got shape \(output.shape)"
        )
    }
    return score
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
        throw CLIError.unsupportedOutputType("unsupported MLMultiArray output data type: \(array.dataType.rawValue)")
    }
}

private func writeJSONLines<T: Encodable>(_ values: [T], to path: String) throws {
    let encoder = JSONEncoder()
    var data = Data()
    for value in values {
        data.append(try encoder.encode(value))
        data.append(Data([0x0A]))
    }

    if path == "-" {
        FileHandle.standardOutput.write(data)
        return
    }

    let url = URL(fileURLWithPath: path)
    try ensureParentDirectory(for: url)
    try data.write(to: url)
}

private func writeJSON<T: Encodable>(_ value: T, to path: String) throws {
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    let data = try encoder.encode(value)
    let url = URL(fileURLWithPath: path)
    try ensureParentDirectory(for: url)
    try data.write(to: url)
}

@available(macOS 14.4, *)
private func buildPlacementReport(modelURL: URL, configuration: MLModelConfiguration) async throws -> PlacementReport {
    let plan = try await MLComputePlan.load(contentsOf: modelURL, configuration: configuration)
    switch plan.modelStructure {
    case .program(let program):
        let function = program.functions["main"] ?? program.functions.values.sorted { $0.inputs.count < $1.inputs.count }.first
        guard let function else {
            throw CLIError.invalid("model structure has no functions")
        }
        var operations: [PlacementOperation] = []
        collectProgramOperations(plan: plan, block: function.block, into: &operations)
        let summary = summarizePlacement(operations)
        return PlacementReport(
            model_path: modelURL.path,
            model_kind: "program",
            compute_units: computeUnitsLabel(configuration.computeUnits),
            summary: summary
        )
    case .neuralNetwork(let network):
        var operations: [PlacementOperation] = []
        for layer in network.layers {
            if let usage = plan.deviceUsage(for: layer) {
                operations.append(
                    PlacementOperation(
                        operator_name: layer.type,
                        preferred_device: deviceLabel(usage.preferred),
                        supported_devices: usage.supported.map(deviceLabel).sorted()
                    )
                )
            }
        }
        let summary = summarizePlacement(operations)
        return PlacementReport(
            model_path: modelURL.path,
            model_kind: "neuralNetwork",
            compute_units: computeUnitsLabel(configuration.computeUnits),
            summary: summary
        )
    case .pipeline(let pipeline):
        throw CLIError.invalid("pipeline placement reporting is not implemented for \(pipeline.subModelNames)")
    case .unsupported:
        throw CLIError.invalid("model structure is unsupported for placement reporting")
    @unknown default:
        throw CLIError.invalid("model structure has an unknown future case")
    }
}

@available(macOS 14.4, *)
private func collectProgramOperations(
    plan: MLComputePlan,
    block: MLModelStructure.Program.Block,
    into operations: inout [PlacementOperation]
) {
    for operation in block.operations {
        if let usage = plan.deviceUsage(for: operation) {
            operations.append(
                PlacementOperation(
                    operator_name: operation.operatorName,
                    preferred_device: deviceLabel(usage.preferred),
                    supported_devices: usage.supported.map(deviceLabel).sorted()
                )
            )
        } else {
            operations.append(
                PlacementOperation(
                    operator_name: operation.operatorName,
                    preferred_device: "unknown",
                    supported_devices: []
                )
            )
        }
        for nestedBlock in operation.blocks {
            collectProgramOperations(plan: plan, block: nestedBlock, into: &operations)
        }
    }
}

private func summarizePlacement(_ operations: [PlacementOperation]) -> PlacementSummary {
    var counts: [String: Int] = [:]
    var dispatchableCounts: [String: Int] = [:]
    var nonNeuralEngine: [PlacementOperation] = []
    var unknownOperations: [PlacementOperation] = []

    for operation in operations {
        counts[operation.preferred_device, default: 0] += 1
        if operation.preferred_device == "unknown" {
            unknownOperations.append(operation)
            continue
        }
        dispatchableCounts[operation.preferred_device, default: 0] += 1
        if operation.preferred_device != "neuralEngine" {
            nonNeuralEngine.append(operation)
        }
    }

    let total = max(operations.count, 1)
    let dispatchableTotal = max(dispatchableCounts.values.reduce(0, +), 1)
    let shares = Dictionary(uniqueKeysWithValues: counts.map { key, value in
        (key, Double(value) / Double(total))
    })
    let dispatchableShares = Dictionary(uniqueKeysWithValues: dispatchableCounts.map { key, value in
        (key, Double(value) / Double(dispatchableTotal))
    })

    return PlacementSummary(
        total_ops: operations.count,
        dispatchable_ops: dispatchableCounts.values.reduce(0, +),
        preferred_device_counts: counts,
        preferred_device_share: shares,
        dispatchable_device_counts: dispatchableCounts,
        dispatchable_device_share: dispatchableShares,
        non_neural_engine_operations: nonNeuralEngine,
        unknown_operations: unknownOperations
    )
}

private func deviceLabel(_ device: MLComputeDevice) -> String {
    switch device {
    case .cpu:
        return "cpu"
    case .gpu:
        return "gpu"
    case .neuralEngine:
        return "neuralEngine"
    @unknown default:
        return "unknown"
    }
}

private func computeUnitsLabel(_ units: MLComputeUnits) -> String {
    switch units {
    case .cpuAndNeuralEngine:
        return "cpuAndNeuralEngine"
    case .cpuAndGPU:
        return "cpuAndGPU"
    case .cpuOnly:
        return "cpuOnly"
    case .all:
        return "all"
    @unknown default:
        return "unknown"
    }
}

private func ensureParentDirectory(for url: URL) throws {
    let parent = url.deletingLastPathComponent()
    try FileManager.default.createDirectory(at: parent, withIntermediateDirectories: true, attributes: nil)
}

private func removeIfExists(_ url: URL) throws {
    if FileManager.default.fileExists(atPath: url.path) {
        try FileManager.default.removeItem(at: url)
    }
}

private func monotonicTime() -> Double {
    ProcessInfo.processInfo.systemUptime
}

private let usageText = """
Usage:
  ane-coreml compile --model <model.mlpackage> --out <model.mlmodelc>
  ane-coreml run --model <model.mlmodelc> [--input rows.jsonl|-] [--output results.jsonl|-]
                 [--stats-out stats.json] [--placement-out placement.json]
                 [--batch-size N] [--output-name name]
                 [--pooling graph|mean|cls] [--pair-scoring]
                 [--compute-units cpuAndNeuralEngine|all|cpuAndGPU|cpuOnly]

Commands:
  compile   Compile a .mlpackage/.mlmodel into a permanent .mlmodelc directory.
  run       Run fixed-bucket pretokenized IDs. Graph pooling consumes an embedding
            already pooled in Core ML; mean and cls pool encoder hidden states.
            Pair scoring emits one {id, score} row per tokenized query/document pair.
  embed     Backward-compatible alias for run; its default pooling mode is mean.
"""
