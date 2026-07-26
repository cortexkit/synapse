import CoreML
import Foundation

private enum CLIError: LocalizedError {
    case invalid(String)

    var errorDescription: String? {
        switch self {
        case .invalid(let message):
            return message
        }
    }
}

private struct TokenizedRow: Codable {
    let id: String
    let input_ids: [Int]
    let attention_mask: [Int]
}

private struct RunStats: Encodable {
    let model_path: String
    let output_name: String
    let window: Int
    let last_k: Int
    let compute_units: String
    let prompts: Int
    let calls: Int
    let warmups: Int
    let target_duration_s: Double?
    let cold_load_s: Double
    let warmup_s: Double
    let infer_wall_s: Double
    let request_latency_p50_ms: Double
    let request_latency_p95_ms: Double
    let requests_per_s: Double
    let output_checksum: Double
    let compiled_size_bytes: Int64
}

private struct CompileStats: Encodable {
    let source_model: String
    let compiled_model: String
    let compile_s: Double
    let source_size_bytes: Int64
    let compiled_size_bytes: Int64
}

private struct ArgmaxRow: Encodable {
    let id: String
    let argmax: Int
    let token_ids: [Int]?
}

private struct ServeRequest: Decodable {
    let token_ids: [Int]
}

private struct ServeResponse: Encodable {
    let draft_ids: [Int]
    let compute_wall_s: Double
}

private struct PlacementOperation: Encodable {
    let operator_index: Int
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
    let operations: [PlacementOperation]
}

@main
private enum Main {
    static func main() async {
        do {
            var arguments = Array(CommandLine.arguments.dropFirst())
            guard let command = arguments.first else {
                throw CLIError.invalid(usage)
            }
            arguments.removeFirst()
            switch command {
            case "compile":
                try await compile(arguments)
            case "run":
                try await run(arguments)
            case "predict":
                try await predict(arguments)
            case "serve":
                try await serve(arguments)
            default:
                throw CLIError.invalid("unknown command \(command)\n\(usage)")
            }
        } catch {
            FileHandle.standardError.write(Data("error: \(error.localizedDescription)\n".utf8))
            exit(1)
        }
    }
}

private func option(_ name: String, in arguments: [String], default defaultValue: String? = nil) throws -> String {
    guard let index = arguments.firstIndex(of: name) else {
        if let defaultValue {
            return defaultValue
        }
        throw CLIError.invalid("missing required option \(name)")
    }
    guard index + 1 < arguments.count else {
        throw CLIError.invalid("missing value for \(name)")
    }
    return arguments[index + 1]
}

private func compile(_ arguments: [String]) async throws {
    let source = URL(fileURLWithPath: try option("--model", in: arguments)).standardizedFileURL
    let destination = URL(fileURLWithPath: try option("--out", in: arguments)).standardizedFileURL
    try removeIfExists(destination)
    try FileManager.default.createDirectory(
        at: destination.deletingLastPathComponent(), withIntermediateDirectories: true
    )
    let started = monotonicTime()
    let temporary = try await MLModel.compileModel(at: source)
    let compileSeconds = monotonicTime() - started
    try FileManager.default.moveItem(at: temporary, to: destination)
    let stats = CompileStats(
        source_model: source.path,
        compiled_model: destination.path,
        compile_s: compileSeconds,
        source_size_bytes: directorySize(source),
        compiled_size_bytes: directorySize(destination)
    )
    try writeJSON(stats, to: try option("--stats", in: arguments, default: "-"))
}

private func run(_ arguments: [String]) async throws {
    let modelURL = URL(fileURLWithPath: try option("--model", in: arguments)).standardizedFileURL
    let inputPath = try option("--input", in: arguments)
    let statsPath = try option("--stats", in: arguments)
    let placementPath = try option("--placement", in: arguments, default: "")
    let computeUnits = try parseComputeUnits(try option("--compute-units", in: arguments, default: "CPU_AND_NE"))
    guard let calls = Int(try option("--calls", in: arguments, default: "200")), calls > 0 else {
        throw CLIError.invalid("--calls must be positive")
    }
    guard let warmupCount = Int(try option("--warmup", in: arguments, default: "20")), warmupCount >= 0 else {
        throw CLIError.invalid("--warmup must be nonnegative")
    }
    guard let targetDuration = Double(try option("--duration-s", in: arguments, default: "0")), targetDuration >= 0 else {
        throw CLIError.invalid("--duration-s must be nonnegative")
    }

    let rows = try loadRows(inputPath)
    guard let first = rows.first else {
        throw CLIError.invalid("input has no rows")
    }
    let window = first.input_ids.count
    for row in rows {
        guard row.input_ids.count == window, row.attention_mask.count == window else {
            throw CLIError.invalid("row \(row.id) does not match fixed window \(window)")
        }
    }
    let providers = try rows.map(featureProvider)
    let configuration = MLModelConfiguration()
    configuration.computeUnits = computeUnits
    let loadStarted = monotonicTime()
    let model = try MLModel(contentsOf: modelURL, configuration: configuration)
    let coldLoadSeconds = monotonicTime() - loadStarted
    let outputNames = Array(model.modelDescription.outputDescriptionsByName.keys)
    guard outputNames.count == 1, let outputName = outputNames.first else {
        throw CLIError.invalid("expected exactly one output, found \(outputNames)")
    }

    let warmupStarted = monotonicTime()
    if warmupCount > 0 {
        for index in 0 ..< warmupCount {
            _ = try await model.prediction(from: providers[index % providers.count])
        }
    }
    let warmupSeconds = monotonicTime() - warmupStarted

    var latencies: [Double] = []
    var checksum = 0.0
    let inferStarted = monotonicTime()
    var requestCount = 0
    if targetDuration > 0 {
        repeat {
            for provider in providers {
                if monotonicTime() - inferStarted >= targetDuration && requestCount > 0 {
                    break
                }
                let requestStarted = monotonicTime()
                let prediction = try await model.prediction(from: provider)
                latencies.append((monotonicTime() - requestStarted) * 1000.0)
                checksum += try outputChecksum(prediction, outputName: outputName)
                requestCount += 1
            }
        } while monotonicTime() - inferStarted < targetDuration
    } else {
        for _ in 0 ..< calls {
            for provider in providers {
                let requestStarted = monotonicTime()
                let prediction = try await model.prediction(from: provider)
                latencies.append((monotonicTime() - requestStarted) * 1000.0)
                checksum += try outputChecksum(prediction, outputName: outputName)
                requestCount += 1
            }
        }
    }
    let inferSeconds = monotonicTime() - inferStarted
    let stats = RunStats(
        model_path: modelURL.path,
        output_name: outputName,
        window: window,
        last_k: outputLastK(model: model, outputName: outputName),
        compute_units: computeUnitsLabel(computeUnits),
        prompts: rows.count,
        calls: requestCount,
        warmups: warmupCount,
        target_duration_s: targetDuration > 0 ? targetDuration : nil,
        cold_load_s: coldLoadSeconds,
        warmup_s: warmupSeconds,
        infer_wall_s: inferSeconds,
        request_latency_p50_ms: percentile(latencies, fraction: 0.50),
        request_latency_p95_ms: percentile(latencies, fraction: 0.95),
        requests_per_s: inferSeconds > 0 ? Double(requestCount) / inferSeconds : 0,
        output_checksum: checksum,
        compiled_size_bytes: directorySize(modelURL)
    )
    try writeJSON(stats, to: statsPath)
    if !placementPath.isEmpty {
        let report = try await buildPlacementReport(modelURL: modelURL, configuration: configuration, computeUnits: computeUnits)
        try writeJSON(report, to: placementPath)
    }
}

private func predict(_ arguments: [String]) async throws {
    let modelURL = URL(fileURLWithPath: try option("--model", in: arguments)).standardizedFileURL
    let inputPath = try option("--input", in: arguments)
    let outputPath = try option("--output", in: arguments)
    let computeUnits = try parseComputeUnits(try option("--compute-units", in: arguments, default: "CPU_AND_NE"))
    let rows = try loadRows(inputPath)
    let providers = try rows.map(featureProvider)
    let configuration = MLModelConfiguration()
    configuration.computeUnits = computeUnits
    let model = try MLModel(contentsOf: modelURL, configuration: configuration)
    let outputNames = Array(model.modelDescription.outputDescriptionsByName.keys)
    guard outputNames.count == 1, let outputName = outputNames.first else {
        throw CLIError.invalid("expected exactly one output, found \(outputNames)")
    }
    var outputs: [ArgmaxRow] = []
    outputs.reserveCapacity(rows.count)
    for (row, provider) in zip(rows, providers) {
        let prediction = try await model.prediction(from: provider)
        guard let multiArray = prediction.featureValue(for: outputName)?.multiArrayValue else {
            throw CLIError.invalid("output \(outputName) is not a multi-array")
        }
        let tokenIDs = outputName == "token_ids" ? try tokenIDs(from: multiArray) : nil
        let argmax: Int
        if let tokenIDs {
            argmax = tokenIDs[tokenIDs.count - 1]
        } else {
            argmax = try argmaxLastPosition(multiArray)
        }
        outputs.append(ArgmaxRow(id: row.id, argmax: argmax, token_ids: tokenIDs))
    }
    try writeJSONLines(outputs, to: outputPath)
}

/// Keeps one Core ML model resident and serves JSONL draft requests. A K4
/// package still contributes only its final-position argmax per inference; the
/// remaining draft tokens are produced by sequential stateless re-encodes.
private func serve(_ arguments: [String]) async throws {
    let modelURL = URL(fileURLWithPath: try option("--model", in: arguments)).standardizedFileURL
    guard let window = Int(try option("--window", in: arguments, default: "32")), window > 0 else {
        throw CLIError.invalid("--window must be positive")
    }
    guard let padTokenID = Int(try option("--pad-token-id", in: arguments, default: "0")), padTokenID >= 0 else {
        throw CLIError.invalid("--pad-token-id must be nonnegative")
    }
    guard let draftTokens = Int(try option("--draft-tokens", in: arguments, default: "4")), draftTokens > 0 else {
        throw CLIError.invalid("--draft-tokens must be positive")
    }
    let computeUnits = try parseComputeUnits(try option("--compute-units", in: arguments, default: "CPU_AND_NE"))
    let configuration = MLModelConfiguration()
    configuration.computeUnits = computeUnits
    let model = try MLModel(contentsOf: modelURL, configuration: configuration)
    let outputNames = Array(model.modelDescription.outputDescriptionsByName.keys)
    guard outputNames.count == 1, let outputName = outputNames.first else {
        throw CLIError.invalid("expected exactly one output, found \(outputNames)")
    }

    while let line = readLine() {
        let request = try JSONDecoder().decode(ServeRequest.self, from: Data(line.utf8))
        guard !request.token_ids.isEmpty else {
            throw CLIError.invalid("draft request has no token ids")
        }
        var context = request.token_ids
        var draftIDs: [Int] = []
        draftIDs.reserveCapacity(draftTokens)
        let started = monotonicTime()
        for _ in 0 ..< draftTokens {
            let row = try leftPaddedRow(tokens: context, window: window, padTokenID: padTokenID)
            let prediction = try await model.prediction(from: try featureProvider(row))
            guard let multiArray = prediction.featureValue(for: outputName)?.multiArrayValue else {
                throw CLIError.invalid("output \(outputName) is not a multi-array")
            }
            let token = try argmaxLastPosition(multiArray)
            draftIDs.append(token)
            context.append(token)
            if context.count > window {
                context.removeFirst(context.count - window)
            }
        }
        let response = ServeResponse(
            draft_ids: draftIDs,
            compute_wall_s: monotonicTime() - started
        )
        let data = try JSONEncoder().encode(response)
        FileHandle.standardOutput.write(data)
        FileHandle.standardOutput.write(Data([0x0A]))
    }
}

private func leftPaddedRow(tokens: [Int], window: Int, padTokenID: Int = 0) throws -> TokenizedRow {
    let suffix = Array(tokens.suffix(window))
    let padding = window - suffix.count
    return TokenizedRow(
        id: "serve",
        input_ids: Array(repeating: padTokenID, count: padding) + suffix,
        attention_mask: Array(repeating: 0, count: padding) + Array(repeating: 1, count: suffix.count)
    )
}

private func loadRows(_ path: String) throws -> [TokenizedRow] {
    let text: String
    if path == "-" {
        text = String(decoding: FileHandle.standardInput.readDataToEndOfFile(), as: UTF8.self)
    } else {
        text = try String(contentsOfFile: path, encoding: .utf8)
    }
    let decoder = JSONDecoder()
    var rows: [TokenizedRow] = []
    for (lineNumber, rawLine) in text.split(whereSeparator: \.isNewline).enumerated() {
        do {
            rows.append(try decoder.decode(TokenizedRow.self, from: Data(rawLine.utf8)))
        } catch {
            throw CLIError.invalid("invalid JSON on line \(lineNumber + 1): \(error)")
        }
    }
    return rows
}

private func featureProvider(_ row: TokenizedRow) throws -> MLFeatureProvider {
    let ids = try multiArray(row.input_ids)
    let mask = try multiArray(row.attention_mask)
    return try MLDictionaryFeatureProvider(dictionary: [
        "input_ids": MLFeatureValue(multiArray: ids),
        "attention_mask": MLFeatureValue(multiArray: mask),
    ])
}

private func multiArray(_ values: [Int]) throws -> MLMultiArray {
    let array = try MLMultiArray(shape: [1, NSNumber(value: values.count)], dataType: .int32)
    try array.withUnsafeMutableBufferPointer(ofType: Int32.self) { buffer, _ in
        guard buffer.count == values.count else {
            throw CLIError.invalid("unexpected MLMultiArray backing length")
        }
        for (index, value) in values.enumerated() {
            guard let converted = Int32(exactly: value) else {
                throw CLIError.invalid("input value does not fit Int32: \(value)")
            }
            buffer[index] = converted
        }
    }
    return array
}

private func outputChecksum(_ prediction: MLFeatureProvider, outputName: String) throws -> Double {
    guard let output = prediction.featureValue(for: outputName)?.multiArrayValue else {
        throw CLIError.invalid("output \(outputName) is not a multi-array")
    }
    guard output.count > 0 else {
        throw CLIError.invalid("output \(outputName) is empty")
    }
    return output[output.count - 1].doubleValue
}

private func tokenIDs(from output: MLMultiArray) throws -> [Int] {
    let shape = output.shape.map { $0.intValue }
    guard shape.count == 2, shape[0] == 1, shape[1] > 0 else {
        throw CLIError.invalid("token id output has invalid shape \(shape)")
    }
    return (0 ..< shape[1]).map { output[$0].intValue }
}

private func argmaxLastPosition(_ output: MLMultiArray) throws -> Int {
    let shape = output.shape.map { $0.intValue }
    guard let vocab = shape.last, vocab > 0, output.count >= vocab else {
        throw CLIError.invalid("logits output has invalid shape \(shape)")
    }
    let base = output.count - vocab
    var bestIndex = 0
    var bestValue = -Double.infinity
    for index in 0 ..< vocab {
        let value = output[base + index].doubleValue
        if value > bestValue {
            bestValue = value
            bestIndex = index
        }
    }
    return bestIndex
}

@available(macOS 14.4, *)
private func buildPlacementReport(
    modelURL: URL,
    configuration: MLModelConfiguration,
    computeUnits: MLComputeUnits
) async throws -> PlacementReport {
    let plan = try await MLComputePlan.load(contentsOf: modelURL, configuration: configuration)
    switch plan.modelStructure {
    case .program(let program):
        let function = program.functions["main"] ?? program.functions.values.first
        guard let function else {
            throw CLIError.invalid("model structure has no functions")
        }
        var operations: [PlacementOperation] = []
        collectProgramOperations(plan: plan, block: function.block, into: &operations)
        return PlacementReport(
            model_path: modelURL.path,
            model_kind: "program",
            compute_units: computeUnitsLabel(computeUnits),
            summary: summarizePlacement(operations),
            operations: operations
        )
    case .neuralNetwork(let network):
        var operations: [PlacementOperation] = []
        for layer in network.layers {
            if let usage = plan.deviceUsage(for: layer) {
                operations.append(PlacementOperation(
                    operator_index: operations.count,
                    operator_name: layer.type,
                    preferred_device: deviceLabel(usage.preferred),
                    supported_devices: usage.supported.map(deviceLabel).sorted()
                ))
            }
        }
        return PlacementReport(
            model_path: modelURL.path,
            model_kind: "neuralNetwork",
            compute_units: computeUnitsLabel(computeUnits),
            summary: summarizePlacement(operations),
            operations: operations
        )
    default:
        throw CLIError.invalid("placement report supports program and neural-network models")
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
            operations.append(PlacementOperation(
                operator_index: operations.count,
                operator_name: operation.operatorName,
                preferred_device: deviceLabel(usage.preferred),
                supported_devices: usage.supported.map(deviceLabel).sorted()
            ))
        } else {
            operations.append(PlacementOperation(
                operator_index: operations.count,
                operator_name: operation.operatorName,
                preferred_device: "unknown",
                supported_devices: []
            ))
        }
        for nested in operation.blocks {
            collectProgramOperations(plan: plan, block: nested, into: &operations)
        }
    }
}

private func summarizePlacement(_ operations: [PlacementOperation]) -> PlacementSummary {
    var counts: [String: Int] = [:]
    var dispatchable: [String: Int] = [:]
    var nonANE: [PlacementOperation] = []
    var unknown: [PlacementOperation] = []
    for operation in operations {
        counts[operation.preferred_device, default: 0] += 1
        if operation.preferred_device == "unknown" {
            unknown.append(operation)
        } else {
            dispatchable[operation.preferred_device, default: 0] += 1
            if operation.preferred_device != "neuralEngine" {
                nonANE.append(operation)
            }
        }
    }
    let total = max(operations.count, 1)
    let dispatchableTotal = max(dispatchable.values.reduce(0, +), 1)
    return PlacementSummary(
        total_ops: operations.count,
        dispatchable_ops: dispatchable.values.reduce(0, +),
        preferred_device_counts: counts,
        preferred_device_share: counts.mapValues { Double($0) / Double(total) },
        dispatchable_device_counts: dispatchable,
        dispatchable_device_share: dispatchable.mapValues { Double($0) / Double(dispatchableTotal) },
        non_neural_engine_operations: nonANE,
        unknown_operations: unknown
    )
}

private func outputLastK(model: MLModel, outputName: String) -> Int {
    guard let description = model.modelDescription.outputDescriptionsByName[outputName],
          let constraint = description.multiArrayConstraint else {
        return 0
    }
    let shape = constraint.shape.map { $0.intValue }
    if outputName == "token_ids" {
        return shape.count == 2 ? shape[1] : 0
    }
    return shape.count >= 2 ? shape[shape.count - 2] : 0
}

private func parseComputeUnits(_ raw: String) throws -> MLComputeUnits {
    switch raw {
    case "CPU_AND_NE", "cpu-and-ne", "cpuAndNeuralEngine":
        return .cpuAndNeuralEngine
    case "CPU_ONLY", "cpu-only", "cpuOnly":
        return .cpuOnly
    case "CPU_AND_GPU", "cpu-and-gpu", "cpuAndGPU":
        return .cpuAndGPU
    case "ALL", "all":
        return .all
    default:
        throw CLIError.invalid("unsupported compute units: \(raw)")
    }
}

private func computeUnitsLabel(_ units: MLComputeUnits) -> String {
    switch units {
    case .cpuAndNeuralEngine:
        return "CPU_AND_NE"
    case .cpuOnly:
        return "CPU_ONLY"
    case .cpuAndGPU:
        return "CPU_AND_GPU"
    case .all:
        return "ALL"
    @unknown default:
        return "UNKNOWN"
    }
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

private func percentile(_ values: [Double], fraction: Double) -> Double {
    guard !values.isEmpty else {
        return 0
    }
    let sorted = values.sorted()
    let index = max(0, min(sorted.count - 1, Int(ceil(fraction * Double(sorted.count))) - 1))
    return sorted[index]
}

private func writeJSON<T: Encodable>(_ value: T, to path: String) throws {
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    var data = try encoder.encode(value)
    data.append(0x0A)
    if path == "-" {
        FileHandle.standardOutput.write(data)
    } else {
        let url = URL(fileURLWithPath: path)
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(), withIntermediateDirectories: true
        )
        try data.write(to: url)
    }
}

private func writeJSONLines<T: Encodable>(_ values: [T], to path: String) throws {
    let encoder = JSONEncoder()
    var data = Data()
    for value in values {
        data.append(try encoder.encode(value))
        data.append(0x0A)
    }
    if path == "-" {
        FileHandle.standardOutput.write(data)
    } else {
        let url = URL(fileURLWithPath: path)
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(), withIntermediateDirectories: true
        )
        try data.write(to: url)
    }
}

private func removeIfExists(_ url: URL) throws {
    if FileManager.default.fileExists(atPath: url.path) {
        try FileManager.default.removeItem(at: url)
    }
}

private func directorySize(_ url: URL) -> Int64 {
    guard let enumerator = FileManager.default.enumerator(
        at: url, includingPropertiesForKeys: [.isRegularFileKey, .fileSizeKey]
    ) else {
        return 0
    }
    var total: Int64 = 0
    for case let file as URL in enumerator {
        if let values = try? file.resourceValues(forKeys: [.isRegularFileKey, .fileSizeKey]),
           values.isRegularFile == true {
            total += Int64(values.fileSize ?? 0)
        }
    }
    return total
}

private func monotonicTime() -> Double {
    Double(DispatchTime.now().uptimeNanoseconds) / 1_000_000_000.0
}

private let usage = """
usage:
  ane-spec-decode compile --model MODEL.mlpackage --out MODEL.mlmodelc [--stats FILE|-]
  ane-spec-decode run --model MODEL.mlmodelc --input ROWS.jsonl --stats FILE
      [--placement FILE] [--compute-units CPU_AND_NE|CPU_ONLY]
      [--calls N] [--warmup N] [--duration-s N]
   ane-spec-decode predict --model MODEL.mlmodelc --input ROWS.jsonl --output TOKENS.jsonl
       [--compute-units CPU_AND_NE|CPU_ONLY]
       (token_ids outputs also include the complete token_ids array)
    ane-spec-decode serve --model MODEL.mlmodelc [--window 32] [--draft-tokens 4]
        [--pad-token-id ID] [--compute-units CPU_AND_NE|CPU_ONLY]
"""
