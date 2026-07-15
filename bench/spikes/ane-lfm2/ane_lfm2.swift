import CoreML
import Foundation

private enum CLIError: LocalizedError {
    case invalid(String)

    var errorDescription: String? {
        switch self {
        case .invalid(let message): return message
        }
    }
}

private struct TokenizedRow: Decodable {
    let id: String
    let input_ids: [Int]
    let attention_mask: [Int]
}

private struct RunStats: Encodable {
    let model_path: String
    let output_name: String
    let bucket: Int
    let prompts: Int
    let iterations: Int
    let requests: Int
    let active_tokens: Int
    let bucket_tokens: Int
    let cold_load_s: Double
    let warmup_s: Double
    let infer_wall_s: Double
    let requests_per_s: Double
    let active_tokens_per_s: Double
    let bucket_tokens_per_s: Double
    let request_latency_p50_ms: Double
    let request_latency_p95_ms: Double
    let output_checksum: Double
    let compute_units: String
    let compiled_size_bytes: Int64
}

private struct CompileStats: Encodable {
    let source_model: String
    let compiled_model: String
    let compile_s: Double
    let source_size_bytes: Int64
    let compiled_size_bytes: Int64
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
    let operator_device_counts: [String: [String: Int]]
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
            guard !arguments.isEmpty else { throw CLIError.invalid(usage) }
            let command = arguments.removeFirst()
            switch command {
            case "compile": try await compile(arguments)
            case "run": try await run(arguments)
            default: throw CLIError.invalid("unknown command \(command)\n\(usage)")
            }
        } catch {
            FileHandle.standardError.write(Data("error: \(error.localizedDescription)\n".utf8))
            exit(1)
        }
    }
}

private func option(_ name: String, in arguments: [String], default defaultValue: String? = nil) throws -> String {
    guard let index = arguments.firstIndex(of: name) else {
        if let defaultValue { return defaultValue }
        throw CLIError.invalid("missing required option \(name)")
    }
    guard index + 1 < arguments.count else { throw CLIError.invalid("missing value for \(name)") }
    return arguments[index + 1]
}

private func compile(_ arguments: [String]) async throws {
    let source = URL(fileURLWithPath: try option("--model", in: arguments)).standardizedFileURL
    let destination = URL(fileURLWithPath: try option("--out", in: arguments)).standardizedFileURL
    try removeIfExists(destination)
    try FileManager.default.createDirectory(
        at: destination.deletingLastPathComponent(),
        withIntermediateDirectories: true
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
    guard let iterations = Int(try option("--iterations", in: arguments, default: "10")), iterations > 0 else {
        throw CLIError.invalid("--iterations must be positive")
    }
    guard let warmupCount = Int(try option("--warmup", in: arguments, default: "2")), warmupCount >= 0 else {
        throw CLIError.invalid("--warmup must be nonnegative")
    }
    let rows = try loadRows(inputPath)
    guard let first = rows.first else { throw CLIError.invalid("input has no rows") }
    let bucket = first.input_ids.count
    for row in rows where row.input_ids.count != bucket || row.attention_mask.count != bucket {
        throw CLIError.invalid("row \(row.id) does not match fixed bucket \(bucket)")
    }
    let providers = try rows.map(featureProvider)
    let configuration = MLModelConfiguration()
    configuration.computeUnits = .cpuAndNeuralEngine
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
    for _ in 0 ..< iterations {
        for provider in providers {
            let requestStarted = monotonicTime()
            let prediction = try await model.prediction(from: provider)
            latencies.append((monotonicTime() - requestStarted) * 1000.0)
            guard let output = prediction.featureValue(for: outputName)?.multiArrayValue else {
                throw CLIError.invalid("output \(outputName) is not a multi-array")
            }
            checksum += output[output.count - 1].doubleValue
        }
    }
    let inferSeconds = monotonicTime() - inferStarted
    let requestCount = rows.count * iterations
    let activePerIteration = rows.reduce(0) { total, row in
        total + row.attention_mask.reduce(0, +)
    }
    let activeTokens = activePerIteration * iterations
    let bucketTokens = requestCount * bucket
    let stats = RunStats(
        model_path: modelURL.path,
        output_name: outputName,
        bucket: bucket,
        prompts: rows.count,
        iterations: iterations,
        requests: requestCount,
        active_tokens: activeTokens,
        bucket_tokens: bucketTokens,
        cold_load_s: coldLoadSeconds,
        warmup_s: warmupSeconds,
        infer_wall_s: inferSeconds,
        requests_per_s: Double(requestCount) / inferSeconds,
        active_tokens_per_s: Double(activeTokens) / inferSeconds,
        bucket_tokens_per_s: Double(bucketTokens) / inferSeconds,
        request_latency_p50_ms: percentile(latencies, fraction: 0.50),
        request_latency_p95_ms: percentile(latencies, fraction: 0.95),
        output_checksum: checksum,
        compute_units: "cpuAndNeuralEngine",
        compiled_size_bytes: directorySize(modelURL)
    )
    try writeJSON(stats, to: statsPath)
    if !placementPath.isEmpty {
        let report = try await buildPlacementReport(modelURL: modelURL, configuration: configuration)
        try writeJSON(report, to: placementPath)
    }
}

private func loadRows(_ path: String) throws -> [TokenizedRow] {
    let text = try String(contentsOfFile: path, encoding: .utf8)
    let decoder = JSONDecoder()
    return try text.split(whereSeparator: \.isNewline).map { line in
        try decoder.decode(TokenizedRow.self, from: Data(line.utf8))
    }
}

private func featureProvider(_ row: TokenizedRow) throws -> MLFeatureProvider {
    let ids = try multiArray(row.input_ids)
    let mask = try multiArray(row.attention_mask)
    return try MLDictionaryFeatureProvider(dictionary: ["input_ids": ids, "attention_mask": mask])
}

private func multiArray(_ values: [Int]) throws -> MLMultiArray {
    let array = try MLMultiArray(shape: [1, NSNumber(value: values.count)], dataType: .int32)
    for (index, value) in values.enumerated() { array[index] = NSNumber(value: value) }
    return array
}

@available(macOS 14.4, *)
private func buildPlacementReport(
    modelURL: URL,
    configuration: MLModelConfiguration
) async throws -> PlacementReport {
    let plan = try await MLComputePlan.load(contentsOf: modelURL, configuration: configuration)
    switch plan.modelStructure {
    case .program(let program):
        let function = program.functions["main"] ?? program.functions.values.first
        guard let function else { throw CLIError.invalid("model structure has no functions") }
        var operations: [PlacementOperation] = []
        collectProgramOperations(plan: plan, block: function.block, into: &operations)
        return PlacementReport(
            model_path: modelURL.path,
            model_kind: "program",
            compute_units: "cpuAndNeuralEngine",
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
            compute_units: "cpuAndNeuralEngine",
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
    var operatorDeviceCounts: [String: [String: Int]] = [:]
    var nonANE: [PlacementOperation] = []
    var unknown: [PlacementOperation] = []
    for operation in operations {
        counts[operation.preferred_device, default: 0] += 1
        operatorDeviceCounts[operation.operator_name, default: [:]][operation.preferred_device, default: 0] += 1
        if operation.preferred_device == "unknown" {
            unknown.append(operation)
        } else {
            dispatchable[operation.preferred_device, default: 0] += 1
            if operation.preferred_device != "neuralEngine" { nonANE.append(operation) }
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
        operator_device_counts: operatorDeviceCounts,
        non_neural_engine_operations: nonANE,
        unknown_operations: unknown
    )
}

private func deviceLabel(_ device: MLComputeDevice) -> String {
    switch device {
    case .cpu: return "cpu"
    case .gpu: return "gpu"
    case .neuralEngine: return "neuralEngine"
    @unknown default: return "unknown"
    }
}

private func percentile(_ values: [Double], fraction: Double) -> Double {
    guard !values.isEmpty else { return 0 }
    let sorted = values.sorted()
    let index = max(0, min(sorted.count - 1, Int(ceil(fraction * Double(sorted.count))) - 1))
    return sorted[index]
}

private func directorySize(_ url: URL) -> Int64 {
    if let values = try? url.resourceValues(forKeys: [.isRegularFileKey, .fileSizeKey]),
       values.isRegularFile == true {
        return Int64(values.fileSize ?? 0)
    }
    guard let enumerator = FileManager.default.enumerator(
        at: url,
        includingPropertiesForKeys: [.isRegularFileKey, .fileSizeKey]
    ) else { return 0 }
    var total: Int64 = 0
    for case let file as URL in enumerator {
        if let values = try? file.resourceValues(forKeys: [.isRegularFileKey, .fileSizeKey]),
           values.isRegularFile == true {
            total += Int64(values.fileSize ?? 0)
        }
    }
    return total
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
            at: url.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        try data.write(to: url)
    }
}

private func removeIfExists(_ url: URL) throws {
    if FileManager.default.fileExists(atPath: url.path) { try FileManager.default.removeItem(at: url) }
}

private func monotonicTime() -> Double {
    Double(DispatchTime.now().uptimeNanoseconds) / 1_000_000_000.0
}

private let usage = """
usage:
  ane-lfm2 compile --model MODEL.mlpackage --out MODEL.mlmodelc [--stats FILE|-]
  ane-lfm2 run --model MODEL.mlmodelc --input PROMPTS.jsonl --stats FILE \\
    [--placement FILE] [--iterations N] [--warmup N]
"""
