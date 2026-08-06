import CoreML
import Darwin
import Foundation

private enum CLIError: LocalizedError {
    case invalid(String)

    var errorDescription: String? {
        switch self {
        case .invalid(let message): message
        }
    }
}

private struct TokenizedRow: Decodable {
    let id: String
    let input_ids: [Int32]
    let attention_mask: [Int32]
}

private struct CompileStats: Encodable {
    let source_model: String
    let compiled_model: String
    let compile_s: Double
    let source_size_bytes: Int64
    let compiled_size_bytes: Int64
}

private struct CallStages: Encodable {
    let prediction_ms: Double
    let kv_copy_layout_ms: Double
    let logits_copy_ms: Double
    let logits_argmax_ms: Double
    let compute_and_copy_ms: Double
}

private struct RunStats: Encodable {
    let model_path: String
    let compute_units: String
    let model_window: Int
    let chunks: Int
    let prompt_tokens: Int
    let cache_bucket: Int
    let layers: Int
    let kv_heads: Int
    let head_dim: Int
    let output_count: Int
    let kv_output_elements_per_request: Int
    let kv_output_bytes_f16_per_request: Int
    let padded_cache_bytes_f16: Int
    let calls: Int
    let warmups: Int
    let cold_load_s: Double
    let warmup_s: Double
    let prediction_wall_s: Double
    let kv_copy_layout_wall_s: Double
    let logits_copy_wall_s: Double
    let logits_argmax_wall_s: Double
    let compute_and_copy_wall_s: Double
    let prediction_p50_ms: Double
    let prediction_p95_ms: Double
    let kv_copy_layout_p50_ms: Double
    let kv_copy_layout_p95_ms: Double
    let logits_copy_p50_ms: Double
    let logits_argmax_p50_ms: Double
    let compute_and_copy_p50_ms: Double
    let compute_and_copy_p95_ms: Double
    let artifact_write_ms: Double
    let cache_path: String
    let logits_path: String
    let logits_argmax: Int
    let logits_top2_gap: Float
    let stages: [CallStages]
}

private struct PlacementOperation: Encodable {
    let operator_index: Int
    let operator_name: String
    let preferred_device: String
    let supported_devices: [String]
}

private struct PlacementSummary: Encodable {
    let total_ops: Int
    let preferred_device_counts: [String: Int]
    let preferred_device_share: [String: Double]
    let dispatchable_ops: Int
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
            default:
                throw CLIError.invalid("unknown command \(command)\n\(usage)")
            }
        } catch {
            FileHandle.standardError.write(Data("error: \(error.localizedDescription)\n".utf8))
            exit(1)
        }
    }
}

private let usage = """
usage:
  ane-prefill-runner compile --model <source.mlpackage> --out <compiled.mlmodelc> --stats <json>
  ane-prefill-runner run --model <compiled.mlmodelc> --input <jsonl> --stats <json> \\
    --cache-out <f16.bin> --logits-out <f32.bin> --model-window <32|128> --chunks <1|4> \\
    [--cache-bucket 512] [--calls 20] [--warmup 3] [--compute-units cpu-and-ne] [--placement <json>]
"""

private func option(
    _ name: String, in arguments: [String], default defaultValue: String? = nil
) throws -> String {
    guard let index = arguments.firstIndex(of: name) else {
        if let defaultValue { return defaultValue }
        throw CLIError.invalid("missing \(name)")
    }
    guard index + 1 < arguments.count else {
        throw CLIError.invalid("missing value after \(name)")
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
    try writeJSON(
        CompileStats(
            source_model: source.path,
            compiled_model: destination.path,
            compile_s: compileSeconds,
            source_size_bytes: directorySize(source),
            compiled_size_bytes: directorySize(destination)
        ),
        to: try option("--stats", in: arguments)
    )
}

private func run(_ arguments: [String]) async throws {
    let modelURL = URL(fileURLWithPath: try option("--model", in: arguments)).standardizedFileURL
    let inputPath = try option("--input", in: arguments)
    let statsPath = try option("--stats", in: arguments)
    let cachePath = try option("--cache-out", in: arguments)
    let logitsPath = try option("--logits-out", in: arguments)
    let placementPath = try option("--placement", in: arguments, default: "")
    let modelWindow = try positiveInt("--model-window", arguments)
    let chunks = try positiveInt("--chunks", arguments)
    let cacheBucket = try positiveInt("--cache-bucket", arguments, default: "512")
    let calls = try positiveInt("--calls", arguments, default: "20")
    let warmups = try nonnegativeInt("--warmup", arguments, default: "3")
    let layers = try positiveInt("--layers", arguments, default: "28")
    let kvHeads = try positiveInt("--kv-heads", arguments, default: "8")
    let headDim = try positiveInt("--head-dim", arguments, default: "128")
    let computeUnits = try parseComputeUnits(
        try option("--compute-units", in: arguments, default: "cpu-and-ne")
    )
    guard modelWindow * chunks <= cacheBucket else {
        throw CLIError.invalid("window x chunks exceeds the cache bucket")
    }
    let rows = try loadRows(inputPath)
    guard rows.count == 1, let row = rows.first else {
        throw CLIError.invalid("the transfer harness requires exactly one input row")
    }
    guard row.input_ids.count == modelWindow * chunks,
          row.attention_mask.count == row.input_ids.count
    else {
        throw CLIError.invalid(
            "input row has \(row.input_ids.count) tokens; expected \(modelWindow * chunks)"
        )
    }
    let providers = try (0 ..< chunks).map {
        try featureProvider(row: row, offset: $0 * modelWindow, count: modelWindow)
    }

    let configuration = MLModelConfiguration()
    configuration.computeUnits = computeUnits
    let loadStarted = monotonicTime()
    let model = try MLModel(contentsOf: modelURL, configuration: configuration)
    let coldLoadSeconds = monotonicTime() - loadStarted
    let outputNames = Set(model.modelDescription.outputDescriptionsByName.keys)
    let expectedNames = Set(
        ["logits"] + (0 ..< layers).flatMap { [String(format: "key_%02d", $0), String(format: "value_%02d", $0)] }
    )
    guard outputNames == expectedNames else {
        throw CLIError.invalid(
            "model outputs do not match logits plus \(layers) K/V pairs: \(outputNames.sorted())"
        )
    }

    let oneCacheElements = kvHeads * cacheBucket * headDim
    var cacheBits = [UInt16](repeating: 0, count: layers * 2 * oneCacheElements)
    var logits = [Float]()
    var selectedTop = (index: 0, gap: -Float.infinity)
    let warmupStarted = monotonicTime()
    for _ in 0 ..< warmups {
        for provider in providers {
            _ = try await model.prediction(from: provider)
        }
    }
    let warmupSeconds = monotonicTime() - warmupStarted

    var stages: [CallStages] = []
    for _ in 0 ..< calls {
        var predictionMS = 0.0
        var copyMS = 0.0
        var logitsMS = 0.0
        var argmaxMS = 0.0
        for (chunk, provider) in providers.enumerated() {
            let predictionStarted = monotonicTime()
            let prediction = try await model.prediction(from: provider)
            predictionMS += (monotonicTime() - predictionStarted) * 1000.0

            let copyStarted = monotonicTime()
            try copyCaches(
                prediction: prediction,
                destination: &cacheBits,
                chunkOffset: chunk * modelWindow,
                modelWindow: modelWindow,
                cacheBucket: cacheBucket,
                layers: layers,
                kvHeads: kvHeads,
                headDim: headDim
            )
            copyMS += (monotonicTime() - copyStarted) * 1000.0
            if chunk == chunks - 1 {
                let logitsStarted = monotonicTime()
                logits = try copyLogits(prediction)
                logitsMS += (monotonicTime() - logitsStarted) * 1000.0
                let argmaxStarted = monotonicTime()
                selectedTop = topTwo(logits)
                argmaxMS += (monotonicTime() - argmaxStarted) * 1000.0
            }
        }
        stages.append(
            CallStages(
                prediction_ms: predictionMS,
                kv_copy_layout_ms: copyMS,
                logits_copy_ms: logitsMS,
                logits_argmax_ms: argmaxMS,
                compute_and_copy_ms: predictionMS + copyMS + logitsMS + argmaxMS
            )
        )
    }

    let artifactStarted = monotonicTime()
    try writeRaw(cacheBits, to: cachePath)
    try writeRaw(logits, to: logitsPath)
    let artifactWriteMS = (monotonicTime() - artifactStarted) * 1000.0
    let predictionValues = stages.map(\.prediction_ms)
    let copyValues = stages.map(\.kv_copy_layout_ms)
    let logitsValues = stages.map(\.logits_copy_ms)
    let argmaxValues = stages.map(\.logits_argmax_ms)
    let totalValues = stages.map(\.compute_and_copy_ms)
    let result = RunStats(
        model_path: modelURL.path,
        compute_units: computeUnitsLabel(computeUnits),
        model_window: modelWindow,
        chunks: chunks,
        prompt_tokens: modelWindow * chunks,
        cache_bucket: cacheBucket,
        layers: layers,
        kv_heads: kvHeads,
        head_dim: headDim,
        output_count: outputNames.count,
        kv_output_elements_per_request: layers * 2 * kvHeads * modelWindow * headDim * chunks,
        kv_output_bytes_f16_per_request: layers * 2 * kvHeads * modelWindow * headDim * chunks * 2,
        padded_cache_bytes_f16: cacheBits.count * 2,
        calls: calls,
        warmups: warmups,
        cold_load_s: coldLoadSeconds,
        warmup_s: warmupSeconds,
        prediction_wall_s: predictionValues.reduce(0, +) / 1000.0,
        kv_copy_layout_wall_s: copyValues.reduce(0, +) / 1000.0,
        logits_copy_wall_s: logitsValues.reduce(0, +) / 1000.0,
        logits_argmax_wall_s: argmaxValues.reduce(0, +) / 1000.0,
        compute_and_copy_wall_s: totalValues.reduce(0, +) / 1000.0,
        prediction_p50_ms: percentile(predictionValues, fraction: 0.50),
        prediction_p95_ms: percentile(predictionValues, fraction: 0.95),
        kv_copy_layout_p50_ms: percentile(copyValues, fraction: 0.50),
        kv_copy_layout_p95_ms: percentile(copyValues, fraction: 0.95),
        logits_copy_p50_ms: percentile(logitsValues, fraction: 0.50),
        logits_argmax_p50_ms: percentile(argmaxValues, fraction: 0.50),
        compute_and_copy_p50_ms: percentile(totalValues, fraction: 0.50),
        compute_and_copy_p95_ms: percentile(totalValues, fraction: 0.95),
        artifact_write_ms: artifactWriteMS,
        cache_path: URL(fileURLWithPath: cachePath).standardized.path,
        logits_path: URL(fileURLWithPath: logitsPath).standardized.path,
        logits_argmax: selectedTop.index,
        logits_top2_gap: selectedTop.gap,
        stages: stages
    )
    try writeJSON(result, to: statsPath)
    if !placementPath.isEmpty {
        let report = try await buildPlacementReport(
            modelURL: modelURL, configuration: configuration, computeUnits: computeUnits
        )
        try writeJSON(report, to: placementPath)
    }
}

private func copyCaches(
    prediction: MLFeatureProvider,
    destination: inout [UInt16],
    chunkOffset: Int,
    modelWindow: Int,
    cacheBucket: Int,
    layers: Int,
    kvHeads: Int,
    headDim: Int
) throws {
    let oneCacheElements = kvHeads * cacheBucket * headDim
    for layer in 0 ..< layers {
        for (kind, prefix) in ["key", "value"].enumerated() {
            let name = String(format: "%@_%02d", prefix, layer)
            guard let array = prediction.featureValue(for: name)?.multiArrayValue else {
                throw CLIError.invalid("prediction has no multi-array output named \(name)")
            }
            let shape = array.shape.map(\.intValue)
            guard shape == [1, kvHeads, modelWindow, headDim] else {
                throw CLIError.invalid("\(name) has shape \(shape)")
            }
            let strides = array.strides.map(\.intValue)
            let destinationBase = (layer * 2 + kind) * oneCacheElements
            try copyF16Tensor(
                array,
                strides: strides,
                destination: &destination,
                destinationBase: destinationBase,
                chunkOffset: chunkOffset,
                modelWindow: modelWindow,
                cacheBucket: cacheBucket,
                kvHeads: kvHeads,
                headDim: headDim
            )
        }
    }
}

private func copyF16Tensor(
    _ array: MLMultiArray,
    strides: [Int],
    destination: inout [UInt16],
    destinationBase: Int,
    chunkOffset: Int,
    modelWindow: Int,
    cacheBucket: Int,
    kvHeads: Int,
    headDim: Int
) throws {
    guard array.dataType == .float16 else {
        throw CLIError.invalid("K/V output dtype is \(array.dataType.rawValue); expected float16")
    }
    let source = array.dataPointer.bindMemory(to: UInt16.self, capacity: array.count)
    destination.withUnsafeMutableBufferPointer { target in
        guard let targetBase = target.baseAddress else { return }
        for head in 0 ..< kvHeads {
            for position in 0 ..< modelWindow {
                let sourceOffset = head * strides[1] + position * strides[2]
                let targetOffset = destinationBase
                    + (head * cacheBucket + chunkOffset + position) * headDim
                if strides[3] == 1 {
                    targetBase.advanced(by: targetOffset).update(
                        from: source.advanced(by: sourceOffset), count: headDim
                    )
                } else {
                    for dimension in 0 ..< headDim {
                        targetBase[targetOffset + dimension] = source[
                            sourceOffset + dimension * strides[3]
                        ]
                    }
                }
            }
        }
    }
}

private func copyLogits(_ prediction: MLFeatureProvider) throws -> [Float] {
    guard let array = prediction.featureValue(for: "logits")?.multiArrayValue else {
        throw CLIError.invalid("prediction has no logits multi-array")
    }
    let shape = array.shape.map(\.intValue)
    let strides = array.strides.map(\.intValue)
    let nonUnitAxes = shape.indices.filter { shape[$0] > 1 }
    guard nonUnitAxes.count == 1, let axis = nonUnitAxes.first, shape[axis] == array.count else {
        throw CLIError.invalid("logits shape \(shape) is not a single vocabulary axis")
    }
    let stride = strides[axis]
    var result = [Float](repeating: 0, count: array.count)
    switch array.dataType {
    case .float16:
        let source = array.dataPointer.bindMemory(
            to: UInt16.self, capacity: (array.count - 1) * stride + 1
        )
        for index in result.indices {
            result[index] = Float(Float16(bitPattern: source[index * stride]))
        }
    case .float32:
        let source = array.dataPointer.bindMemory(
            to: Float.self, capacity: (array.count - 1) * stride + 1
        )
        for index in result.indices { result[index] = source[index * stride] }
    case .double:
        let source = array.dataPointer.bindMemory(
            to: Double.self, capacity: (array.count - 1) * stride + 1
        )
        for index in result.indices { result[index] = Float(source[index * stride]) }
    default:
        throw CLIError.invalid("unsupported logits dtype \(array.dataType.rawValue)")
    }
    return result
}

private func topTwo(_ values: [Float]) -> (Int, Float) {
    var best = (index: 0, value: -Float.infinity)
    var second = -Float.infinity
    for (index, value) in values.enumerated() {
        if value > best.value {
            second = best.value
            best = (index, value)
        } else if value > second {
            second = value
        }
    }
    return (best.index, best.value - second)
}

private func featureProvider(
    row: TokenizedRow, offset: Int, count: Int
) throws -> MLDictionaryFeatureProvider {
    let ids = try multiArray(Array(row.input_ids[offset ..< offset + count]))
    let mask = try multiArray(Array(row.attention_mask[offset ..< offset + count]))
    return try MLDictionaryFeatureProvider(dictionary: ["input_ids": ids, "attention_mask": mask])
}

private func multiArray(_ values: [Int32]) throws -> MLMultiArray {
    let array = try MLMultiArray(shape: [1, NSNumber(value: values.count)], dataType: .int32)
    let pointer = array.dataPointer.bindMemory(to: Int32.self, capacity: values.count)
    for (index, value) in values.enumerated() { pointer[index] = value }
    return array
}

private func loadRows(_ path: String) throws -> [TokenizedRow] {
    let text = try String(contentsOfFile: path, encoding: .utf8)
    let decoder = JSONDecoder()
    return try text.split(whereSeparator: \.isNewline).map {
        try decoder.decode(TokenizedRow.self, from: Data($0.utf8))
    }
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
        guard let function = program.functions["main"] ?? program.functions.values.first else {
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
    default:
        throw CLIError.invalid("placement report requires an ML Program")
    }
}

@available(macOS 14.4, *)
private func collectProgramOperations(
    plan: MLComputePlan,
    block: MLModelStructure.Program.Block,
    into operations: inout [PlacementOperation]
) {
    for operation in block.operations {
        let usage = plan.deviceUsage(for: operation)
        operations.append(
            PlacementOperation(
                operator_index: operations.count,
                operator_name: operation.operatorName,
                preferred_device: usage.map { deviceLabel($0.preferred) } ?? "unknown",
                supported_devices: usage.map { $0.supported.map(deviceLabel).sorted() } ?? []
            )
        )
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
            if operation.preferred_device != "neuralEngine" { nonANE.append(operation) }
        }
    }
    let total = max(operations.count, 1)
    let dispatchableTotal = max(dispatchable.values.reduce(0, +), 1)
    return PlacementSummary(
        total_ops: operations.count,
        preferred_device_counts: counts,
        preferred_device_share: counts.mapValues { Double($0) / Double(total) },
        dispatchable_ops: dispatchable.values.reduce(0, +),
        dispatchable_device_counts: dispatchable,
        dispatchable_device_share: dispatchable.mapValues {
            Double($0) / Double(dispatchableTotal)
        },
        non_neural_engine_operations: nonANE,
        unknown_operations: unknown
    )
}

private func parseComputeUnits(_ value: String) throws -> MLComputeUnits {
    switch value.lowercased().replacingOccurrences(of: "_", with: "-") {
    case "all": return .all
    case "cpu-only": return .cpuOnly
    case "cpu-and-gpu": return .cpuAndGPU
    case "cpu-and-ne", "cpu-and-neural-engine": return .cpuAndNeuralEngine
    default: throw CLIError.invalid("unknown compute units \(value)")
    }
}

private func computeUnitsLabel(_ value: MLComputeUnits) -> String {
    switch value {
    case .all: "ALL"
    case .cpuOnly: "CPU_ONLY"
    case .cpuAndGPU: "CPU_AND_GPU"
    case .cpuAndNeuralEngine: "CPU_AND_NE"
    @unknown default: "UNKNOWN"
    }
}

private func deviceLabel(_ device: MLComputeDevice) -> String {
    switch device {
    case .cpu: "cpu"
    case .gpu: "gpu"
    case .neuralEngine: "neuralEngine"
    @unknown default: "unknown"
    }
}

private func positiveInt(
    _ name: String, _ arguments: [String], default defaultValue: String? = nil
) throws -> Int {
    guard let value = Int(try option(name, in: arguments, default: defaultValue)), value > 0 else {
        throw CLIError.invalid("\(name) must be positive")
    }
    return value
}

private func nonnegativeInt(
    _ name: String, _ arguments: [String], default defaultValue: String? = nil
) throws -> Int {
    guard let value = Int(try option(name, in: arguments, default: defaultValue)), value >= 0 else {
        throw CLIError.invalid("\(name) must be nonnegative")
    }
    return value
}

private func percentile(_ values: [Double], fraction: Double) -> Double {
    guard !values.isEmpty else { return 0 }
    let sorted = values.sorted()
    let index = min(sorted.count - 1, Int(Double(sorted.count - 1) * fraction))
    return sorted[index]
}

private func writeRaw<T>(_ values: [T], to path: String) throws {
    let url = URL(fileURLWithPath: path).standardizedFileURL
    try FileManager.default.createDirectory(
        at: url.deletingLastPathComponent(), withIntermediateDirectories: true
    )
    let data = values.withUnsafeBytes { Data($0) }
    try data.write(to: url, options: .atomic)
}

private func writeJSON<T: Encodable>(_ value: T, to path: String) throws {
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    let data = try encoder.encode(value)
    let url = URL(fileURLWithPath: path).standardizedFileURL
    try FileManager.default.createDirectory(
        at: url.deletingLastPathComponent(), withIntermediateDirectories: true
    )
    try data.write(to: url, options: .atomic)
}

private func removeIfExists(_ url: URL) throws {
    if FileManager.default.fileExists(atPath: url.path) {
        try FileManager.default.removeItem(at: url)
    }
}

private func directorySize(_ url: URL) -> Int64 {
    guard let enumerator = FileManager.default.enumerator(
        at: url, includingPropertiesForKeys: [.fileSizeKey]
    ) else { return 0 }
    var total: Int64 = 0
    for case let file as URL in enumerator {
        total += Int64((try? file.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0)
    }
    return total
}

private func monotonicTime() -> Double {
    var value = timespec()
    clock_gettime(CLOCK_MONOTONIC_RAW, &value)
    return Double(value.tv_sec) + Double(value.tv_nsec) / 1_000_000_000.0
}
