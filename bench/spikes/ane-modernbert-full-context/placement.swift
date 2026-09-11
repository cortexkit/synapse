import CoreML
import Foundation

private struct OperationPlacement: Encodable {
    let operator_name: String
    let preferred_device: String
    let supported_devices: [String]
}

private struct PlacementReport: Encodable {
    let model_path: String
    let compute_units: String
    let evidence_kind: String
    let total_operations: Int
    let preferred_device_counts: [String: Int]
    let operator_device_counts: [String: [String: Int]]
    let expensive_operator_device_counts: [String: [String: Int]]
    let non_neural_engine_operations: [OperationPlacement]
    let unknown_operation_counts: [String: Int]
}

private let expensiveOperators: Set<String> = [
    "conv", "einsum", "linear", "matmul", "reduce_mean", "softmax", "layer_norm",
]

@main
private enum ModernBertPlacement {
    static func main() async throws {
        guard CommandLine.arguments.count == 3 || CommandLine.arguments.count == 4 else {
            throw NSError(
                domain: "ModernBertPlacement",
                code: 2,
                userInfo: [
                    NSLocalizedDescriptionKey:
                        "usage: placement MODEL.mlpackage REPORT.json [cpu-and-ne|all]"
                ]
            )
        }
        let modelURL = URL(fileURLWithPath: CommandLine.arguments[1]).standardizedFileURL
        let reportURL = URL(fileURLWithPath: CommandLine.arguments[2]).standardizedFileURL
        let computeUnitsName = CommandLine.arguments.count == 4 ? CommandLine.arguments[3] : "cpu-and-ne"
        let configuration = MLModelConfiguration()
        switch computeUnitsName {
        case "cpu-and-ne":
            configuration.computeUnits = .cpuAndNeuralEngine
        case "all":
            configuration.computeUnits = .all
        default:
            throw NSError(
                domain: "ModernBertPlacement",
                code: 2,
                userInfo: [NSLocalizedDescriptionKey: "compute units must be cpu-and-ne or all"]
            )
        }
        if #available(macOS 14.4, *) {
            configuration.optimizationHints.reshapeFrequency = .infrequent
        }
        let planURL = modelURL.pathExtension == "mlmodelc"
            ? modelURL
            : try await MLModel.compileModel(at: modelURL)
        let plan = try await MLComputePlan.load(contentsOf: planURL, configuration: configuration)
        guard case .program(let program) = plan.modelStructure else {
            throw NSError(
                domain: "ModernBertPlacement",
                code: 3,
                userInfo: [NSLocalizedDescriptionKey: "expected an ML Program model"]
            )
        }
        let function = program.functions["main"] ?? program.functions.values.first
        guard let function else {
            throw NSError(
                domain: "ModernBertPlacement",
                code: 4,
                userInfo: [NSLocalizedDescriptionKey: "ML Program has no function"]
            )
        }
        var operations: [OperationPlacement] = []
        collect(plan: plan, block: function.block, into: &operations)
        let report = summarize(
            modelURL: modelURL,
            computeUnits: computeUnitsName,
            operations: operations
        )
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try FileManager.default.createDirectory(
            at: reportURL.deletingLastPathComponent(), withIntermediateDirectories: true
        )
        try encoder.encode(report).write(to: reportURL)
    }
}

@available(macOS 14.4, *)
private func collect(
    plan: MLComputePlan,
    block: MLModelStructure.Program.Block,
    into operations: inout [OperationPlacement]
) {
    for operation in block.operations {
        if let usage = plan.deviceUsage(for: operation) {
            operations.append(
                OperationPlacement(
                    operator_name: operation.operatorName,
                    preferred_device: deviceLabel(usage.preferred),
                    supported_devices: usage.supported.map(deviceLabel).sorted()
                )
            )
        } else {
            operations.append(
                OperationPlacement(
                    operator_name: operation.operatorName,
                    preferred_device: "unknown",
                    supported_devices: []
                )
            )
        }
        for nested in operation.blocks {
            collect(plan: plan, block: nested, into: &operations)
        }
    }
}

private func summarize(
    modelURL: URL,
    computeUnits: String,
    operations: [OperationPlacement]
) -> PlacementReport {
    var preferredCounts: [String: Int] = [:]
    var operatorCounts: [String: [String: Int]] = [:]
    var expensiveCounts: [String: [String: Int]] = [:]
    var nonANE: [OperationPlacement] = []
    var unknownCounts: [String: Int] = [:]
    for operation in operations {
        preferredCounts[operation.preferred_device, default: 0] += 1
        operatorCounts[operation.operator_name, default: [:]][operation.preferred_device, default: 0] += 1
        if let expensiveName = normalizedExpensiveName(operation.operator_name) {
            expensiveCounts[expensiveName, default: [:]][operation.preferred_device, default: 0] += 1
        }
        if operation.preferred_device == "unknown" {
            unknownCounts[operation.operator_name, default: 0] += 1
        } else if operation.preferred_device != "neuralEngine" {
            nonANE.append(operation)
        }
    }
    return PlacementReport(
        model_path: modelURL.path,
        compute_units: computeUnits,
        evidence_kind: "MLComputePlan preferred placement; not a runtime dispatch trace",
        total_operations: operations.count,
        preferred_device_counts: preferredCounts,
        operator_device_counts: operatorCounts,
        expensive_operator_device_counts: expensiveCounts,
        non_neural_engine_operations: nonANE,
        unknown_operation_counts: unknownCounts
    )
}

private func normalizedExpensiveName(_ operatorName: String) -> String? {
    for candidate in expensiveOperators {
        if operatorName == candidate || operatorName.hasSuffix(".\(candidate)") {
            return candidate
        }
    }
    return nil
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
