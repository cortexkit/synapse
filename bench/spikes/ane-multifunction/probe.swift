// Core ML probe for the multi-function ModernBERT embedding spike.
//
// Every subcommand loads models with compute units `.cpuAndNeuralEngine`, the
// setting the production Swift ANE worker uses, and selects a function of a
// multi-function package through `MLModelConfiguration.functionName`. A function
// argument of "-" means "single-function package, no function name".
//
// Subcommands (all print one JSON object on stdout):
//   compile PACKAGE.mlpackage DEST.mlmodelc
//       Compile a package and move the bundle to a stable destination path. Core ML
//       caches the Neural Engine specialization by bundle path, so loads must come
//       from a stable path rather than from the temporary compile output.
//   placement MODEL.mlmodelc FUNCTION
//       MLComputePlan preferred-device counts for one function, with the share
//       computed exactly as the production worker computes placement_share.
//   load MODEL.mlmodelc FUNCTION ROWS.json BUCKET
//       Time one model load and the first prediction on the first row of BUCKET.
//   embed MODEL.mlmodelc FUNCTION ROWS.json BUCKET OUT.json
//       Embed every row of BUCKET and write the output vectors.
//   latency PLAN.json OUT.json
//       Interleaved warm latency across several loaded models (see LatencyPlan).
//   memory ROWS.json MODEL.mlmodelc:FUNCTION:BUCKET ...
//       Process physical footprint before loading, after loading every listed
//       model, and after one prediction on each.

import CoreML
import Darwin
import Foundation
import SQLite3

private struct Row: Decodable {
    let id: String
    let ids: [Int32]
}

private typealias RowsByBucket = [String: [Row]]

private func fail(_ message: String) -> Never {
    FileHandle.standardError.write(Data((message + "\n").utf8))
    exit(2)
}

private func nowNanos() -> UInt64 { clock_gettime_nsec_np(CLOCK_UPTIME_RAW) }

private func millis(since start: UInt64) -> Double { Double(nowNanos() - start) / 1_000_000 }

private func loadAverage() -> [Double] {
    var values = [Double](repeating: 0, count: 3)
    _ = getloadavg(&values, 3)
    return values
}

/// Physical footprint (dirty and compressed memory the process is charged for)
/// and resident size (which also counts clean file-backed pages such as mapped
/// weight files). Neural Engine allocations live outside this process and
/// appear in neither.
private func memoryUsage() -> (footprint: UInt64, resident: UInt64) {
    var info = task_vm_info_data_t()
    var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
    let result = withUnsafeMutablePointer(to: &info) {
        $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
            task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
        }
    }
    return result == KERN_SUCCESS ? (info.phys_footprint, info.resident_size) : (0, 0)
}

/// Number of rows in the campaign rig claim table. A non-zero count means a
/// campaign holds the benchmark rig and Neural Engine measurements must wait.
private func rigClaimCount() -> Int {
    let path = ProcessInfo.processInfo.environment["RIG_CLAIM_DB"]
        ?? (NSHomeDirectory() + "/.local/share/cortexkit/prefrontal-core/store.db")
    var db: OpaquePointer?
    guard sqlite3_open_v2("file:\(path)?mode=ro", &db, SQLITE_OPEN_READONLY | SQLITE_OPEN_URI, nil) == SQLITE_OK else {
        fail("cannot open rig claim database \(path)")
    }
    defer { sqlite3_close(db) }
    var statement: OpaquePointer?
    guard sqlite3_prepare_v2(db, "SELECT count(*) FROM campaign_rig_claim", -1, &statement, nil) == SQLITE_OK else {
        fail("cannot query campaign_rig_claim")
    }
    defer { sqlite3_finalize(statement) }
    guard sqlite3_step(statement) == SQLITE_ROW else { fail("campaign_rig_claim query returned no row") }
    return Int(sqlite3_column_int64(statement, 0))
}

/// Block until no campaign holds the rig and the 1-minute load average is at or
/// below `maxLoad`. Returns the seconds spent waiting.
private func waitForQuietRig(maxLoad: Double) -> Double {
    let start = nowNanos()
    while rigClaimCount() > 0 || loadAverage()[0] > maxLoad {
        Thread.sleep(forTimeInterval: 5)
    }
    return millis(since: start) / 1000
}

private func configuration(function: String) -> MLModelConfiguration {
    let configuration = MLModelConfiguration()
    configuration.computeUnits = .cpuAndNeuralEngine
    if function != "-" {
        configuration.functionName = function
    }
    return configuration
}

private func loadModel(_ path: String, function: String) throws -> MLModel {
    try MLModel(contentsOf: URL(fileURLWithPath: path), configuration: configuration(function: function))
}

/// Synchronous prediction, as the production worker calls it. Inside an async
/// context Swift would otherwise pick the async overload.
private func predict(_ model: MLModel, _ provider: MLFeatureProvider) throws -> MLFeatureProvider {
    try model.prediction(from: provider)
}

private func readRows(_ path: String) -> RowsByBucket {
    do {
        return try JSONDecoder().decode(RowsByBucket.self, from: Data(contentsOf: URL(fileURLWithPath: path)))
    } catch {
        fail("cannot read rows \(path): \(error)")
    }
}

/// Fixed-shape inputs padded the way the production worker pads: token id 0 and
/// attention mask 0 after the real tokens.
private func inputs(for row: Row, bucket: Int) throws -> MLDictionaryFeatureProvider {
    guard row.ids.count <= bucket else { fail("row \(row.id) has \(row.ids.count) tokens > \(bucket)") }
    let ids = try MLMultiArray(shape: [1, NSNumber(value: bucket)], dataType: .int32)
    let mask = try MLMultiArray(shape: [1, NSNumber(value: bucket)], dataType: .int32)
    let idPointer = ids.dataPointer.bindMemory(to: Int32.self, capacity: bucket)
    let maskPointer = mask.dataPointer.bindMemory(to: Int32.self, capacity: bucket)
    for index in 0..<bucket {
        let real = index < row.ids.count
        idPointer[index] = real ? row.ids[index] : 0
        maskPointer[index] = real ? 1 : 0
    }
    return try MLDictionaryFeatureProvider(dictionary: [
        "input_ids": MLFeatureValue(multiArray: ids),
        "attention_mask": MLFeatureValue(multiArray: mask),
    ])
}

private func vector(_ output: MLFeatureProvider) -> (values: [Float], dataType: String) {
    guard let array = output.featureValue(for: "embedding")?.multiArrayValue else {
        fail("model output has no `embedding` multi-array")
    }
    var values = [Float](repeating: 0, count: array.count)
    for index in 0..<array.count {
        values[index] = array[index].floatValue
    }
    let dataType: String
    switch array.dataType {
    case .float16: dataType = "float16"
    case .float32: dataType = "float32"
    case .double: dataType = "float64"
    default: dataType = "other"
    }
    return (values, dataType)
}

private func emit(_ object: Any) {
    guard let data = try? JSONSerialization.data(withJSONObject: object, options: [.sortedKeys]) else {
        fail("cannot encode result")
    }
    FileHandle.standardOutput.write(data)
    FileHandle.standardOutput.write(Data("\n".utf8))
}

private func write(_ object: Any, to path: String) {
    guard let data = try? JSONSerialization.data(withJSONObject: object, options: [.sortedKeys]) else {
        fail("cannot encode result")
    }
    do {
        try data.write(to: URL(fileURLWithPath: path))
    } catch {
        fail("cannot write \(path): \(error)")
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

private func collect(
    plan: MLComputePlan, block: MLModelStructure.Program.Block,
    counts: inout [String: Int], byOperator: inout [String: [String: Int]]
) {
    for operation in block.operations {
        let device = plan.deviceUsage(for: operation).map { deviceLabel($0.preferred) } ?? "unknown"
        counts[device, default: 0] += 1
        byOperator[operation.operatorName, default: [:]][device, default: 0] += 1
        for nested in operation.blocks {
            collect(plan: plan, block: nested, counts: &counts, byOperator: &byOperator)
        }
    }
}

private func placement(model: String, function: String) async throws -> [String: Any] {
    let plan = try await MLComputePlan.load(
        contentsOf: URL(fileURLWithPath: model), configuration: configuration(function: function))
    guard case .program(let program) = plan.modelStructure else { fail("expected an ML Program") }
    let name = function == "-" ? "main" : function
    guard let target = program.functions[name] else {
        fail("function \(name) not in \(program.functions.keys.sorted())")
    }
    var counts: [String: Int] = [:]
    var byOperator: [String: [String: Int]] = [:]
    collect(plan: plan, block: target.block, counts: &counts, byOperator: &byOperator)
    let dispatchable = counts.filter { $0.key != "unknown" }.values.reduce(0, +)
    let neural = counts["neuralEngine", default: 0]
    let nonNeural = byOperator.compactMapValues { devices -> [String: Int]? in
        let filtered = devices.filter { $0.key != "neuralEngine" && $0.key != "unknown" }
        return filtered.isEmpty ? nil : filtered
    }
    return [
        "function": name,
        "program_functions": program.functions.keys.sorted(),
        "total_operations": counts.values.reduce(0, +),
        "preferred_device_counts": counts,
        "placement_share": dispatchable == 0 ? NSNull() : Double(neural) / Double(dispatchable) as Any,
        "non_neural_engine_operators": nonNeural,
    ]
}

/// One model that the latency subcommand times; `label` identifies it in the output.
private struct LatencyTarget: Decodable {
    let label: String
    let model: String
    let function: String
    let bucket: Int
}

/// Interleaved latency plan. Each iteration runs every group once; within a
/// group the targets alternate order between iterations so neither package
/// systematically runs first. One run embeds every row of the group's bucket.
private struct LatencyPlan: Decodable {
    let rows: String
    let groups: [[LatencyTarget]]
    let warmup: Int
    let iterations: Int
    let max_load: Double
}

private func latency(planPath: String, outPath: String) throws {
    let plan = try JSONDecoder().decode(LatencyPlan.self, from: Data(contentsOf: URL(fileURLWithPath: planPath)))
    let rows = readRows(plan.rows)
    var models: [String: MLModel] = [:]
    var loadMs: [String: Double] = [:]
    var prepared: [String: [MLDictionaryFeatureProvider]] = [:]
    for target in plan.groups.flatMap({ $0 }) {
        let start = nowNanos()
        models[target.label] = try loadModel(target.model, function: target.function)
        loadMs[target.label] = millis(since: start)
        guard let bucketRows = rows[String(target.bucket)] else { fail("no rows for \(target.bucket)") }
        prepared[target.label] = try bucketRows.map { try inputs(for: $0, bucket: target.bucket) }
    }
    func run(_ label: String) throws -> Double {
        let model = models[label]!
        let start = nowNanos()
        for provider in prepared[label]! {
            _ = try model.prediction(from: provider)
        }
        return millis(since: start)
    }
    for _ in 0..<plan.warmup {
        for target in plan.groups.flatMap({ $0 }) { _ = try run(target.label) }
    }
    var samples: [[String: Any]] = []
    for iteration in 0..<plan.iterations {
        for group in plan.groups {
            let ordered = iteration % 2 == 0 ? group : group.reversed()
            for target in ordered {
                let waited = waitForQuietRig(maxLoad: plan.max_load)
                let before = loadAverage()
                let elapsed = try run(target.label)
                let after = loadAverage()
                samples.append([
                    "iteration": iteration,
                    "label": target.label,
                    "bucket": target.bucket,
                    "rows": prepared[target.label]!.count,
                    "run_ms": elapsed,
                    "load_before": before,
                    "load_after": after,
                    "waited_s": waited,
                ])
            }
        }
    }
    write(["load_ms": loadMs, "samples": samples], to: outPath)
    emit(["samples": samples.count])
}

@main
private enum Probe {
    static func main() async throws {
        let args = CommandLine.arguments
        guard args.count >= 2 else { fail("usage: probe SUBCOMMAND ...") }
        switch args[1] {
        case "compile" where args.count == 4:
            let destination = URL(fileURLWithPath: args[3])
            let start = nowNanos()
            let compiled = try await MLModel.compileModel(at: URL(fileURLWithPath: args[2]))
            let compileMs = millis(since: start)
            if FileManager.default.fileExists(atPath: destination.path) {
                try FileManager.default.removeItem(at: destination)
            }
            try FileManager.default.createDirectory(
                at: destination.deletingLastPathComponent(), withIntermediateDirectories: true)
            try FileManager.default.moveItem(at: compiled, to: destination)
            emit(["compile_ms": compileMs, "path": destination.path])
        case "placement" where args.count == 4:
            emit(try await placement(model: args[2], function: args[3]))
        case "load" where args.count == 6:
            let rows = readRows(args[4])
            let bucket = Int(args[5])!
            let waited = waitForQuietRig(maxLoad: 16)
            let loadBefore = loadAverage()
            let start = nowNanos()
            let model = try loadModel(args[2], function: args[3])
            let loadMs = millis(since: start)
            let provider = try inputs(for: rows[String(bucket)]![0], bucket: bucket)
            let predictStart = nowNanos()
            _ = try predict(model, provider)
            let firstPredictMs = millis(since: predictStart)
            emit([
                "load_ms": loadMs, "first_predict_ms": firstPredictMs,
                "load_before": loadBefore, "load_after": loadAverage(), "waited_s": waited,
            ])
        case "embed" where args.count == 7:
            let rows = readRows(args[4])
            let bucket = Int(args[5])!
            let model = try loadModel(args[2], function: args[3])
            var vectors: [[String: Any]] = []
            var dataType = ""
            for row in rows[String(bucket)] ?? [] {
                let output = try predict(model, try inputs(for: row, bucket: bucket))
                let result = vector(output)
                dataType = result.dataType
                vectors.append(["id": row.id, "tokens": row.ids.count, "vector": result.values.map { Double($0) }])
            }
            write(["output_data_type": dataType, "rows": vectors], to: args[6])
            emit(["rows": vectors.count, "output_data_type": dataType])
        case "latency" where args.count == 4:
            try latency(planPath: args[2], outPath: args[3])
        case "memory" where args.count >= 4:
            let rows = readRows(args[2])
            let baseline = memoryUsage()
            var loaded: [(MLModel, MLDictionaryFeatureProvider)] = []
            var loadMs: [Double] = []
            for spec in args[3...] {
                let parts = spec.split(separator: ":", omittingEmptySubsequences: false).map(String.init)
                guard parts.count == 3, let bucket = Int(parts[2]) else { fail("bad target \(spec)") }
                let start = nowNanos()
                let model = try loadModel(parts[0], function: parts[1])
                loadMs.append(millis(since: start))
                loaded.append((model, try inputs(for: rows[parts[2]]![0], bucket: bucket)))
            }
            let afterLoad = memoryUsage()
            for (model, provider) in loaded { _ = try predict(model, provider) }
            let afterPredict = memoryUsage()
            emit([
                "baseline_bytes": baseline.footprint, "after_load_bytes": afterLoad.footprint,
                "after_predict_bytes": afterPredict.footprint,
                "baseline_resident_bytes": baseline.resident, "after_load_resident_bytes": afterLoad.resident,
                "after_predict_resident_bytes": afterPredict.resident,
                "models": loaded.count, "load_ms": loadMs,
            ])
            // With PROBE_HOLD=1 the models stay loaded until stdin closes, so the
            // driver can sample system-wide memory while they are resident.
            if ProcessInfo.processInfo.environment["PROBE_HOLD"] == "1" {
                _ = FileHandle.standardInput.readDataToEndOfFile()
                withExtendedLifetime(loaded) {}
            }
        default:
            fail("unknown or malformed subcommand: \(args.dropFirst().joined(separator: " "))")
        }
    }
}
