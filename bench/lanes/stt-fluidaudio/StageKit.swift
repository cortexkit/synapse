#if STAGE_KIT
import CoreML
import Foundation

/// A clip's task-local accumulator follows child window tasks without global clip IDs.
public final class StageKit: @unchecked Sendable {
    @TaskLocal public static var current: StageKit?
    private let lock = NSLock()
    public let enabled: Bool
    private var milliseconds: [String: Double] = ["encoder_ms": 0, "predictor_ms": 0, "joint_ms": 0]
    private var counts: [String: Int] = ["encoder_calls": 0, "predictor_calls": 0, "joint_calls": 0, "accepted_tokens": 0]
    private var units: [String: String] = [:]
    private var rungs: [Int] = []

    public init(enabled: Bool) { self.enabled = enabled }
    public static func start() -> Double? {
        current?.enabled == true ? ProcessInfo.processInfo.systemUptime : nil
    }
    public static func finish(_ name: String, _ start: Double?, model: MLModel) {
        guard let start, let current else { return }
        let elapsed = (ProcessInfo.processInfo.systemUptime - start) * 1000
        current.lock.lock()
        defer { current.lock.unlock() }
        current.milliseconds[name + "_ms", default: 0] += elapsed
        current.counts[name + "_calls", default: 0] += 1
        current.units[name] = String(describing: model.configuration.computeUnits)
    }
    public static func token() {
        guard let current, current.enabled else { return }
        current.lock.lock()
        defer { current.lock.unlock() }
        current.counts["accepted_tokens", default: 0] += 1
    }
    public static func segment(_ start: Double?, valid: Int, padded: MLMultiArray) {
        guard let start, let current else { return }
        let elapsed = (ProcessInfo.processInfo.systemUptime - start) * 1000
        current.lock.lock()
        defer { current.lock.unlock() }
        current.milliseconds["model_total_ms", default: 0] += elapsed
        current.counts["segment_count", default: 0] += 1
        current.counts["valid_frames", default: 0] += valid
        let timeAxis = padded.shape[1].intValue == 1024 ? 2 : 1
        let paddedFrames = padded.shape[timeAxis].intValue
        current.counts["padded_frames", default: 0] += paddedFrames
        current.rungs.append(paddedFrames)
    }
    public func snapshot() -> [String: Any] {
        lock.lock()
        defer { lock.unlock() }
        var record: [String: Any] = [:]
        for (key, value) in milliseconds where enabled || key == "model_total_ms" { record[key] = value }
        for (key, value) in counts where enabled || ["segment_count", "valid_frames", "padded_frames"].contains(key) {
            record[key] = value
        }
        record["call_compute_units"] = units
        record["rung"] = rungs
        if enabled { record["joint_steps"] = counts["joint_calls", default: 0] }
        return record
    }
}
#endif
