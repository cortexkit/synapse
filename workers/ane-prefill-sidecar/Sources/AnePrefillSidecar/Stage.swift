import Dispatch
import Foundation

/// The sidecar owns cancellation after an execution has been admitted. The host
/// sends ABORT, while the stage prevents the aborted execution from publishing a
/// logits/K/V payload when CoreML returns.
public final class ExecutionTicket: @unchecked Sendable {
    public let executionID: String
    private let lock = NSLock()
    private var cancelled = false
    private var completed = false

    fileprivate init(executionID: String) {
        self.executionID = executionID
    }

    @discardableResult
    fileprivate func abort() -> Bool {
        lock.lock()
        defer { lock.unlock() }
        guard !completed else { return false }
        cancelled = true
        return true
    }

    fileprivate func finish() -> Bool {
        lock.lock()
        defer { lock.unlock() }
        completed = true
        return cancelled
    }

    public var isCancelled: Bool {
        lock.lock()
        defer { lock.unlock() }
        return cancelled
    }
}

/// A single CoreML sidecar has one active prediction. Keeping this state inside
/// the sidecar makes an ABORT deterministic and prevents concurrent predictions
/// from accidentally sharing model-owned CoreML buffers.
public final class ExecutionRegistry: @unchecked Sendable {
    private let lock = NSLock()
    private var active: [String: ExecutionTicket] = [:]

    public init() {}

    public func begin(executionID: String) throws -> ExecutionTicket {
        lock.lock()
        defer { lock.unlock() }
        guard active.isEmpty else { throw SidecarError.busy }
        guard active[executionID] == nil else {
            throw SidecarError.invalid("execution_id is already active")
        }
        let ticket = ExecutionTicket(executionID: executionID)
        active[executionID] = ticket
        return ticket
    }

    @discardableResult
    public func abort(executionID: String) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        return active[executionID]?.abort() ?? false
    }

    public func finish(_ ticket: ExecutionTicket) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        guard active[ticket.executionID] === ticket else { return ticket.finish() }
        let wasCancelled = ticket.finish()
        active.removeValue(forKey: ticket.executionID)
        return wasCancelled
    }

    public func abortAll() {
        lock.lock()
        defer { lock.unlock() }
        active.values.forEach { _ = $0.abort() }
    }
}

public struct StageTimings: Codable, Equatable, Sendable {
    public let executionID: String
    public let readinessMilliseconds: Double?
    public let predictionMilliseconds: Double
    public let kvLayoutMilliseconds: Double
    public let logitsCopyMilliseconds: Double
    public let totalMilliseconds: Double
    public let cancelled: Bool

    public init(
        executionID: String,
        readinessMilliseconds: Double?,
        predictionMilliseconds: Double,
        kvLayoutMilliseconds: Double,
        logitsCopyMilliseconds: Double,
        totalMilliseconds: Double,
        cancelled: Bool
    ) {
        self.executionID = executionID
        self.readinessMilliseconds = readinessMilliseconds
        self.predictionMilliseconds = predictionMilliseconds
        self.kvLayoutMilliseconds = kvLayoutMilliseconds
        self.logitsCopyMilliseconds = logitsCopyMilliseconds
        self.totalMilliseconds = totalMilliseconds
        self.cancelled = cancelled
    }

    enum CodingKeys: String, CodingKey {
        case executionID = "execution_id"
        case readinessMilliseconds = "readiness_ms"
        case predictionMilliseconds = "prediction_ms"
        case kvLayoutMilliseconds = "kv_layout_ms"
        case logitsCopyMilliseconds = "logits_copy_ms"
        case totalMilliseconds = "total_ms"
        case cancelled
    }
}

/// Timing records are bounded so a hostile or malfunctioning host cannot retain
/// per-request state indefinitely. Readback is sidecar-local and does not create
/// request provenance.
public final class TimingLedger: @unchecked Sendable {
    private let lock = NSLock()
    private let capacity: Int
    private var order: [String] = []
    private var entries: [String: StageTimings] = [:]

    public init(capacity: Int = 128) {
        self.capacity = max(capacity, 1)
    }

    public func record(_ timing: StageTimings) {
        lock.lock()
        defer { lock.unlock() }
        if entries[timing.executionID] == nil {
            order.append(timing.executionID)
        }
        entries[timing.executionID] = timing
        while order.count > capacity {
            entries.removeValue(forKey: order.removeFirst())
        }
    }

    public func read(executionID: String) -> StageTimings? {
        lock.lock()
        defer { lock.unlock() }
        return entries[executionID]
    }
}

private final class FirstCompletion<Value: Sendable>: @unchecked Sendable {
    private let lock = NSLock()
    private var didComplete = false
    private let continuation: CheckedContinuation<Value, any Error>

    init(_ continuation: CheckedContinuation<Value, any Error>) {
        self.continuation = continuation
    }

    func succeed(_ value: Value) {
        lock.lock()
        guard !didComplete else {
            lock.unlock()
            return
        }
        didComplete = true
        lock.unlock()
        continuation.resume(returning: value)
    }

    func fail(_ error: SidecarError) {
        lock.lock()
        guard !didComplete else {
            lock.unlock()
            return
        }
        didComplete = true
        lock.unlock()
        continuation.resume(throwing: error)
    }
}

/// Runs CoreML readiness work under a real wall-clock bound. A synchronous CoreML
/// load cannot be force-cancelled by its API, so a late result is discarded and is
/// never inserted into the installed-model table. Supervisor shutdown remains able
/// to terminate the separate sidecar process if the platform call itself hangs.
public func withinReadinessBudget<Value: Sendable>(
    milliseconds: UInt64,
    operation: @escaping @Sendable () throws -> Value
) async throws -> Value {
    guard milliseconds > 0 else {
        throw SidecarError.invalid("readiness budget must be positive")
    }
    return try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Value, any Error>) in
        let first = FirstCompletion(continuation)
        DispatchQueue.global(qos: .userInitiated).async {
            do {
                first.succeed(try operation())
            } catch let error as SidecarError {
                first.fail(error)
            } catch {
                first.fail(.invalid("readiness operation failed: \(error)"))
            }
        }
        DispatchQueue.global(qos: .userInitiated).asyncAfter(
            deadline: .now() + .milliseconds(Int(milliseconds))
        ) {
            first.fail(.readinessTimedOut)
        }
    }
}

public func monotonicMilliseconds() -> Double {
    Double(DispatchTime.now().uptimeNanoseconds) / 1_000_000.0
}
