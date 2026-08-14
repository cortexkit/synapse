import Foundation

/// The sidecar protocol is intentionally separate from the general worker protocol.
/// A host must negotiate this exact version before it can install a model or submit
/// tokenized work.
public let sidecarProtocolVersion = 2
public let sidecarEngine = EngineIdentity(
    name: "ane-prefill-coreml",
    version: "coreml-fixed-window-v1"
)
public let defaultMaxFrameBytes = 64 * 1024 * 1024
public let sharedMemoryHandoffKind = "file_mmap_v1"
public let sharedMemoryHeaderBytes = 4 * 1024

public struct EngineIdentity: Codable, Equatable, Sendable {
    public let name: String
    public let version: String

    public init(name: String, version: String) {
        self.name = name
        self.version = version
    }
}

public struct HelloAck: Decodable, Equatable, Sendable {
    public let protocolVersion: Int
    public let accept: Bool
    public let expectedEngine: EngineIdentity
    public let nonce: String
    public let maxFrameBytes: Int

    enum CodingKeys: String, CodingKey {
        case protocolVersion = "protocol"
        case accept
        case expectedEngine = "expected_engine"
        case nonce
        case maxFrameBytes = "max_frame_bytes"
    }
}

public struct InstallRequest: Decodable, Equatable, Sendable {
    public let requestID: String
    public let modelRef: String
    public let artifactPath: String
    public let artifactSHA256: String
    public let window: Int
    public let layers: Int
    public let kvHeads: Int
    public let headDimension: Int
    public let vocabularySize: Int
    public let readinessTimeoutMilliseconds: UInt64

    enum CodingKeys: String, CodingKey {
        case requestID = "request_id"
        case modelRef = "model_ref"
        case artifactPath = "artifact_path"
        case artifactSHA256 = "artifact_sha256"
        case window
        case layers
        case kvHeads = "kv_heads"
        case headDimension = "head_dimension"
        case vocabularySize = "vocabulary_size"
        case readinessTimeoutMilliseconds = "readiness_timeout_ms"
    }
}

public struct SharedMemoryHandoffDescriptor: Decodable, Equatable, Sendable {
    public let kind: String
    public let path: String
    public let capacityBytes: Int
    public let generation: UInt64
    public let logitsOffset: Int
    public let logitsBytes: Int
    public let kvOffset: Int
    public let kvBytes: Int
    public let cacheTokens: Int

    public init(
        kind: String,
        path: String,
        capacityBytes: Int,
        generation: UInt64,
        logitsOffset: Int,
        logitsBytes: Int,
        kvOffset: Int,
        kvBytes: Int,
        cacheTokens: Int
    ) {
        self.kind = kind
        self.path = path
        self.capacityBytes = capacityBytes
        self.generation = generation
        self.logitsOffset = logitsOffset
        self.logitsBytes = logitsBytes
        self.kvOffset = kvOffset
        self.kvBytes = kvBytes
        self.cacheTokens = cacheTokens
    }

    enum CodingKeys: String, CodingKey {
        case kind
        case path
        case capacityBytes = "capacity_bytes"
        case generation
        case logitsOffset = "logits_offset"
        case logitsBytes = "logits_bytes"
        case kvOffset = "kv_offset"
        case kvBytes = "kv_bytes"
        case cacheTokens = "cache_tokens"
    }
}

public struct ExecuteRequest: Decodable, Equatable, Sendable {
    public let requestID: String
    public let modelRef: String
    public let activeTokens: Int
    public let inputIDs: [Int32]
    public let attentionMask: [Int32]
    public let handoff: SharedMemoryHandoffDescriptor

    enum CodingKeys: String, CodingKey {
        case requestID = "request_id"
        case modelRef = "model_ref"
        case activeTokens = "active_tokens"
        case inputIDs = "input_ids"
        case attentionMask = "attention_mask"
        case handoff
    }
}

public struct AbortRequest: Decodable, Equatable, Sendable {
    public let requestID: String
    public let executionID: String

    enum CodingKeys: String, CodingKey {
        case requestID = "request_id"
        case executionID = "execution_id"
    }
}

public struct TimingReadbackRequest: Decodable, Equatable, Sendable {
    public let requestID: String
    public let executionID: String

    enum CodingKeys: String, CodingKey {
        case requestID = "request_id"
        case executionID = "execution_id"
    }
}

public struct ShutdownRequest: Decodable, Equatable, Sendable {
    public let requestID: String

    enum CodingKeys: String, CodingKey {
        case requestID = "request_id"
    }
}

public enum SidecarCommand: Equatable, Sendable {
    case install(InstallRequest)
    case execute(ExecuteRequest)
    case abort(AbortRequest)
    case timingReadback(TimingReadbackRequest)
    case shutdown(ShutdownRequest)
}

public enum SidecarError: Error, Equatable, CustomStringConvertible, Sendable {
    case malformedFrame(String)
    case unexpectedField(String)
    case protocolMismatch(expected: Int, got: Int)
    case engineMismatch(expected: EngineIdentity, got: EngineIdentity)
    case rejectedHandshake
    case nonceMismatch
    case invalid(String)
    case readinessTimedOut
    case busy
    case cancelled
    case notFound(String)
    case artifactMismatch
    case kvConversion(String)
    case io(String)

    public var code: String {
        switch self {
        case .malformedFrame, .unexpectedField, .invalid:
            return "invalid_request"
        case .protocolMismatch:
            return "protocol_mismatch"
        case .engineMismatch:
            return "engine_mismatch"
        case .rejectedHandshake, .nonceMismatch:
            return "handshake_rejected"
        case .readinessTimedOut:
            return "readiness_timeout"
        case .busy:
            return "sidecar_busy"
        case .cancelled:
            return "cancelled"
        case .notFound:
            return "not_found"
        case .artifactMismatch:
            return "artifact_mismatch"
        case .kvConversion:
            return "kv_conversion_failure"
        case .io:
            return "io_failure"
        }
    }

    public var description: String {
        switch self {
        case .malformedFrame(let message), .invalid(let message), .kvConversion(let message),
            .io(let message):
            return message
        case .unexpectedField(let name):
            return "unexpected field \(name)"
        case .protocolMismatch(let expected, let got):
            return "protocol \(got) does not match required protocol \(expected)"
        case .engineMismatch(let expected, let got):
            return
                "engine \(got.name)@\(got.version) does not match \(expected.name)@\(expected.version)"
        case .rejectedHandshake:
            return "host rejected sidecar handshake"
        case .nonceMismatch:
            return "host did not acknowledge the connection nonce"
        case .readinessTimedOut:
            return "CoreML model did not become ready before the readiness deadline"
        case .busy:
            return "a prefill execution is already active"
        case .cancelled:
            return "the sidecar owns and observed cancellation for this execution"
        case .notFound(let reference):
            return "no installed model or execution named \(reference)"
        case .artifactMismatch:
            return "compiled CoreML artifact digest does not match the installed identity"
        }
    }
}

/// Validates the host's reply before any model state or request state exists. A
/// negotiation failure is therefore a connection failure, not an execution result.
public func negotiateHelloAck(_ frame: Data, nonce: String) throws -> Int {
    let object = try strictObject(frame)
    try requireExactFields(
        object,
        allowed: ["type", "protocol", "accept", "expected_engine", "nonce", "max_frame_bytes"]
    )
    guard object["type"] as? String == "HELLO_ACK" else {
        throw SidecarError.malformedFrame("expected HELLO_ACK")
    }
    try validateEngineObject(object["expected_engine"])
    let ack = try JSONDecoder().decode(HelloAck.self, from: frame)
    guard ack.protocolVersion == sidecarProtocolVersion else {
        throw SidecarError.protocolMismatch(
            expected: sidecarProtocolVersion, got: ack.protocolVersion)
    }
    guard ack.accept else {
        throw SidecarError.rejectedHandshake
    }
    guard ack.expectedEngine == sidecarEngine else {
        throw SidecarError.engineMismatch(expected: sidecarEngine, got: ack.expectedEngine)
    }
    guard ack.nonce == nonce else {
        throw SidecarError.nonceMismatch
    }
    guard (1...defaultMaxFrameBytes).contains(ack.maxFrameBytes) else {
        throw SidecarError.invalid("HELLO_ACK max_frame_bytes is outside the supported range")
    }
    return ack.maxFrameBytes
}

public func parseCommand(_ frame: Data) throws -> SidecarCommand {
    let object = try strictObject(frame)
    guard let type = object["type"] as? String else {
        throw SidecarError.malformedFrame("request does not contain a string type")
    }

    let decoder = JSONDecoder()
    switch type {
    case "INSTALL":
        try requireExactFields(
            object,
            allowed: [
                "type", "request_id", "model_ref", "artifact_path", "artifact_sha256", "window",
                "layers", "kv_heads", "head_dimension", "vocabulary_size", "readiness_timeout_ms",
            ]
        )
        let request = try decoder.decode(InstallRequest.self, from: frame)
        try validate(request)
        return .install(request)
    case "EXECUTE":
        try requireExactFields(
            object,
            allowed: [
                "type", "request_id", "model_ref", "active_tokens", "input_ids", "attention_mask",
                "handoff",
            ]
        )
        try validateHandoffObject(object["handoff"])
        let request = try decoder.decode(ExecuteRequest.self, from: frame)
        try validate(request)
        return .execute(request)
    case "ABORT":
        try requireExactFields(object, allowed: ["type", "request_id", "execution_id"])
        let request = try decoder.decode(AbortRequest.self, from: frame)
        try requireID(request.requestID, named: "request_id")
        try requireID(request.executionID, named: "execution_id")
        return .abort(request)
    case "TIMING_READBACK":
        try requireExactFields(object, allowed: ["type", "request_id", "execution_id"])
        let request = try decoder.decode(TimingReadbackRequest.self, from: frame)
        try requireID(request.requestID, named: "request_id")
        try requireID(request.executionID, named: "execution_id")
        return .timingReadback(request)
    case "SHUTDOWN":
        try requireExactFields(object, allowed: ["type", "request_id"])
        let request = try decoder.decode(ShutdownRequest.self, from: frame)
        try requireID(request.requestID, named: "request_id")
        return .shutdown(request)
    default:
        throw SidecarError.invalid("unknown sidecar request type \(type)")
    }
}

private func strictObject(_ frame: Data) throws -> [String: Any] {
    let value: Any
    do {
        value = try JSONSerialization.jsonObject(with: frame, options: [])
    } catch {
        throw SidecarError.malformedFrame("request is not valid JSON: \(error)")
    }
    guard let object = value as? [String: Any] else {
        throw SidecarError.malformedFrame("request must be a JSON object")
    }
    return object
}

private func requireExactFields(_ object: [String: Any], allowed: Set<String>) throws {
    for key in object.keys where !allowed.contains(key) {
        throw SidecarError.unexpectedField(key)
    }
}

private func validateHandoffObject(_ value: Any?) throws {
    guard let handoff = value as? [String: Any] else {
        throw SidecarError.malformedFrame("handoff must be an object")
    }
    try requireExactFields(
        handoff,
        allowed: [
            "kind", "path", "capacity_bytes", "generation", "logits_offset", "logits_bytes",
            "kv_offset", "kv_bytes", "cache_tokens",
        ]
    )
}

private func validateEngineObject(_ value: Any?) throws {
    guard let engine = value as? [String: Any] else {
        throw SidecarError.malformedFrame("expected_engine must be an object")
    }
    try requireExactFields(engine, allowed: ["name", "version"])
    guard engine["name"] as? String != nil, engine["version"] as? String != nil else {
        throw SidecarError.malformedFrame("expected_engine must contain string name and version")
    }
}

private func validate(_ request: InstallRequest) throws {
    try requireID(request.requestID, named: "request_id")
    try requireID(request.modelRef, named: "model_ref")
    guard !request.artifactPath.isEmpty, !request.artifactSHA256.isEmpty else {
        throw SidecarError.invalid("INSTALL requires artifact_path and artifact_sha256")
    }
    guard request.window > 0, request.layers > 0, request.kvHeads > 0,
        request.headDimension > 0, request.vocabularySize > 0
    else {
        throw SidecarError.invalid("INSTALL dimensions must be positive")
    }
    guard (1...3_600_000).contains(request.readinessTimeoutMilliseconds) else {
        throw SidecarError.invalid("readiness_timeout_ms must be between 1 and 3600000")
    }
}

private func validate(_ request: ExecuteRequest) throws {
    try requireID(request.requestID, named: "request_id")
    try requireID(request.modelRef, named: "model_ref")
    guard request.activeTokens > 0 else {
        throw SidecarError.invalid("active_tokens must be positive")
    }
    guard request.inputIDs.count == request.attentionMask.count else {
        throw SidecarError.invalid("input_ids and attention_mask lengths differ")
    }
    let handoff = request.handoff
    let (logitsEnd, logitsOverflow) = handoff.logitsOffset.addingReportingOverflow(
        handoff.logitsBytes)
    let (kvEnd, kvOverflow) = handoff.kvOffset.addingReportingOverflow(handoff.kvBytes)
    guard handoff.kind == sharedMemoryHandoffKind, handoff.path.hasPrefix("/"),
        handoff.capacityBytes > sharedMemoryHeaderBytes, handoff.generation > 0,
        handoff.logitsOffset >= sharedMemoryHeaderBytes, handoff.logitsBytes > 0,
        !logitsOverflow, handoff.kvOffset >= logitsEnd, handoff.kvBytes > 0,
        !kvOverflow, kvEnd <= handoff.capacityBytes,
        handoff.cacheTokens >= request.activeTokens
    else {
        throw SidecarError.invalid(
            "EXECUTE shared-memory descriptor is outside its mapped capacity")
    }
}

private func requireID(_ value: String, named: String) throws {
    guard !value.isEmpty, value.utf8.count <= 256 else {
        throw SidecarError.invalid("\(named) must be a non-empty identifier of at most 256 bytes")
    }
}
