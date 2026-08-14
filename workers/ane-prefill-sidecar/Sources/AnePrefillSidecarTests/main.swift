import AnePrefillSidecar
import CryptoKit
import Darwin
import Foundation

private struct HarnessFailure: Error, CustomStringConvertible {
    let description: String
}

@main
private struct TestHarness {
    static func main() async {
        do {
            try handshakeRejectsVersionAndEngineBeforeRequestsExist()
            try protocolRejectsUnknownCommandFields()
            try protocolRejectsUnknownHandoffFields()
            try executionRegistryMakesCancellationSidecarOwned()
            try logitsUseLastActiveTokenWithNonContiguousStride()
            try kvUsesStrideAwareActivePositionLayout()
            try sharedMemoryPublicationIsBitFaithful()
            try sharedMemoryRejectsTornWorkerHeader()
            try await readinessBudgetReturnsBeforeLateLoadCompletes()
            print("ane-prefill-sidecar tests passed")
        } catch {
            fputs("ane-prefill-sidecar tests failed: \(error)\n", stderr)
            Foundation.exit(1)
        }
    }
}

private func handshakeRejectsVersionAndEngineBeforeRequestsExist() throws {
    let wrongVersion = try json([
        "type": "HELLO_ACK",
        "protocol": sidecarProtocolVersion + 1,
        "accept": true,
        "expected_engine": ["name": sidecarEngine.name, "version": sidecarEngine.version],
        "nonce": "n",
        "max_frame_bytes": 1024,
    ])
    try expectSidecarError(
        .protocolMismatch(expected: sidecarProtocolVersion, got: sidecarProtocolVersion + 1)
    ) {
        _ = try negotiateHelloAck(wrongVersion, nonce: "n")
    }

    let wrongEngine = try json([
        "type": "HELLO_ACK",
        "protocol": sidecarProtocolVersion,
        "accept": true,
        "expected_engine": ["name": "gpu-prefill", "version": "v1"],
        "nonce": "n",
        "max_frame_bytes": 1024,
    ])
    try expectSidecarError(
        .engineMismatch(
            expected: sidecarEngine, got: EngineIdentity(name: "gpu-prefill", version: "v1"))
    ) {
        _ = try negotiateHelloAck(wrongEngine, nonce: "n")
    }
}

private func protocolRejectsUnknownCommandFields() throws {
    let frame = try json([
        "type": "ABORT",
        "request_id": "abort-1",
        "execution_id": "execute-1",
        "provenance": "must-not-be-accepted",
    ])
    try expectSidecarError(.unexpectedField("provenance")) {
        _ = try parseCommand(frame)
    }
}

private func protocolRejectsUnknownHandoffFields() throws {
    let frame = try json([
        "type": "EXECUTE",
        "request_id": "execute-1",
        "model_ref": "model",
        "active_tokens": 1,
        "input_ids": [1],
        "attention_mask": [1],
        "handoff": [
            "kind": sharedMemoryHandoffKind,
            "path": "/tmp/payload",
            "capacity_bytes": 8_256,
            "generation": 1,
            "logits_offset": sharedMemoryHeaderBytes,
            "logits_bytes": 16,
            "kv_offset": 4_160,
            "kv_bytes": 4_096,
            "cache_tokens": 4,
            "socket_payload": true,
        ],
    ])
    try expectSidecarError(.unexpectedField("socket_payload")) {
        _ = try parseCommand(frame)
    }
}

private func executionRegistryMakesCancellationSidecarOwned() throws {
    let registry = ExecutionRegistry()
    let ticket = try registry.begin(executionID: "execute-1")
    try require(!ticket.isCancelled, "new sidecar ticket is unexpectedly cancelled")
    try require(
        registry.abort(executionID: "execute-1"), "active execution did not acknowledge abort")
    try require(ticket.isCancelled, "sidecar ticket did not retain cancellation")
    try require(registry.finish(ticket), "finished ticket did not report cancellation")
    try require(!registry.abort(executionID: "execute-1"), "completed execution remained abortable")
}

private func logitsUseLastActiveTokenWithNonContiguousStride() throws {
    var values = [Float](repeating: -1, count: 64)
    for position in 0..<4 {
        for vocabulary in 0..<3 {
            values[position * 9 + vocabulary * 2] = Float(position * 10 + vocabulary)
        }
    }
    let tensor = try StridedTensor(
        shape: [1, 4, 3],
        strides: [50, 9, 2],
        storage: .float32(values)
    )
    try require(
        try copyActiveLogits(tensor, activeTokens: 2, window: 4, vocabularySize: 3) == [
            10, 11, 12,
        ],
        "logits were not read from the final active token"
    )

    let projectedTensor = try StridedTensor(
        shape: [1, 3, 1],
        strides: [9, 2, 1],
        storage: .float32([20, -1, 21, -1, 22])
    )
    try require(
        try copyActiveLogits(projectedTensor, activeTokens: 2, window: 4, vocabularySize: 3) == [
            20, 21, 22,
        ],
        "projected final-position logits were not accepted without a window axis"
    )

    var convolutionValues = [Float](repeating: -1, count: 64)
    for position in 0..<4 {
        for vocabulary in 0..<3 {
            convolutionValues[vocabulary * 11 + position * 2] = Float(position * 10 + vocabulary)
        }
    }
    let convolutionTensor = try StridedTensor(
        shape: [1, 3, 4],
        strides: [50, 11, 2],
        storage: .float32(convolutionValues)
    )
    try require(
        try copyActiveLogits(convolutionTensor, activeTokens: 2, window: 4, vocabularySize: 3) == [
            10, 11, 12,
        ],
        "convolution-shaped logits were not read with their actual strides"
    )
}

private func kvUsesStrideAwareActivePositionLayout() throws {
    var values = [UInt16](repeating: 0, count: 80)
    for head in 0..<2 {
        for position in 0..<4 {
            for dimension in 0..<3 {
                let offset = head * 40 + position * 7 + dimension * 2
                values[offset] = UInt16(head * 100 + position * 10 + dimension)
            }
        }
    }
    let tensor = try StridedTensor(
        shape: [1, 2, 4, 3],
        strides: [79, 40, 7, 2],
        storage: .float16Bits(values)
    )
    var copied: [UInt16] = []
    try appendActiveKV(
        tensor,
        to: &copied,
        activeTokens: 2,
        window: 4,
        kvHeads: 2,
        headDimension: 3
    )
    try require(
        copied == [0, 1, 2, 10, 11, 12, 100, 101, 102, 110, 111, 112],
        "K/V copy did not walk the tensor strides in wire layout order"
    )

    var padded = [UInt16](repeating: 0, count: 48)
    try values.withUnsafeBufferPointer { source in
        try padded.withUnsafeMutableBufferPointer { destination in
            for keyValueIndex in 0..<2 {
                try copyActiveKVToPaddedCache(
                    source: source,
                    tensorShape: [1, 2, 4, 3],
                    strides: [79, 40, 7, 2],
                    destination: destination,
                    layer: 0,
                    keyValueIndex: keyValueIndex,
                    activeTokens: 2,
                    window: 4,
                    layers: 1,
                    kvHeads: 2,
                    cacheTokens: 4,
                    headDimension: 3
                )
            }
        }
    }
    let paddedPlane: [UInt16] = [
        0, 1, 2, 10, 11, 12, 0, 0, 0, 0, 0, 0,
        100, 101, 102, 110, 111, 112, 0, 0, 0, 0, 0, 0,
    ]
    try require(
        padded == paddedPlane + paddedPlane,
        "shared-memory K/V layout changed active bits or admitted padded positions"
    )
}

private func sharedMemoryPublicationIsBitFaithful() throws {
    let path = FileManager.default.temporaryDirectory
        .appendingPathComponent("ane-prefill-sidecar-test-\(UUID().uuidString).shm")
    defer { try? FileManager.default.removeItem(at: path) }
    let logits = Data((0..<16).map(UInt8.init))
    let kv = Data((32..<64).map(UInt8.init))
    let descriptor = SharedMemoryHandoffDescriptor(
        kind: sharedMemoryHandoffKind,
        path: path.path,
        capacityBytes: 4_192,
        generation: 7,
        logitsOffset: sharedMemoryHeaderBytes,
        logitsBytes: logits.count,
        kvOffset: 4_160,
        kvBytes: kv.count,
        cacheTokens: 4
    )
    var file = Data(repeating: 0, count: descriptor.capacityBytes)
    file.replaceSubrange(
        0..<sharedMemoryHeaderBytes, with: sharedMemoryInitialHeader(for: descriptor))
    try file.write(to: path)
    try require(chmod(path.path, 0o600) == 0, "could not protect shared-memory fixture")

    let publication = try SharedMemoryPublication(descriptor: descriptor)
    try publication.begin()
    try publication.zeroKV()
    try publication.withMutableBytes(offset: descriptor.logitsOffset, count: logits.count) {
        destination in
        _ = logits.copyBytes(to: destination)
    }
    try publication.withMutableBytes(offset: descriptor.kvOffset, count: kv.count) { destination in
        _ = kv.copyBytes(to: destination)
    }
    let digest = try publication.finish()
    let expectedDigest = SHA256.hash(data: logits + kv).map { String(format: "%02x", $0) }.joined()
    try require(digest == expectedDigest, "shared-memory integrity digest changed payload bytes")

    let published = try Data(contentsOf: path)
    try require(
        published.subdata(in: descriptor.logitsOffset..<descriptor.logitsOffset + logits.count)
            == logits,
        "shared-memory logits publication was not bit-faithful"
    )
    try require(
        published.subdata(in: descriptor.kvOffset..<descriptor.kvOffset + kv.count) == kv,
        "shared-memory K/V publication was not bit-faithful"
    )
}

private func sharedMemoryRejectsTornWorkerHeader() throws {
    let path = FileManager.default.temporaryDirectory
        .appendingPathComponent("ane-prefill-sidecar-test-\(UUID().uuidString).shm")
    defer { try? FileManager.default.removeItem(at: path) }
    let descriptor = SharedMemoryHandoffDescriptor(
        kind: sharedMemoryHandoffKind,
        path: path.path,
        capacityBytes: 4_192,
        generation: 8,
        logitsOffset: sharedMemoryHeaderBytes,
        logitsBytes: 16,
        kvOffset: 4_160,
        kvBytes: 32,
        cacheTokens: 4
    )
    try Data(repeating: 0, count: descriptor.capacityBytes).write(to: path)
    try require(chmod(path.path, 0o600) == 0, "could not protect torn shared-memory fixture")
    try expectSidecarError(.io("shared-memory control header does not match EXECUTE")) {
        _ = try SharedMemoryPublication(descriptor: descriptor)
    }
}

private func readinessBudgetReturnsBeforeLateLoadCompletes() async throws {
    let clock = ContinuousClock()
    let started = clock.now
    do {
        _ = try await withinReadinessBudget(milliseconds: 25) {
            Thread.sleep(forTimeInterval: 0.15)
            return 1
        }
        throw HarnessFailure(description: "late readiness work must not become installed")
    } catch let error as SidecarError {
        try require(error == .readinessTimedOut, "readiness deadline returned \(error)")
        try require(
            clock.now - started < .milliseconds(100), "readiness deadline waited for the late load")
    }
}

private func expectSidecarError(
    _ expected: SidecarError,
    operation: () throws -> Void
) throws {
    do {
        try operation()
        throw HarnessFailure(description: "expected \(expected)")
    } catch let error as SidecarError {
        try require(error == expected, "expected \(expected), got \(error)")
    }
}

private func require(_ condition: Bool, _ message: String) throws {
    if !condition { throw HarnessFailure(description: message) }
}

private func json(_ object: [String: Any]) throws -> Data {
    try JSONSerialization.data(withJSONObject: object)
}
