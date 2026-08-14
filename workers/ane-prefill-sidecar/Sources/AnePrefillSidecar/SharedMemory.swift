import CryptoKit
import Darwin
import Foundation

private let sharedMemoryMagic = Array("CKANESM2".utf8)
private let sharedMemoryProtocolOffset = 8
private let sharedMemoryStateOffset = 16
private let sharedMemoryGenerationOffset = 24
private let sharedMemoryLogitsOffsetOffset = 32
private let sharedMemoryLogitsBytesOffset = 40
private let sharedMemoryKVOffsetOffset = 48
private let sharedMemoryKVBytesOffset = 56
private let sharedMemoryDigestOffset = 64
private let sharedMemoryDigestBytes = 32
private let sharedMemoryEmpty: UInt64 = 0
private let sharedMemoryWriting: UInt64 = 1
private let sharedMemoryReady: UInt64 = 2

/// Publishes one payload into a worker-owned mapping. The READY state is written
/// last, after the payload digest, so the worker never admits a partial write.
public final class SharedMemoryPublication: @unchecked Sendable {
    public let descriptor: SharedMemoryHandoffDescriptor

    private let fileDescriptor: Int32
    private let baseAddress: UnsafeMutableRawPointer
    private var publishing = false
    private var published = false

    public init(descriptor: SharedMemoryHandoffDescriptor) throws {
        let fd = Darwin.open(descriptor.path, O_RDWR | O_NOFOLLOW | O_CLOEXEC)
        guard fd >= 0 else {
            throw SidecarError.io("open shared-memory handoff: \(String(cString: strerror(errno)))")
        }
        var metadata = stat()
        guard fstat(fd, &metadata) == 0 else {
            let message = String(cString: strerror(errno))
            Darwin.close(fd)
            throw SidecarError.io("inspect shared-memory handoff: \(message)")
        }
        guard metadata.st_uid == geteuid(), metadata.st_nlink == 1,
            (metadata.st_mode & S_IFMT) == S_IFREG,
            (metadata.st_mode & 0o777) == 0o600,
            metadata.st_size == descriptor.capacityBytes
        else {
            Darwin.close(fd)
            throw SidecarError.io(
                "shared-memory handoff is not the worker-owned regular file described by EXECUTE")
        }
        let mapping = mmap(nil, descriptor.capacityBytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0)
        guard mapping != MAP_FAILED, let mapping else {
            let message = String(cString: strerror(errno))
            Darwin.close(fd)
            throw SidecarError.io("map shared-memory handoff: \(message)")
        }
        fileDescriptor = fd
        baseAddress = mapping
        self.descriptor = descriptor
        do {
            try validateInitialHeader()
        } catch {
            munmap(mapping, descriptor.capacityBytes)
            Darwin.close(fd)
            throw error
        }
    }

    deinit {
        munmap(baseAddress, descriptor.capacityBytes)
        Darwin.close(fileDescriptor)
    }

    public func begin() throws {
        guard !publishing, !published else {
            throw SidecarError.io("shared-memory handoff publication was already started")
        }
        try validateInitialHeader()
        // WRITING lets the worker distinguish a sidecar crash during publication
        // from a prediction failure that happened before handoff began.
        writeUInt64(sharedMemoryWriting, at: sharedMemoryStateOffset)
        publishing = true
    }

    public func zeroKV() throws {
        guard publishing, !published else {
            throw SidecarError.io("shared-memory K/V region cannot be cleared outside publication")
        }
        memset(baseAddress.advanced(by: descriptor.kvOffset), 0, descriptor.kvBytes)
    }

    public func withMutableBytes<Result>(
        offset: Int,
        count: Int,
        _ body: (UnsafeMutableRawBufferPointer) throws -> Result
    ) throws -> Result {
        guard publishing, !published else {
            throw SidecarError.io("shared-memory payload cannot be written outside publication")
        }
        let (end, overflow) = offset.addingReportingOverflow(count)
        guard offset >= sharedMemoryHeaderBytes, count >= 0, !overflow,
            end <= descriptor.capacityBytes
        else {
            throw SidecarError.io("shared-memory write is outside the mapped payload")
        }
        return try body(
            UnsafeMutableRawBufferPointer(
                start: baseAddress.advanced(by: offset),
                count: count
            ))
    }

    public func finish() throws -> String {
        guard publishing, !published else {
            throw SidecarError.io("shared-memory handoff publication is not active")
        }
        var hasher = SHA256()
        hasher.update(
            data: dataView(offset: descriptor.logitsOffset, count: descriptor.logitsBytes))
        hasher.update(data: dataView(offset: descriptor.kvOffset, count: descriptor.kvBytes))
        let digest = Array(hasher.finalize())
        digest.withUnsafeBytes { bytes in
            baseAddress.advanced(by: sharedMemoryDigestOffset).copyMemory(
                from: bytes.baseAddress!,
                byteCount: sharedMemoryDigestBytes
            )
        }
        writeUInt64(sharedMemoryReady, at: sharedMemoryStateOffset)
        published = true
        return digest.map { String(format: "%02x", $0) }.joined()
    }

    private func validateInitialHeader() throws {
        let magic = UnsafeRawBufferPointer(start: baseAddress, count: sharedMemoryMagic.count)
        guard Array(magic) == sharedMemoryMagic,
            readUInt64(at: sharedMemoryProtocolOffset) == UInt64(sidecarProtocolVersion),
            readUInt64(at: sharedMemoryStateOffset) == sharedMemoryEmpty,
            readUInt64(at: sharedMemoryGenerationOffset) == descriptor.generation,
            readUInt64(at: sharedMemoryLogitsOffsetOffset) == UInt64(descriptor.logitsOffset),
            readUInt64(at: sharedMemoryLogitsBytesOffset) == UInt64(descriptor.logitsBytes),
            readUInt64(at: sharedMemoryKVOffsetOffset) == UInt64(descriptor.kvOffset),
            readUInt64(at: sharedMemoryKVBytesOffset) == UInt64(descriptor.kvBytes)
        else {
            throw SidecarError.io("shared-memory control header does not match EXECUTE")
        }
    }

    private func dataView(offset: Int, count: Int) -> Data {
        Data(
            bytesNoCopy: baseAddress.advanced(by: offset),
            count: count,
            deallocator: .none
        )
    }

    private func readUInt64(at offset: Int) -> UInt64 {
        UInt64(littleEndian: baseAddress.advanced(by: offset).loadUnaligned(as: UInt64.self))
    }

    private func writeUInt64(_ value: UInt64, at offset: Int) {
        baseAddress.advanced(by: offset).storeBytes(of: value.littleEndian, as: UInt64.self)
    }
}

public func sharedMemoryInitialHeader(for descriptor: SharedMemoryHandoffDescriptor) -> Data {
    var header = Data(repeating: 0, count: sharedMemoryHeaderBytes)
    header.replaceSubrange(0..<sharedMemoryMagic.count, with: sharedMemoryMagic)
    writeUInt64(UInt64(sidecarProtocolVersion), to: &header, at: sharedMemoryProtocolOffset)
    writeUInt64(sharedMemoryEmpty, to: &header, at: sharedMemoryStateOffset)
    writeUInt64(descriptor.generation, to: &header, at: sharedMemoryGenerationOffset)
    writeUInt64(UInt64(descriptor.logitsOffset), to: &header, at: sharedMemoryLogitsOffsetOffset)
    writeUInt64(UInt64(descriptor.logitsBytes), to: &header, at: sharedMemoryLogitsBytesOffset)
    writeUInt64(UInt64(descriptor.kvOffset), to: &header, at: sharedMemoryKVOffsetOffset)
    writeUInt64(UInt64(descriptor.kvBytes), to: &header, at: sharedMemoryKVBytesOffset)
    return header
}

private func writeUInt64(_ value: UInt64, to data: inout Data, at offset: Int) {
    var littleEndian = value.littleEndian
    withUnsafeBytes(of: &littleEndian) { bytes in
        data.replaceSubrange(offset..<offset + bytes.count, with: bytes)
    }
}
