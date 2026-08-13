import Foundation

/// A strided tensor view used by the CoreML adapter and unit tests. The payload
/// materializer never assumes CoreML's storage is contiguous.
public enum TensorStorage: Equatable, Sendable {
    case float16Bits([UInt16])
    case float32([Float])
    case float64([Double])

    var count: Int {
        switch self {
        case let .float16Bits(values): values.count
        case let .float32(values): values.count
        case let .float64(values): values.count
        }
    }
}

public struct StridedTensor: Equatable, Sendable {
    public let shape: [Int]
    public let strides: [Int]
    public let storage: TensorStorage

    public init(shape: [Int], strides: [Int], storage: TensorStorage) throws {
        guard !shape.isEmpty, shape.count == strides.count,
              shape.allSatisfy({ $0 > 0 }), strides.allSatisfy({ $0 > 0 })
        else {
            throw SidecarError.invalid("tensor shape and strides must have matching positive dimensions")
        }
        var largestOffset = 0
        for (dimension, stride) in zip(shape, strides) {
            let (term, overflow) = (dimension - 1).multipliedReportingOverflow(by: stride)
            guard !overflow else {
                throw SidecarError.invalid("tensor stride calculation overflowed")
            }
            let (next, additionOverflow) = largestOffset.addingReportingOverflow(term)
            guard !additionOverflow else {
                throw SidecarError.invalid("tensor stride calculation overflowed")
            }
            largestOffset = next
        }
        guard largestOffset < storage.count else {
            throw SidecarError.invalid("tensor storage is smaller than its shape and strides require")
        }
        self.shape = shape
        self.strides = strides
        self.storage = storage
    }

    func offset(_ indices: [Int]) throws -> Int {
        guard indices.count == shape.count else {
            throw SidecarError.invalid("tensor index rank does not match tensor shape")
        }
        var offset = 0
        for axis in indices.indices {
            let index = indices[axis]
            let dimension = shape[axis]
            guard (0 ..< dimension).contains(index) else {
                throw SidecarError.invalid("tensor index is outside its shape")
            }
            offset += index * strides[axis]
        }
        return offset
    }

    func floatValue(at offset: Int) throws -> Float {
        switch storage {
        case let .float16Bits(values):
            return Float(Float16(bitPattern: values[offset]))
        case let .float32(values):
            return values[offset]
        case let .float64(values):
            return Float(values[offset])
        }
    }

    func float16Bits(at offset: Int) throws -> UInt16 {
        guard case let .float16Bits(values) = storage else {
            throw SidecarError.invalid("K/V output must have float16 storage")
        }
        return values[offset]
    }
}

/// Fixed-window publication metadata. K/V bytes are ordered
/// `[layer][key_or_value][head][position][dimension]` and contain exactly the
/// active positions. The host writes those positions into its own cache; no padded
/// position is published for decode admission.
public struct PrefillPayload: Equatable, Sendable {
    public let activeTokens: Int
    public let vocabularySize: Int
    public let logits: [Float]
    public let kvBits: [UInt16]
    public let layers: Int
    public let kvHeads: Int
    public let headDimension: Int

    public init(
        activeTokens: Int,
        vocabularySize: Int,
        logits: [Float],
        kvBits: [UInt16],
        layers: Int,
        kvHeads: Int,
        headDimension: Int
    ) {
        self.activeTokens = activeTokens
        self.vocabularySize = vocabularySize
        self.logits = logits
        self.kvBits = kvBits
        self.layers = layers
        self.kvHeads = kvHeads
        self.headDimension = headDimension
    }

    public var kvByteCount: Int { kvBits.count * MemoryLayout<UInt16>.size }
    public var logitsByteCount: Int { logits.count * MemoryLayout<Float>.size }

    public var layout: [String: Any] {
        [
            "kind": "f16_le",
            "order": ["layer", "key_or_value", "head", "position", "dimension"],
            "layers": layers,
            "key_value_count": 2,
            "kv_heads": kvHeads,
            "active_tokens": activeTokens,
            "head_dimension": headDimension,
        ]
    }
}

/// Copies logits from the active token (`activeTokens - 1`), not the padded tail.
/// CoreML convolution exports commonly use `[1, vocab, window]`, while other
/// exporters use `[1, window, vocab]`; unit axes may appear around either form.
public func copyActiveLogits(
    _ tensor: StridedTensor,
    activeTokens: Int,
    window: Int,
    vocabularySize: Int
) throws -> [Float] {
    guard activeTokens > 0, activeTokens <= window else {
        throw SidecarError.invalid("active token count is outside the installed fixed window")
    }
    let sequenceAxes = tensor.shape.indices.filter { tensor.shape[$0] == window }
    let vocabularyAxes = tensor.shape.indices.filter { tensor.shape[$0] == vocabularySize }
    guard sequenceAxes.count == 1, vocabularyAxes.count == 1,
          let sequenceAxis = sequenceAxes.first, let vocabularyAxis = vocabularyAxes.first,
          sequenceAxis != vocabularyAxis,
          tensor.shape.indices.allSatisfy({ axis in
              axis == sequenceAxis || axis == vocabularyAxis || tensor.shape[axis] == 1
          })
    else {
        throw SidecarError.invalid(
            "logits shape \(tensor.shape) must contain exactly one window and vocabulary axis"
        )
    }

    var result = [Float](repeating: 0, count: vocabularySize)
    var index = [Int](repeating: 0, count: tensor.shape.count)
    index[sequenceAxis] = activeTokens - 1
    for vocabularyIndex in 0 ..< vocabularySize {
        index[vocabularyAxis] = vocabularyIndex
        result[vocabularyIndex] = try tensor.floatValue(at: tensor.offset(index))
    }
    return result
}

/// Copies one CoreML K/V output into the wire layout while walking CoreML's actual
/// strides. The model is fixed-width, but only active cache positions cross the
/// sidecar boundary so padding cannot become decode cache state.
public func appendActiveKV(
    _ tensor: StridedTensor,
    to destination: inout [UInt16],
    activeTokens: Int,
    window: Int,
    kvHeads: Int,
    headDimension: Int
) throws {
    guard activeTokens > 0, activeTokens <= window else {
        throw SidecarError.invalid("active token count is outside the installed fixed window")
    }
    guard tensor.shape == [1, kvHeads, window, headDimension] else {
        throw SidecarError.invalid(
            "K/V shape \(tensor.shape) must be [1, \(kvHeads), \(window), \(headDimension)]"
        )
    }
    let expectedAdditional = kvHeads * activeTokens * headDimension
    destination.reserveCapacity(destination.count + expectedAdditional)
    for head in 0 ..< kvHeads {
        for position in 0 ..< activeTokens {
            for dimension in 0 ..< headDimension {
                let offset = try tensor.offset([0, head, position, dimension])
                destination.append(try tensor.float16Bits(at: offset))
            }
        }
    }
}

public func littleEndianFloat32Frame(_ values: [Float]) -> Data {
    var data = Data(capacity: values.count * MemoryLayout<UInt32>.size)
    for value in values {
        var bits = value.bitPattern.littleEndian
        withUnsafeBytes(of: &bits) { data.append(contentsOf: $0) }
    }
    return data
}

public func littleEndianFloat16Frame(_ values: [UInt16]) -> Data {
    var data = Data(capacity: values.count * MemoryLayout<UInt16>.size)
    for value in values {
        var bits = value.littleEndian
        withUnsafeBytes(of: &bits) { data.append(contentsOf: $0) }
    }
    return data
}
