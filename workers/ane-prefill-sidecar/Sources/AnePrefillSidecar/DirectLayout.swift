import CoreML
import Foundation

/// Owns the CoreML output views that point into one worker-provided K/V mapping.
/// CoreML writes each fixed-window output with cache-sized head strides, so the
/// returned bytes already use the Metal import order without a host copy.
public final class MappedKVOutputBackings {
    public let outputBackings: [String: Any]

    private let arrays: [String: MLMultiArray]
    private let window: Int
    private let kvHeads: Int
    private let cacheTokens: Int
    private let headDimension: Int

    public init(
        destination: UnsafeMutableRawBufferPointer,
        window: Int,
        layers: Int,
        kvHeads: Int,
        cacheTokens: Int,
        headDimension: Int
    ) throws {
        let planeElements = try checkedKVProduct(
            [kvHeads, cacheTokens, headDimension],
            failure: "mapped K/V plane dimensions overflowed"
        )
        let totalElements = try checkedKVProduct(
            [layers, 2, planeElements],
            failure: "mapped K/V cache dimensions overflowed"
        )
        let expectedBytes = try checkedKVProduct(
            [totalElements, MemoryLayout<UInt16>.size],
            failure: "mapped K/V cache byte count overflowed"
        )
        guard window > 0, window <= cacheTokens,
            let baseAddress = destination.baseAddress,
            destination.count == expectedBytes,
            Int(bitPattern: baseAddress).isMultiple(of: 64)
        else {
            throw SidecarError.kvConversion(
                "mapped K/V backing does not match the installed cache layout")
        }

        let shape = [
            1,
            NSNumber(value: kvHeads),
            NSNumber(value: window),
            NSNumber(value: headDimension),
        ]
        let strides = [
            NSNumber(value: planeElements),
            NSNumber(value: cacheTokens * headDimension),
            NSNumber(value: headDimension),
            1,
        ]
        var mappedArrays: [String: MLMultiArray] = [:]
        mappedArrays.reserveCapacity(layers * 2)
        do {
            for layer in 0..<layers {
                for (keyValueIndex, outputName) in ["key", "value"].enumerated() {
                    let name = String(format: "%@_%02d", outputName, layer)
                    let plane = layer * 2 + keyValueIndex
                    let pointer = baseAddress.advanced(
                        by: plane * planeElements * MemoryLayout<UInt16>.size)
                    mappedArrays[name] = try MLMultiArray(
                        dataPointer: pointer,
                        shape: shape,
                        dataType: .float16,
                        strides: strides,
                        deallocator: { _ in }
                    )
                }
            }
        } catch {
            throw SidecarError.kvConversion(
                "CoreML could not create mapped K/V output backings: \(error)")
        }

        arrays = mappedArrays
        outputBackings = mappedArrays.mapValues { $0 as Any }
        self.window = window
        self.kvHeads = kvHeads
        self.cacheTokens = cacheTokens
        self.headDimension = headDimension
    }

    /// Checks that CoreML wrote through each `MLMultiArray` view owned by this mapping
    /// before callers use the mapped K/V data. Fixed-window outputs include right-padding
    /// positions, so those positions are cleared when the request has fewer active tokens.
    public func validateAndClearInactive(
        prediction: MLFeatureProvider,
        activeTokens: Int
    ) throws {
        guard activeTokens > 0, activeTokens <= window else {
            throw SidecarError.kvConversion(
                "active token count is outside the mapped K/V output window")
        }
        for (name, backing) in arrays {
            guard let returned = prediction.featureValue(for: name)?.multiArrayValue,
                returned === backing,
                returned.dataPointer == backing.dataPointer,
                returned.dataType == .float16,
                returned.shape == backing.shape,
                returned.strides == backing.strides
            else {
                throw SidecarError.kvConversion(
                    "CoreML did not use the mapped K/V output backing named \(name)")
            }
        }

        let inactiveTokens = window - activeTokens
        guard inactiveTokens > 0 else { return }
        let inactiveBytes = inactiveTokens * headDimension * MemoryLayout<UInt16>.size
        for backing in arrays.values {
            for head in 0..<kvHeads {
                let firstInactiveElement =
                    (head * cacheTokens + activeTokens) * headDimension
                memset(
                    backing.dataPointer.advanced(
                        by: firstInactiveElement * MemoryLayout<UInt16>.size),
                    0,
                    inactiveBytes
                )
            }
        }
    }
}

/// Copies active CoreML values directly into the padded Metal cache layout.
/// Copy-based adapters can use block copies for contiguous positions while preserving
/// a scalar stride walk for non-contiguous dimensions.
public func copyActiveKVToPaddedCache(
    source: UnsafeBufferPointer<UInt16>,
    tensorShape: [Int],
    strides: [Int],
    destination: UnsafeMutableBufferPointer<UInt16>,
    layer: Int,
    keyValueIndex: Int,
    activeTokens: Int,
    window: Int,
    layers: Int,
    kvHeads: Int,
    cacheTokens: Int,
    headDimension: Int
) throws {
    guard tensorShape == [1, kvHeads, window, headDimension],
        strides.count == tensorShape.count,
        strides.allSatisfy({ $0 > 0 }),
        activeTokens > 0,
        activeTokens <= window,
        activeTokens <= cacheTokens,
        (0..<layers).contains(layer),
        (0..<2).contains(keyValueIndex)
    else {
        throw SidecarError.kvConversion(
            "K/V output or destination does not match the installed fixed-window shape")
    }
    let requiredSourceCount = try requiredKVStorageCount(shape: tensorShape, strides: strides)
    let expectedDestinationCount = try checkedKVProduct(
        [layers, 2, kvHeads, cacheTokens, headDimension],
        failure: "K/V destination dimensions overflowed"
    )
    guard source.count >= requiredSourceCount,
        destination.count == expectedDestinationCount,
        let sourceAddress = source.baseAddress,
        let destinationAddress = destination.baseAddress
    else {
        throw SidecarError.kvConversion("K/V storage is smaller than the declared layout")
    }

    let rowBytes = headDimension * MemoryLayout<UInt16>.size
    for head in 0..<kvHeads {
        let sourceHeadBase = head * strides[1]
        let destinationHeadBase =
            (((layer * 2 + keyValueIndex) * kvHeads + head) * cacheTokens)
            * headDimension
        if strides[3] == 1, strides[2] == headDimension {
            UnsafeMutableRawPointer(destinationAddress.advanced(by: destinationHeadBase))
                .copyMemory(
                    from: UnsafeRawPointer(sourceAddress.advanced(by: sourceHeadBase)),
                    byteCount: activeTokens * rowBytes
                )
            continue
        }
        for position in 0..<activeTokens {
            let sourceBase = sourceHeadBase + position * strides[2]
            let destinationBase = destinationHeadBase + position * headDimension
            if strides[3] == 1 {
                UnsafeMutableRawPointer(destinationAddress.advanced(by: destinationBase))
                    .copyMemory(
                        from: UnsafeRawPointer(sourceAddress.advanced(by: sourceBase)),
                        byteCount: rowBytes
                    )
                continue
            }
            for dimension in 0..<headDimension {
                destinationAddress[destinationBase + dimension] =
                    sourceAddress[sourceBase + dimension * strides[3]].littleEndian
            }
        }
    }
}

private func requiredKVStorageCount(shape: [Int], strides: [Int]) throws -> Int {
    var largestOffset = 0
    for (dimension, stride) in zip(shape, strides) {
        let (term, multiplicationOverflow) = (dimension - 1).multipliedReportingOverflow(by: stride)
        let (next, additionOverflow) = largestOffset.addingReportingOverflow(term)
        guard !multiplicationOverflow, !additionOverflow else {
            throw SidecarError.kvConversion("K/V source stride calculation overflowed")
        }
        largestOffset = next
    }
    let (count, overflow) = largestOffset.addingReportingOverflow(1)
    guard !overflow else {
        throw SidecarError.kvConversion("K/V source stride calculation overflowed")
    }
    return count
}

private func checkedKVProduct(_ factors: [Int], failure: String) throws -> Int {
    var product = 1
    for factor in factors {
        guard factor > 0 else { throw SidecarError.kvConversion(failure) }
        let (next, overflow) = product.multipliedReportingOverflow(by: factor)
        guard !overflow else { throw SidecarError.kvConversion(failure) }
        product = next
    }
    return product
}
