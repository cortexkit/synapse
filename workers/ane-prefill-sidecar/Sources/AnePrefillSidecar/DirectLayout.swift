import Foundation

/// Copies active CoreML values directly into the padded Metal cache layout.
/// The caller clears the destination first, keeping every padding position zero.
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
    let requiredSourceCount = zip(tensorShape, strides).reduce(1) {
        $0 + ($1.0 - 1) * $1.1
    }
    let expectedDestinationCount = layers * 2 * kvHeads * cacheTokens * headDimension
    guard source.count >= requiredSourceCount, destination.count == expectedDestinationCount else {
        throw SidecarError.kvConversion("K/V storage is smaller than the declared layout")
    }

    for head in 0..<kvHeads {
        for position in 0..<activeTokens {
            let sourceBase = head * strides[1] + position * strides[2]
            let destinationBase =
                (((layer * 2 + keyValueIndex) * kvHeads + head) * cacheTokens + position)
                * headDimension
            for dimension in 0..<headDimension {
                destination[destinationBase + dimension] =
                    source[sourceBase + dimension * strides[3]].littleEndian
            }
        }
    }
}
