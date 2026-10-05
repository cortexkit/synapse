/// Dynamic-weight 1x1 convolution (conv1x1 with weights from a variable tensor).
///
/// Unlike `ConvOp`, the weights are NOT constant blobs — they come from a
/// runtime tensor (slice/reshape of the input IOSurface), enabling the
/// dynamic-weight training pattern used in rustane.
///
/// MIL equivalent:
///   conv(x = acts_f16, weight = dyn_weights_f16, ...)
/// where dyn_weights_f16 is a reshaped slice of the input placeholder.
#[derive(Clone)]
pub struct DynConvOp {
    pub name: String,
    /// Source (activation) tensor name.
    pub source: String,
    /// Dynamic weight tensor name (must have shape [OC, IC, 1, 1]).
    pub weight_source: String,
    /// Output tensor name.
    pub top: String,
    pub input_channels: usize,
    pub output_channels: usize,
}
