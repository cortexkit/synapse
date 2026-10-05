//! Rust bindings for Apple Neural Engine (ANE) via the private `AppleNeuralEngine.framework`.
//!
//! Provides a symbolic graph builder and the compile -> run lifecycle through
//! `_ANEInMemoryModel`, using IOSurface-backed zero-copy I/O.
//!
//! # Lifecycle
//!
//! ```ignore
//! let mut g = Graph::new();
//! let x   = g.placeholder(Shape::channels(64));
//! let w   = g.constant(&weights, Shape::spatial(64, 1, 1));
//! let out = g.convolution_2d_1x1(x, w, None);
//!
//! let executable = g.compile(NSQualityOfService::Default)?;
//!
//! let input  = TensorData::with_f32(&data, Shape::channels(64));
//! let output = TensorData::new(Shape::channels(64));
//! executable.run(&[&input], &[&output])?;
//! ```

#![cfg(target_os = "macos")]
#![allow(
    dead_code,
    deprecated,
    private_interfaces,
    clippy::too_many_arguments,
    clippy::not_unsafe_ptr_arg_deref,
    clippy::missing_safety_doc
)]

mod ane_client;
pub mod ane_in_memory_model;
pub mod ane_in_memory_model_descriptor;
mod ane_io_surface_object;
mod ane_performance_stats;
mod ane_request;
pub mod client;
pub mod diagnostics;

pub use objc2::rc::autoreleasepool;
mod error;
mod executable;
pub mod graph;
pub mod io_surface;
pub mod neon_convert;
pub mod ops;
pub(crate) mod request;
mod tensor_data;

pub use error::Error;
pub use executable::Executable;
pub use graph::{
    Convolution2dDescriptor, ConvolutionTranspose2dDescriptor, Graph, MIN_SPATIAL_WIDTH, Tensor,
};
pub use io_surface::IOSurfaceExt;
pub use objc2_foundation::NSQualityOfService;
pub use ops::{
    ActivationMode, ActivationOp, ConcatOp, ConstantOp, ConvOp, DeconvOp, ElementwiseOp,
    ElementwiseOpType, FlattenOp, InnerProductOp, InstanceNormOp, MatmulOp, Op, PadFillMode,
    PadMode, PaddingOp, PoolType, PoolingOp, ReductionMode, ReductionOp, ReshapeOp, ScalarOp,
    ScalarOpType, Shape, SliceBySizeOp, SoftmaxOp, TransposeOp,
};
pub use tensor_data::{LockedSlice, LockedSliceMut, TensorData};

/// Convert f32 values to IEEE 754 fp16 bytes (2 bytes per element, little-endian).
pub fn f32_to_fp16_bytes(values: &[f32]) -> Box<[u8]> {
    let mut bytes = vec![0u8; values.len() * 2];
    for (index, &value) in values.iter().enumerate() {
        let f16 = ops::weights::f32_to_f16(value);
        bytes[index * 2] = (f16 & 0xFF) as u8;
        bytes[index * 2 + 1] = (f16 >> 8) as u8;
    }
    bytes.into_boxed_slice()
}
pub mod coreml;
pub mod ffi;
