use std::ops::{Deref, DerefMut};
use std::ptr;

use objc2::rc::Retained;
use objc2_io_surface::{IOSurface, IOSurfaceLockOptions};

use crate::Shape;
use crate::io_surface::IOSurfaceExt;

/// IOSurface-backed tensor storage for ANE I/O.
///
/// The underlying IOSurface is sized for **fp16** (2 bytes per element).
/// MIL function signatures declare inputs/outputs as `tensor<fp16, ...>`.
/// CPU-side staging converts f32→fp16 on write, fp16→f32 on read.
///
/// For zero-copy fp32 access (with automatic conversion), use
/// [`as_f32_slice`](Self::as_f32_slice) and [`as_f32_slice_mut`](Self::as_f32_slice_mut).
pub struct TensorData {
    surface: Retained<IOSurface>,
    shape: Shape,
    /// Scratch buffer for f32↔f16 conversion (avoids alloc per lock)
    f32_buf: std::cell::UnsafeCell<Vec<f32>>,
}

unsafe impl Send for TensorData {}
unsafe impl Sync for TensorData {}

impl TensorData {
    /// Allocate an empty IOSurface sized for the given shape (fp16 = 2 bytes/element).
    pub fn new(shape: Shape) -> Self {
        let elements = shape.total_elements();
        let byte_count = elements * 2; // fp16 = 2 bytes
        let surface = IOSurface::with_byte_count(byte_count);
        Self {
            surface,
            shape,
            f32_buf: std::cell::UnsafeCell::new(vec![0.0f32; elements]),
        }
    }

    /// Allocate an IOSurface and write fp32 data into it (converts to fp16).
    pub fn with_f32(data: &[f32], shape: Shape) -> Self {
        let tensor_data = Self::new(shape);
        tensor_data.copy_from_f32(data);
        tensor_data
    }

    /// Wrap an existing IOSurface.
    pub fn from_surface(surface: Retained<IOSurface>, shape: Shape) -> Self {
        let elements = shape.total_elements();
        Self {
            surface,
            shape,
            f32_buf: std::cell::UnsafeCell::new(vec![0.0f32; elements]),
        }
    }

    /// Write fp32 data into the surface (converts to fp16 via NEON SIMD).
    pub fn copy_from_f32(&self, data: &[f32]) {
        unsafe {
            self.surface
                .lockWithOptions_seed(IOSurfaceLockOptions(0), ptr::null_mut());
            let dst = std::slice::from_raw_parts_mut(
                self.surface.baseAddress().as_ptr().cast::<u16>(),
                data.len(),
            );
            crate::neon_convert::f32_to_f16_bulk(data, dst);
            self.surface
                .unlockWithOptions_seed(IOSurfaceLockOptions(0), ptr::null_mut());
        }
    }

    /// Lock the surface and return an RAII guard exposing `&[f32]`.
    /// The guard reads fp16 from the surface and presents it as f32 via a scratch buffer.
    /// **Note:** This is a COPY, not a zero-copy view. Writes to the returned slice
    /// do NOT propagate back to the IOSurface. Use `as_f32_slice_mut` for writes.
    pub fn as_f32_slice(&self) -> LockedSlice<'_> {
        let element_count = self.shape.total_elements();
        self.surface
            .lockWithOptions_seed(IOSurfaceLockOptions::ReadOnly, ptr::null_mut());
        // Convert fp16 → f32 into scratch buffer via NEON SIMD
        let buf = unsafe { &mut *self.f32_buf.get() };
        unsafe {
            let src = std::slice::from_raw_parts(
                self.surface.baseAddress().as_ptr().cast::<u16>(),
                element_count,
            );
            crate::neon_convert::f16_to_f32_bulk(src, buf);
        }
        LockedSlice {
            surface: &self.surface,
            pointer: buf.as_ptr(),
            element_count,
        }
    }

    /// Lock the surface for writing and return an RAII guard exposing `&mut [f32]`.
    /// Writes to the returned slice are converted to fp16 and flushed to the IOSurface
    /// when the guard is dropped.
    pub fn as_f32_slice_mut(&self) -> LockedSliceMut<'_> {
        let element_count = self.shape.total_elements();
        self.surface
            .lockWithOptions_seed(IOSurfaceLockOptions(0), ptr::null_mut());
        let buf = unsafe { &mut *self.f32_buf.get() };
        // NOTE: we do NOT read current surface data into the buffer.
        // The caller must write ALL needed data before the guard is dropped.
        // This saves one full fp16→f32 conversion per lock (massive perf win).
        // If you need read-modify-write, use as_f32_slice() first then as_f32_slice_mut().
        LockedSliceMut {
            surface: &self.surface,
            pointer: buf.as_mut_ptr(),
            element_count,
        }
    }

    /// Read the surface contents back as fp32 values (allocating).
    pub fn read_f32(&self) -> Box<[f32]> {
        let slice = self.as_f32_slice();
        slice.to_vec().into_boxed_slice()
    }

    /// Write a single f32 value at a specific flat index (converts to fp16).
    /// Caller must lock/unlock around batches of writes using `lock`/`unlock`.
    pub fn write_f32_at(&self, index: usize, value: f32) {
        unsafe {
            self.surface
                .lockWithOptions_seed(IOSurfaceLockOptions(0), ptr::null_mut());
            let ptr = self.surface.baseAddress().as_ptr().cast::<u16>();
            *ptr.add(index) = crate::ops::weights::f32_to_f16(value);
            self.surface
                .unlockWithOptions_seed(IOSurfaceLockOptions(0), ptr::null_mut());
        }
    }

    /// Write multiple f32 values at specific flat indices (batch, one lock).
    pub fn write_f32_sparse(&self, indices: &[usize], values: &[f32]) {
        unsafe {
            self.surface
                .lockWithOptions_seed(IOSurfaceLockOptions(0), ptr::null_mut());
            let ptr = self.surface.baseAddress().as_ptr().cast::<u16>();
            for (&idx, &val) in indices.iter().zip(values.iter()) {
                *ptr.add(idx) = crate::ops::weights::f32_to_f16(val);
            }
            self.surface
                .unlockWithOptions_seed(IOSurfaceLockOptions(0), ptr::null_mut());
        }
    }

    pub fn shape(&self) -> Shape {
        self.shape
    }

    pub fn surface(&self) -> &IOSurface {
        &self.surface
    }
}

/// RAII guard that holds a read-only lock on an IOSurface and derefs to `&[f32]`.
pub struct LockedSlice<'a> {
    surface: &'a IOSurface,
    pointer: *const f32,
    element_count: usize,
}

impl Deref for LockedSlice<'_> {
    type Target = [f32];
    fn deref(&self) -> &[f32] {
        unsafe { std::slice::from_raw_parts(self.pointer, self.element_count) }
    }
}

impl Drop for LockedSlice<'_> {
    fn drop(&mut self) {
        self.surface
            .unlockWithOptions_seed(IOSurfaceLockOptions::ReadOnly, ptr::null_mut());
    }
}

/// RAII guard that holds a read-write lock on an IOSurface and derefs to `&mut [f32]`.
pub struct LockedSliceMut<'a> {
    surface: &'a IOSurface,
    pointer: *mut f32,
    element_count: usize,
}

impl Deref for LockedSliceMut<'_> {
    type Target = [f32];
    fn deref(&self) -> &[f32] {
        unsafe { std::slice::from_raw_parts(self.pointer, self.element_count) }
    }
}

impl DerefMut for LockedSliceMut<'_> {
    fn deref_mut(&mut self) -> &mut [f32] {
        unsafe { std::slice::from_raw_parts_mut(self.pointer, self.element_count) }
    }
}

impl Drop for LockedSliceMut<'_> {
    fn drop(&mut self) {
        // Flush f32 scratch buffer → fp16 IOSurface via NEON SIMD
        unsafe {
            let src = std::slice::from_raw_parts(self.pointer, self.element_count);
            let dst = std::slice::from_raw_parts_mut(
                self.surface.baseAddress().as_ptr().cast::<u16>(),
                self.element_count,
            );
            crate::neon_convert::f32_to_f16_bulk(src, dst);
        }
        self.surface
            .unlockWithOptions_seed(IOSurfaceLockOptions(0), ptr::null_mut());
    }
}
