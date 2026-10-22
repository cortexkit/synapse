use std::collections::HashSet;

use objc2_foundation::NSQualityOfService;

use crate::ops::weights::WeightBlob;
use crate::{ConstantOp, Op, Shape};

use super::Graph;
use super::tensor::Tensor;

/// Minimum spatial width (the W dimension in NCHW) that the ANE hardware
/// can execute.  Programs compiled with smaller widths succeed at the MIL
/// level but fail at runtime with `ANEProgramProcessRequestDirect` status
/// 0x1d.  Discovered empirically on M-series (M1–M5) / macOS 25–26.
pub const MIN_SPATIAL_WIDTH: usize = 64;

impl Graph {
    #[allow(clippy::type_complexity)]
    pub(crate) fn to_ops_and_shapes(&self) -> (Box<[Op]>, Box<[(String, Shape)]>) {
        let mut shapes: Vec<(String, Shape)> = self
            .inputs
            .iter()
            .map(|t| (Self::tensor_name(*t), t.shape))
            .collect();

        let regular_ops: Vec<Op> = self
            .ops
            .iter()
            .map(|op| {
                shapes.push((Self::tensor_name(op.output), op.output.shape));
                op.op.clone()
            })
            .collect();

        let all_bottoms: HashSet<&str> = regular_ops
            .iter()
            .flat_map(|op| op.bottom_names())
            .collect();

        // Emit constants in tensor-id order. `constants` is a HashMap, whose
        // iteration order differs between map instances; following it made
        // the MIL text and the packing of the weight blob differ on every
        // build of the same graph. The private compiler identifies a program
        // by hashes of exactly those bytes, and aned keeps compiled programs
        // on disk under that identifier, so only byte-identical sources can
        // reuse an earlier compile. The order does not change the program's
        // results: each constant is still declared once, before any use.
        let mut constants: Vec<_> = self.constants.iter().collect();
        constants.sort_unstable_by_key(|(id, _)| **id);
        let mut all_ops: Vec<Op> = Vec::new();
        for (&id, (data, shape)) in constants {
            let name = Self::tensor_name(Tensor { id, shape: *shape });
            if all_bottoms.contains(name.as_str()) {
                shapes.push((name.clone(), *shape));
                all_ops.push(Op::Constant(ConstantOp {
                    name: format!("const_{name}"),
                    top: name,
                    data: WeightBlob::from_f16_bytes(data.clone()),
                }));
            }
        }
        all_ops.extend(regular_ops);

        (all_ops.into_boxed_slice(), shapes.into_boxed_slice())
    }

    /// Return the exact MIL text and packed weights submitted to the private
    /// compiler. This measures source payloads, not opaque compiled programs.
    #[doc(hidden)]
    pub fn source_payload(&self) -> (String, Box<[u8]>) {
        let (ops, shapes) = self.to_ops_and_shapes();
        crate::ops::mil::emit_mil(&ops, &shapes)
    }

    /// Compile this graph to an ANE executable.
    ///
    /// Returns [`crate::Error::SpatialWidthTooSmall`] if any placeholder has a
    /// width smaller than [`MIN_SPATIAL_WIDTH`].  Pad the width dimension
    /// to at least 64 before compiling.
    pub fn compile(
        &self,
        quality_of_service: NSQualityOfService,
    ) -> Result<crate::Executable, crate::Error> {
        self.compile_gated(quality_of_service, || Ok::<(), crate::Error>(()))
    }

    /// [`Graph::compile`], calling `before_compile` only when the program has
    /// to be compiled, just before its sources are written. A program that
    /// aned already holds compiled is loaded without calling it. An error from
    /// `before_compile` is returned unchanged, and nothing is compiled.
    pub fn compile_gated<E: From<crate::Error>>(
        &self,
        quality_of_service: NSQualityOfService,
        before_compile: impl FnOnce() -> Result<(), E>,
    ) -> Result<crate::Executable, E> {
        for input in &self.inputs {
            if input.shape.width < MIN_SPATIAL_WIDTH {
                return Err(crate::Error::SpatialWidthTooSmall {
                    name: Self::tensor_name(*input),
                    width: input.shape.width,
                    min: MIN_SPATIAL_WIDTH,
                }
                .into());
            }
        }
        crate::client::compile_network_gated(self, quality_of_service, before_compile)
    }

    /// aned's identifier for this graph's program, without compiling it. See
    /// [`crate::client::program_identifier`].
    #[doc(hidden)]
    pub fn program_identifier(&self) -> Result<Option<String>, crate::Error> {
        crate::client::program_identifier(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A graph with many constants, so that two maps holding them are very
    /// unlikely to iterate in the same order by chance.
    fn graph_with_constants() -> Graph {
        let shape = Shape {
            batch: 1,
            channels: 8,
            height: 1,
            width: MIN_SPATIAL_WIDTH,
        };
        let mut graph = Graph::new();
        let mut value = graph.placeholder(shape);
        for step in 0..24 {
            let data: Vec<f32> = (0..shape.total_elements())
                .map(|index| step as f32 + index as f32 / 1024.0)
                .collect();
            let constant = graph.constant(&data, shape);
            value = graph.addition(value, constant);
        }
        let scale = graph.constant_with_scalar(0.5, Shape::channels(1));
        let _ = graph.multiplication(value, scale);
        graph
    }

    #[test]
    fn identical_graphs_emit_identical_sources() {
        let (mil_text, weight_bytes) = graph_with_constants().source_payload();
        assert!(!weight_bytes.is_empty());
        for build in 1..8 {
            let (other_mil, other_weights) = graph_with_constants().source_payload();
            assert!(
                other_mil == mil_text,
                "build {build}: the MIL text of an identical graph changed"
            );
            assert!(
                other_weights == weight_bytes,
                "build {build}: the weight bytes of an identical graph changed"
            );
        }
    }

    #[test]
    fn constants_are_declared_in_tensor_id_order() {
        let (ops, _) = graph_with_constants().to_ops_and_shapes();
        let ids: Vec<usize> = ops
            .iter()
            .filter_map(|op| match op {
                Op::Constant(constant) => Some(constant.top[1..].parse().unwrap()),
                _ => None,
            })
            .collect();
        assert_eq!(ids.len(), 25);
        assert!(
            ids.windows(2).all(|pair| pair[0] < pair[1]),
            "constants out of tensor-id order: {ids:?}"
        );
    }

    #[test]
    fn identical_graphs_get_the_same_compiler_identifier() {
        // Computing the identifier hashes the descriptor's bytes; it neither
        // compiles nor talks to the Neural Engine.
        let identifier = graph_with_constants().program_identifier().unwrap();
        let identifier = identifier.expect("the framework reports an identifier");
        assert_eq!(identifier.split('_').count(), 3, "{identifier}");
        for build in 1..4 {
            assert_eq!(
                graph_with_constants()
                    .program_identifier()
                    .unwrap()
                    .as_deref(),
                Some(identifier.as_str()),
                "build {build}"
            );
        }
    }
}
