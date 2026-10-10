//! Hardware cells derived from catalog declarations, without module dependencies.
use std::collections::BTreeMap;

/// Hardware rows and the catalog backend each certifies. Models are not a row
/// property: adding or removing a catalog backend changes the required cells.
pub const ROW_BACKENDS: [(&str, &str); 8] = [
    ("vulkan-windows-amd", "vulkan"),
    ("vulkan-linux-amd", "vulkan"),
    ("vulkan-linux-nvidia", "vulkan"),
    ("vulkan-windows-nvidia", "vulkan"),
    ("cuda-linux-nvidia", "cuda"),
    ("cuda-windows-nvidia", "cuda"),
    ("metal-m5", "metal"),
    ("ane-m5", "ane"),
];

/// `(catalog_id, backend, engine, fingerprint)`, supplied by the candidate's
/// compiled catalog rather than by checkout metadata or a certification record.
pub type CatalogLane = (String, String, String, String);

/// Plain catalog data at the module-to-validator boundary. Tasks are keyed by
/// catalog id and contain the catalog's `embed` or `rerank` task.
#[derive(Clone, Debug, Default)]
pub struct Catalog {
    pub lanes: Vec<CatalogLane>,
    pub tasks: BTreeMap<String, String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Cell {
    pub row_id: String,
    pub model: String,
    pub engine: String,
    pub fingerprint: String,
}

pub fn row_backend(row: &str) -> Option<&'static str> {
    ROW_BACKENDS
        .iter()
        .find(|(id, _)| *id == row)
        .map(|(_, backend)| *backend)
}

/// A pure derivation over lane declarations. An undeclared backend never gets
/// a cell, even if another backend for the same catalog id exists.
pub fn derive_cells(lanes: &[CatalogLane]) -> Vec<Cell> {
    ROW_BACKENDS
        .iter()
        .flat_map(|(row, backend)| {
            lanes
                .iter()
                .filter(move |(_, declared, _, _)| declared == backend)
                .map(move |(model, _, engine, fingerprint)| Cell {
                    row_id: (*row).into(),
                    model: model.clone(),
                    engine: engine.clone(),
                    fingerprint: fingerprint.clone(),
                })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cells_follow_only_declared_backends_and_their_removal() {
        let mut lanes = vec![
            (
                "first".into(),
                "metal".into(),
                "owned-metal".into(),
                "m".into(),
            ),
            (
                "first".into(),
                "cuda".into(),
                "owned-cuda".into(),
                "c".into(),
            ),
            (
                "second".into(),
                "vulkan".into(),
                "owned-vulkan".into(),
                "v".into(),
            ),
        ];
        let cells = derive_cells(&lanes);
        assert_eq!(cells.len(), 7);
        for (row, backend) in ROW_BACKENDS {
            for model in ["first", "second"] {
                assert_eq!(
                    cells
                        .iter()
                        .any(|cell| cell.row_id == row && cell.model == model),
                    lanes
                        .iter()
                        .any(|lane| lane.0 == model && lane.1 == backend),
                    "{row}/{model}"
                );
            }
        }
        assert_eq!(cells[0].fingerprint, "v");
        lanes.retain(|lane| lane.1 != "cuda");
        let after = derive_cells(&lanes);
        assert_eq!(after.len(), 5);
        assert!(after.iter().all(|cell| !cell.row_id.starts_with("cuda-")));
        assert_eq!(
            after,
            cells
                .into_iter()
                .filter(|cell| !cell.row_id.starts_with("cuda-"))
                .collect::<Vec<_>>()
        );
        assert!(derive_cells(&[]).is_empty());
    }
}
