//! Offline validation using candidate-owned catalog pins and embedded references.
use crate::{
    derive_cells, hex, identity, refuse, safe_file, self_check, Catalog, Record, Result,
    RECORD_SCHEMA,
};
use std::collections::BTreeSet;
use std::path::Path;
use synapse_parity::evaluator::{fixture_set_id, FixtureSet};
use synapse_parity::manifest::ToleranceClass;

/// Rechecks a record's claims without executing the candidate or running Git.
/// Only fixture corpus bytes come from the checkout; their paths and digests,
/// profile settings and architecture come from the candidate's embedded data.
pub fn validate_record(
    record: &Record,
    checkout: &Path,
    assets: &Path,
    catalog: &Catalog,
) -> Result<()> {
    identity(record)?;
    let cells = derive_cells(&catalog.lanes);
    let cell = cells
        .iter()
        .find(|cell| cell.row_id == record.row_id && cell.model == record.model)
        .ok_or_else(|| {
            refuse(format!(
                "undeclared_certification_cell: {}/{}",
                record.row_id, record.model
            ))
        })?;
    if record.status != "passed" {
        return Err(refuse("invalid_certification_status"));
    }
    if !record.parity.passed() {
        return Err(refuse("parity_gate_failed"));
    }
    if !record.admission.passed() {
        return Err(refuse("admission_gate_failed"));
    }
    let manifest = self_check::manifest()?;
    let model = manifest
        .model(&record.model)
        .map_err(|error| refuse(error.to_string()))?;
    let expected_id = fixture_set_id(&manifest, &record.model);
    if record.parity.fixture_set_id != expected_id {
        return Err(refuse("fixture_set_id_mismatch"));
    }
    let index = self_check::source_index()?;
    let entry = &index[&expected_id];
    let path = entry["path"]
        .as_str()
        .ok_or_else(|| refuse("embedded_fixture_index_invalid"))?;
    let digest = entry["sha256"]
        .as_str()
        .ok_or_else(|| refuse("embedded_fixture_index_invalid"))?;
    safe_file(path)?;
    let bytes = std::fs::read(checkout.join("bench/parity").join(path))
        .map_err(|error| refuse(format!("fixture_file_unavailable: {path}: {error}")))?;
    let fixtures = FixtureSet::from_bytes(&manifest, &record.model, &bytes, digest)
        .map_err(|error| refuse(error.to_string()))?;
    let count = fixtures.cases().len();
    if record.parity.completed_fixtures != count || record.parity.expected_fixtures != count {
        return Err(refuse("fixture_count_mismatch"));
    }
    if record.fingerprint != cell.fingerprint {
        return Err(refuse("catalog_fingerprint_mismatch"));
    }
    let expected_profile = format!("{}.{}", cell.model, cell.engine);
    if record.profile_id != expected_profile || record.parity.profile_id != expected_profile {
        return Err(refuse("catalog_profile_mismatch"));
    }
    let profile = manifest
        .profiles
        .get(&expected_profile)
        .ok_or_else(|| refuse("embedded_profile_missing"))?;
    let tolerance = match profile.rerank_tolerance_class {
        ToleranceClass::Fp32 => "fp32",
        ToleranceClass::Fp16 => "fp16",
    };
    if record.parity.tolerance_class != tolerance {
        return Err(refuse("tolerance_class_mismatch"));
    }
    let task = catalog
        .tasks
        .get(&record.model)
        .ok_or_else(|| refuse("catalog_task_missing"))?;
    if record.operation != *task || record.parity.operation != *task {
        return Err(refuse("catalog_operation_mismatch"));
    }
    if record.row_id == "ane-m5" {
        let placement = record
            .placement
            .as_ref()
            .ok_or_else(|| refuse("placement_missing"))?;
        let layers = model
            .architecture
            .int("num_hidden_layers")
            .map_err(|error| refuse(error.to_string()))? as usize;
        if placement.layer_count != layers {
            return Err(refuse("placement_layer_count_mismatch"));
        }
        if placement.admitted_count == 0 {
            return Err(refuse("placement_admission_missing"));
        }
        if placement.inventories.len() != placement.admitted_count {
            return Err(refuse("placement_inventory_count_mismatch"));
        }
        if !placement
            .inventories
            .iter()
            .all(|inventory| inventory.passed(layers, &record.model))
        {
            return Err(refuse("placement_inventory_failed"));
        }
    }
    for artifact in &record.executed_artifacts {
        if crate::digest(assets, &artifact.file)? != artifact.sha256 {
            return Err(refuse(format!(
                "executed artifact digest mismatch: {}",
                artifact.file
            )));
        }
    }
    Ok(())
}

/// Requires exactly the cells declared by the compiled catalog and applies all
/// per-record checks. A record's `passed` status alone never admits a cell.
pub fn validate(
    records: &[Record],
    checkout: &Path,
    assets: &Path,
    catalog: &Catalog,
) -> Result<Vec<(String, String)>> {
    let required = derive_cells(&catalog.lanes)
        .into_iter()
        .map(|cell| (cell.row_id, cell.model))
        .collect::<BTreeSet<_>>();
    let mut found = BTreeSet::new();
    for record in records {
        let key = (record.row_id.clone(), record.model.clone());
        if !required.contains(&key) {
            return Err(refuse(format!(
                "undeclared_certification_cell: {}/{}",
                key.0, key.1
            )));
        }
        if !found.insert(key) {
            return Err(refuse("duplicate_certification_cell"));
        }
        validate_record(record, checkout, assets, catalog)?;
    }
    if let Some((row, model)) = required.difference(&found).next() {
        return Err(refuse(format!("missing_certification_cell: {row}/{model}")));
    }
    Ok(found.into_iter().collect())
}

/// Enumerates the evidence directory, including extra records, rather than
/// reading only expected paths and silently overlooking an undeclared cell.
pub fn validate_checkout(
    checkout: &Path,
    assets: &Path,
    source: &str,
    catalog: &Catalog,
) -> Result<Vec<(String, String)>> {
    if !hex(source, 40) {
        return Err(refuse("invalid source commit"));
    }
    let directory = checkout.join("docs/evidence/certification").join(source);
    if !directory.exists() {
        return validate(&[], checkout, assets, catalog);
    }
    let mut records = Vec::new();
    let rows = std::fs::read_dir(&directory).map_err(|error| refuse(error.to_string()))?;
    for row in rows {
        let row = row.map_err(|error| refuse(error.to_string()))?;
        if !row
            .file_type()
            .map_err(|error| refuse(error.to_string()))?
            .is_dir()
        {
            return Err(refuse("record_path_mismatch"));
        }
        let row_id = row
            .file_name()
            .into_string()
            .map_err(|_| refuse("record_path_mismatch"))?;
        for file in std::fs::read_dir(row.path()).map_err(|error| refuse(error.to_string()))? {
            let file = file.map_err(|error| refuse(error.to_string()))?;
            if !file
                .file_type()
                .map_err(|error| refuse(error.to_string()))?
                .is_file()
            {
                return Err(refuse("record_path_mismatch"));
            }
            let path = file.path();
            if path.extension().and_then(|s| s.to_str()) != Some("json") {
                return Err(refuse("record_path_mismatch"));
            }
            let bytes = std::fs::read(&path).map_err(|error| refuse(error.to_string()))?;
            // Check the version before decoding fields, so obsolete record fields
            // cannot obscure the named schema refusal.
            let value: serde_json::Value =
                serde_json::from_slice(&bytes).map_err(|error| refuse(error.to_string()))?;
            if value["schema"] != RECORD_SCHEMA {
                return Err(refuse("record_schema_mismatch"));
            }
            let record: Record =
                serde_json::from_value(value).map_err(|error| refuse(error.to_string()))?;
            if record.row_id != row_id
                || Some(record.model.as_str()) != path.file_stem().and_then(|s| s.to_str())
                || record.source_commit != source
            {
                return Err(refuse("record_path_mismatch"));
            }
            records.push(record);
        }
    }
    validate(&records, checkout, assets, catalog)
}
