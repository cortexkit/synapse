use super::*;

impl SynapseStore {
    /// Restore supported registrations without reviving retired engines.
    ///
    /// Core ML rows stay in the table until the legacy-model migration removes
    /// them. Other disallowed embed/rerank rows are deleted and their blob pins
    /// released unless a remaining registration still uses the blob. Reserved
    /// catalog ids stay in the table but are not registered by this path.
    pub fn reconcile_persisted_registrations(
        &self,
        is_reserved: &dyn Fn(&str) -> bool,
        cache: &dyn CatalogBlobCache,
    ) -> Result<RegistrationReconciliation, SynapseStoreError> {
        let reconciliation = self.store.with_conn_fenced(|tx| {
            tx.execute(
                "DELETE FROM catalog_file_verifications WHERE NOT EXISTS (
                SELECT 1 FROM catalog_install_members m WHERE
                m.catalog_id=catalog_file_verifications.catalog_id AND
                m.manifest_digest=catalog_file_verifications.manifest_digest AND
                m.backend=catalog_file_verifications.backend AND
                m.path=catalog_file_verifications.path AND
                m.digest=catalog_file_verifications.expected_sha256)",
                [],
            )?;
            let mut stmt = tx.prepare(
                "SELECT model_id, engine, task, config_json FROM models
                 ORDER BY created_ms ASC, model_id ASC",
            )?;
            let rows = stmt
                .query_map([], |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, String>(2)?,
                        row.get::<_, Vec<u8>>(3)?,
                    ))
                })?
                .collect::<Result<Vec<_>, _>>()?;
            drop(stmt);

            let mut reconciliation = RegistrationReconciliation::default();
            let mut released = BTreeSet::new();
            for (model_id, engine, task, config_json) in rows {
                let embed_or_rerank = task == "embed" || task == "rerank";
                if matches!(
                    engine.as_str(),
                    "ane" | "ane-coreml-worker" | "coreml" | "neural_engine"
                ) {
                    reconciliation.skipped_retired.push((model_id, engine));
                } else if embed_or_rerank && !OWNED_EMBED_RERANK_ENGINES.contains(&engine.as_str())
                {
                    tx.execute("DELETE FROM models WHERE model_id = ?1", params![model_id])?;
                    released.extend(registration_blob_digests(&config_json));
                    reconciliation.deleted.push((model_id, engine));
                } else if is_reserved(&model_id) {
                    reconciliation.skipped_reserved.push(model_id);
                } else {
                    reconciliation
                        .registrable
                        .push(decode_model_config(config_json)?);
                }
            }
            for digest in released {
                if free_form_references_digest_tx(tx, &digest)? {
                    continue;
                }
                cache
                    .remove_pins(&digest)
                    .map_err(|error| cache_error("remove pins of", &digest, error))?;
                reconciliation.unpinned_digests.push(digest);
            }
            Ok(reconciliation)
        })?;
        Ok(reconciliation)
    }
}
