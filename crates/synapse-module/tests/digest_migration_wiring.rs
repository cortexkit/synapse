use syn::{visit::Visit, Block, Expr, ExprCall, ImplItem, Item};

#[derive(Default)]
struct MigrationCalls(Vec<String>);

impl<'ast> Visit<'ast> for MigrationCalls {
    fn visit_expr_call(&mut self, call: &'ast ExprCall) {
        if let Expr::Path(path) = &*call.func {
            let name = path
                .path
                .segments
                .iter()
                .map(|segment| segment.ident.to_string())
                .collect::<Vec<_>>()
                .join("::");
            if matches!(
                name.as_str(),
                "ModelCache::default_root"
                    | "migrate_compiled_catalog_digests"
                    | "sync_and_load_catalog_models"
            ) {
                self.0.push(name);
            }
        }
        syn::visit::visit_expr_call(self, call);
    }
}

fn calls(block: &Block) -> Vec<String> {
    let mut visitor = MigrationCalls::default();
    visitor.visit_block(block);
    visitor.0
}

#[test]
fn initialize_migrates_digests_after_cache_root_and_before_catalog_reconciliation() {
    let file = syn::parse_file(include_str!("../src/lib.rs")).unwrap();
    let blocks = file
        .items
        .into_iter()
        .filter_map(|item| match item {
            Item::Impl(item) => Some(item.items),
            _ => None,
        })
        .flatten()
        .filter_map(|item| match item {
            ImplItem::Fn(method) if method.sig.ident == "initialize" => Some(method.block),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(blocks.len(), 1, "expected exactly one initialize method");
    assert_eq!(
        calls(&blocks[0]),
        [
            "ModelCache::default_root",
            "migrate_compiled_catalog_digests",
            "sync_and_load_catalog_models",
        ]
    );
}

#[test]
fn migration_order_guard_detects_missing_duplicate_and_misordered_calls() {
    let expected = [
        "ModelCache::default_root",
        "migrate_compiled_catalog_digests",
        "sync_and_load_catalog_models",
    ];
    for block in [
        syn::parse_quote!({
            ModelCache::default_root();
            sync_and_load_catalog_models();
        }),
        syn::parse_quote!({
            migrate_compiled_catalog_digests();
            ModelCache::default_root();
            sync_and_load_catalog_models();
        }),
        syn::parse_quote!({
            ModelCache::default_root();
            sync_and_load_catalog_models();
            migrate_compiled_catalog_digests();
        }),
        syn::parse_quote!({
            ModelCache::default_root();
            migrate_compiled_catalog_digests();
            migrate_compiled_catalog_digests();
            sync_and_load_catalog_models();
        }),
    ] {
        assert_ne!(calls(&block), expected);
    }
}
