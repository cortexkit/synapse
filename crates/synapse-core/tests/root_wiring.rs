use syn::{visit::Visit, Block, Expr, ExprCall, ImplItem, Item, Type};

#[derive(Default)]
struct Calls {
    environment: Vec<String>,
    resolvers: Vec<String>,
}

impl<'ast> Visit<'ast> for Calls {
    fn visit_expr_call(&mut self, call: &'ast ExprCall) {
        if let Expr::Path(path) = &*call.func {
            let names = path
                .path
                .segments
                .iter()
                .map(|segment| segment.ident.to_string())
                .collect::<Vec<_>>();
            if names.starts_with(&["std".to_string(), "env".to_string()])
                || (names.first().is_some_and(|name| name == "env")
                    && names
                        .get(1)
                        .is_some_and(|name| name == "var" || name == "var_os"))
            {
                self.environment.push(names.join("::"));
            }
            if let Some(name) = names.last().filter(|name| {
                *name == "resolve_data_root" || *name == "resolve_data_root_from_process"
            }) {
                self.resolvers.push(name.clone());
            }
        }
        syn::visit::visit_expr_call(self, call);
    }
}

fn wiring_errors(block: &Block, resolver: &str) -> Vec<String> {
    let mut calls = Calls::default();
    calls.visit_block(block);
    let mut errors = calls.environment;
    if !calls.resolvers.iter().any(|name| name == resolver) {
        errors.push(format!("missing {resolver} call"));
    }
    errors
}

fn function(source: &str, name: &str) -> Block {
    syn::parse_file(source)
        .unwrap()
        .items
        .into_iter()
        .find_map(|item| match item {
            Item::Fn(function) if function.sig.ident == name => Some(*function.block),
            _ => None,
        })
        .unwrap_or_else(|| panic!("missing {name}"))
}

#[test]
fn lease_root_uses_process_resolver_without_environment_reads() {
    let block = function(
        include_str!("../../synapse-module/src/lib.rs"),
        "synapse_lease_root",
    );
    assert_eq!(
        wiring_errors(&block, "resolve_data_root_from_process"),
        Vec::<String>::new()
    );
}

#[test]
fn model_cache_root_uses_process_resolver_without_environment_reads() {
    let source = syn::parse_file(include_str!("../src/cache.rs")).unwrap();
    let blocks = source
        .items
        .into_iter()
        .filter_map(|item| {
            let Item::Impl(item) = item else { return None };
            let Type::Path(path) = &*item.self_ty else {
                return None;
            };
            if !path.path.is_ident("ModelCache") {
                return None;
            }
            item.items.into_iter().find_map(|item| match item {
                ImplItem::Fn(method) if method.sig.ident == "default_root" => Some(method.block),
                _ => None,
            })
        })
        .collect::<Vec<_>>();
    assert_eq!(blocks.len(), 1, "expected ModelCache::default_root");
    assert_eq!(
        wiring_errors(&blocks[0], "resolve_data_root_from_process"),
        Vec::<String>::new()
    );
}

#[test]
fn store_root_uses_injected_resolver_without_environment_reads() {
    let block = function(
        include_str!("../../synapse-module/src/lib.rs"),
        "default_storage_descriptor_with_environment",
    );
    assert_eq!(
        wiring_errors(&block, "resolve_data_root"),
        Vec::<String>::new()
    );
}

#[test]
fn wiring_guard_rejects_inline_environment_reads_and_missing_resolvers() {
    for block in [
        syn::parse_quote!({
            env::var("HOME");
            resolve_data_root(root, platform, lookup)
        }),
        syn::parse_quote!({
            env::var_os("HOME");
            resolve_data_root(root, platform, lookup)
        }),
        syn::parse_quote!({
            std::env::current_dir();
            resolve_data_root(root, platform, lookup)
        }),
        syn::parse_quote!({ unrelated_resolver(root) }),
    ] {
        assert_eq!(wiring_errors(&block, "resolve_data_root").len(), 1);
    }
}
