use syn::{punctuated::Punctuated, visit::Visit, Attribute, Expr, Item, Meta, Token};

const SOURCE: &str = include_str!("../src/machine_profile.rs");
const FORBIDDEN: &[&str] = &[
    "command_stdout",
    "command_stdout_within",
    "Command",
    "Prober",
];

fn cfg_matches(meta: &Meta, os: &str, testing: bool) -> bool {
    match meta {
        Meta::Path(path) if path.is_ident("test") => testing,
        Meta::Path(path) if path.is_ident("unix") => os != "windows",
        Meta::Path(path) if path.is_ident("windows") => os == "windows",
        Meta::NameValue(value) if value.path.is_ident("target_os") => {
            let Expr::Lit(literal) = &value.value else {
                panic!("non-literal target_os")
            };
            let syn::Lit::Str(value) = &literal.lit else {
                panic!("non-string target_os")
            };
            value.value() == os
        }
        Meta::List(list) => {
            let arguments = list
                .parse_args_with(Punctuated::<Meta, Token![,]>::parse_terminated)
                .unwrap();
            if list.path.is_ident("any") {
                arguments.iter().any(|meta| cfg_matches(meta, os, testing))
            } else if list.path.is_ident("all") {
                arguments.iter().all(|meta| cfg_matches(meta, os, testing))
            } else if list.path.is_ident("not") {
                assert_eq!(arguments.len(), 1);
                !cfg_matches(&arguments[0], os, testing)
            } else {
                panic!("unsupported cfg list")
            }
        }
        // Other feature and target predicates are conservatively admitted.
        _ => true,
    }
}

fn active(attributes: &[Attribute], os: &str, testing: bool) -> bool {
    attributes
        .iter()
        .filter(|attribute| attribute.path().is_ident("cfg"))
        .all(|attribute| cfg_matches(&attribute.parse_args::<Meta>().unwrap(), os, testing))
}

fn attributes(item: &Item) -> &[Attribute] {
    match item {
        Item::Const(item) => &item.attrs,
        Item::Enum(item) => &item.attrs,
        Item::ExternCrate(item) => &item.attrs,
        Item::Fn(item) => &item.attrs,
        Item::ForeignMod(item) => &item.attrs,
        Item::Impl(item) => &item.attrs,
        Item::Macro(item) => &item.attrs,
        Item::Mod(item) => &item.attrs,
        Item::Static(item) => &item.attrs,
        Item::Struct(item) => &item.attrs,
        Item::Trait(item) => &item.attrs,
        Item::TraitAlias(item) => &item.attrs,
        Item::Type(item) => &item.attrs,
        Item::Union(item) => &item.attrs,
        Item::Use(item) => &item.attrs,
        _ => &[],
    }
}

struct NativeItems<'a> {
    os: &'a str,
    forbidden: Vec<String>,
}

impl<'ast> Visit<'ast> for NativeItems<'_> {
    fn visit_item(&mut self, item: &'ast Item) {
        if active(attributes(item), self.os, false) {
            syn::visit::visit_item(self, item);
        }
    }

    fn visit_impl_item_fn(&mut self, method: &'ast syn::ImplItemFn) {
        if active(&method.attrs, self.os, false) {
            syn::visit::visit_impl_item_fn(self, method);
        }
    }

    fn visit_expr_block(&mut self, block: &'ast syn::ExprBlock) {
        if active(&block.attrs, self.os, false) {
            syn::visit::visit_expr_block(self, block);
        }
    }

    fn visit_ident(&mut self, ident: &'ast syn::Ident) {
        let name = ident.to_string();
        if FORBIDDEN.contains(&name.as_str()) {
            self.forbidden.push(name);
        }
    }

    fn visit_macro(&mut self, mac: &'ast syn::Macro) {
        use syn::parse::Parser;
        // Macro arguments are token streams, not visited expressions by default.
        // Parse expressions or statements so a command hidden in one is visible.
        if let Ok(arguments) =
            Punctuated::<Expr, Token![,]>::parse_terminated.parse2(mac.tokens.clone())
        {
            for argument in &arguments {
                self.visit_expr(argument);
            }
        } else if let Ok(block) = syn::parse_str::<syn::Block>(&format!("{{{}}}", mac.tokens)) {
            self.visit_block(&block);
        }
        syn::visit::visit_macro(self, mac);
    }
}

fn forbidden_native_items(source: &str, os: &str) -> Vec<String> {
    let file = syn::parse_file(source).expect("valid machine-profile source");
    let mut visitor = NativeItems {
        os,
        forbidden: Vec::new(),
    };
    // Scanning all production items also covers function-pointer source seams
    // and future helper functions, not only direct calls from the collector.
    visitor.visit_file(&file);
    visitor.forbidden
}

#[test]
fn non_macos_profile_collection_has_no_command_path() {
    for os in ["linux", "windows"] {
        assert_eq!(
            forbidden_native_items(SOURCE, os),
            Vec::<String>::new(),
            "{os} collection must stay native"
        );
    }
}

#[test]
fn structural_guard_rejects_commands_in_native_items_and_blocks() {
    for os in ["linux", "windows"] {
        for name in FORBIDDEN {
            for source in [
                format!("fn collect() {{ {name}(); }}"),
                format!("#[cfg(not(target_os = \"macos\"))] fn collect() {{ {name}(); }}"),
                format!("fn collect() {{ #[cfg(target_os = \"{os}\")] {{ let _ = {name}; }} }}"),
                format!("impl Collector {{ fn collect(&self) {{ wrapper!({name}()); }} }}"),
            ] {
                assert_eq!(forbidden_native_items(&source, os), [name.to_string()]);
            }
        }
        assert!(forbidden_native_items(
            r#"
            #[cfg(target_os = "macos")] fn os_build(probe: Prober) { command_stdout(); }
            fn collect() { #[cfg(target_os = "macos")] { Command::new("sysctl"); } }
            #[cfg(any(target_os = "macos", test))] fn command_stdout() { Command::new("sysctl"); }
        "#,
            os
        )
        .is_empty());
    }
}

#[test]
fn command_runner_and_deadline_tests_remain_enabled_on_both_runners() {
    let file = syn::parse_file(SOURCE).unwrap();
    for os in ["linux", "windows"] {
        for name in ["command_stdout", "command_stdout_within"] {
            let function = file
                .items
                .iter()
                .find_map(|item| match item {
                    Item::Fn(function) if function.sig.ident == name => Some(function),
                    _ => None,
                })
                .unwrap();
            assert!(!active(&function.attrs, os, false));
            assert!(active(&function.attrs, os, true));
        }
        let tests = file
            .items
            .iter()
            .find_map(|item| match item {
                Item::Mod(module) if module.ident == "tests" => module.content.as_ref(),
                _ => None,
            })
            .unwrap();
        for name in [
            "a_hanging_probe_refuses_within_its_budget",
            "a_missing_or_failing_probe_refuses_and_names_the_program",
        ] {
            let function = tests
                .1
                .iter()
                .find_map(|item| match item {
                    Item::Fn(function) if function.sig.ident == name => Some(function),
                    _ => None,
                })
                .unwrap();
            assert!(active(&function.attrs, os, true), "{name} must run on {os}");
            assert!(!function
                .attrs
                .iter()
                .any(|attribute| attribute.path().is_ident("ignore")));
        }
    }
}
