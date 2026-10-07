//! Parse launch expressions rather than allowing a nearby helper call to bless
//! an unrelated statement. Local path aliases are followed within each block.

use std::{collections::HashSet, fs, path::Path};
use syn::{
    visit::{self, Visit},
    Expr, ExprCall, ExprMacro, Lit, Pat,
};

#[derive(Default)]
struct Guard {
    raw_paths: HashSet<String>,
    violations: Vec<String>,
}

fn call_name(call: &ExprCall) -> String {
    if let Expr::Path(path) = &*call.func {
        path.path
            .segments
            .iter()
            .map(|part| part.ident.to_string())
            .collect::<Vec<_>>()
            .join("::")
    } else {
        String::new()
    }
}

impl Guard {
    fn raw(&self, expression: &Expr) -> bool {
        struct Raw<'a> {
            guard: &'a Guard,
            found: bool,
        }
        impl<'ast> Visit<'ast> for Raw<'_> {
            fn visit_expr_call(&mut self, call: &'ast ExprCall) {
                let name = call_name(call);
                if name.ends_with("ckdev_binary") || name.ends_with("ckdev_binary_hard_link") {
                    return;
                }
                visit::visit_expr_call(self, call);
            }
            fn visit_expr_path(&mut self, path: &'ast syn::ExprPath) {
                if path
                    .path
                    .get_ident()
                    .is_some_and(|name| self.guard.raw_paths.contains(&name.to_string()))
                {
                    self.found = true;
                }
            }
            fn visit_expr_macro(&mut self, expression: &'ast ExprMacro) {
                // env! and format! carry paths in tokens, not expression nodes.
                let tokens = expression.mac.tokens.to_string();
                if tokens.contains("CARGO_BIN_EXE_ck-") || tokens.contains("ck-synapse") {
                    self.found = true;
                }
            }
            fn visit_lit(&mut self, literal: &'ast Lit) {
                if let Lit::Str(text) = literal {
                    let text = text.value();
                    if text.starts_with("ck-")
                        || text.split(['/', '\\']).any(|part| part.starts_with("ck-"))
                    {
                        self.found = true;
                    }
                }
            }
        }
        let mut raw = Raw {
            guard: self,
            found: false,
        };
        raw.visit_expr(expression);
        raw.found
    }
}

impl<'ast> Visit<'ast> for Guard {
    fn visit_block(&mut self, block: &'ast syn::Block) {
        let previous = self.raw_paths.clone();
        visit::visit_block(self, block);
        self.raw_paths = previous;
    }
    fn visit_local(&mut self, local: &'ast syn::Local) {
        if let (Pat::Ident(name), Some(init)) = (&local.pat, &local.init) {
            if self.raw(&init.expr) {
                self.raw_paths.insert(name.ident.to_string());
            } else {
                self.raw_paths.remove(&name.ident.to_string());
            }
        }
        visit::visit_local(self, local);
    }
    fn visit_expr_assign(&mut self, assign: &'ast syn::ExprAssign) {
        if let Expr::Path(path) = &*assign.left {
            if let Some(name) = path.path.get_ident() {
                if self.raw(&assign.right) {
                    self.raw_paths.insert(name.to_string());
                } else {
                    self.raw_paths.remove(&name.to_string());
                }
            }
        }
        visit::visit_expr_assign(self, assign);
    }
    fn visit_expr_call(&mut self, call: &'ast ExprCall) {
        let name = call_name(call);
        if (name.ends_with("Command::new") || name.ends_with("WorkerHostConfig::new"))
            && call.args.first().is_some_and(|arg| self.raw(arg))
        {
            self.violations.push(format!(
                "{name}: production-named executable bypasses ckdev_binary"
            ));
        }
        visit::visit_expr_call(self, call);
    }
}

fn scan(source: &str) -> Vec<String> {
    let file = syn::parse_file(source).expect("source scan must parse Rust, not silently skip it");
    let mut guard = Guard::default();
    guard.visit_file(&file);
    guard.violations
}

fn rust_tests(path: &Path, failures: &mut Vec<String>, count: &mut usize) {
    for entry in fs::read_dir(path).unwrap() {
        let path = entry.unwrap().path();
        if path.is_dir() {
            rust_tests(&path, failures, count);
        } else if path.extension().is_some_and(|extension| extension == "rs") {
            *count += 1;
            for violation in scan(&fs::read_to_string(&path).unwrap()) {
                failures.push(format!("{}: {violation}", path.display()));
            }
        }
    }
}

fn inline_tests(path: &Path, failures: &mut Vec<String>, count: &mut usize) {
    for entry in fs::read_dir(path).unwrap() {
        let path = entry.unwrap().path();
        if path.is_dir() {
            inline_tests(&path, failures, count);
        } else if path.extension().is_some_and(|extension| extension == "rs") {
            struct Tests {
                guard: Guard,
            }
            impl<'ast> Visit<'ast> for Tests {
                fn visit_item_mod(&mut self, module: &'ast syn::ItemMod) {
                    let test_module = module.attrs.iter().any(|attr| match &attr.meta {
                        syn::Meta::List(list) => {
                            list.path.is_ident("cfg") && list.tokens.to_string() == "test"
                        }
                        _ => false,
                    });
                    if test_module {
                        self.guard.visit_item_mod(module);
                    } else {
                        visit::visit_item_mod(self, module);
                    }
                }
                fn visit_item_fn(&mut self, function: &'ast syn::ItemFn) {
                    if function.attrs.iter().any(|attr| {
                        attr.path()
                            .segments
                            .last()
                            .is_some_and(|segment| segment.ident == "test")
                    }) {
                        self.guard.visit_item_fn(function);
                    }
                }
            }
            let file = syn::parse_file(&fs::read_to_string(&path).unwrap()).unwrap();
            let mut tests = Tests {
                guard: Guard::default(),
            };
            tests.visit_file(&file);
            *count += 1;
            for violation in tests.guard.violations {
                failures.push(format!("{}: {violation}", path.display()));
            }
        }
    }
}

#[test]
fn test_launches_use_development_images() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let mut failures = Vec::new();
    let mut count = 0;
    for entry in fs::read_dir(root.join("crates")).unwrap() {
        let crate_root = entry.unwrap().path();
        let tests = crate_root.join("tests");
        if tests.is_dir() {
            rust_tests(&tests, &mut failures, &mut count);
        }
        let source = crate_root.join("src");
        if source.is_dir() {
            inline_tests(&source, &mut failures, &mut count);
        }
    }
    println!("scanned {count} Rust sources (integration tests and inline test code)");
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn planted_direct_spawn_after_wrapped_spawn_is_rejected() {
    let source = r#"fn test() {
        Command::new(ckdev_binary(env!("CARGO_BIN_EXE_ck-synapse"), &scratch).unwrap()).spawn();
        Command::new(env!("CARGO_BIN_EXE_ck-synapse")).spawn();
    }"#;
    assert_eq!(scan(source).len(), 1);
}

#[test]
fn planted_paths_aliases_and_multiline_wrappers_are_judged_separately() {
    let source = r#"fn test() {
        Command::new("target/release/ck-synapse").output();
        let binary = PathBuf::from("target/debug").join("ck-synapse-worker-cuda.exe");
        Command::new(&binary).spawn();
        WorkerHostConfig::new(binary, &scratch);
        Command::new(ckdev_binary(
            env!("CARGO_BIN_EXE_ck-synapse"),
            &scratch
        ).unwrap()).spawn();
        let binary = ckdev_binary("target/debug/ck-synapse", &scratch).unwrap();
        Command::new(binary).output();
        let evidence = fs::read(env!("CARGO_BIN_EXE_ck-synapse"));
        let documentation = "Command::new(env!(\"CARGO_BIN_EXE_ck-synapse\"))";
    }"#;
    assert_eq!(scan(source).len(), 3);
}
