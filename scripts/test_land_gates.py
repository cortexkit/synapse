"""Exercise landing selection without substituting fakes for the real gates."""

from pathlib import Path
import runpy
import unittest


HELPERS = runpy.run_path(str(Path(__file__).with_name("land-gates.py")))
ROOT = Path("/checkout")


def package(name, directory, targets):
    return {"name": name, "manifest_path": str(ROOT / directory / "Cargo.toml"), "targets": [
        {"name": target, "kind": [kind], "src_path": str(ROOT / directory / path)}
        for target, kind, path in targets
    ]}


class LandingSelectionTests(unittest.TestCase):
    def setUp(self):
        self.metadata = {"packages": [
            package("outer", "crates/outer", [("outer", "lib", "src/lib.rs"), ("app", "bin", "src/main.rs")]),
            package("inner", "crates/outer/inner", [("inner", "lib", "src/lib.rs")]),
            package("synapse-module", "crates/module", [
                ("module", "lib", "src/lib.rs"),
                ("it", "test", "tests/it/main.rs"),
                ("wire_contract", "test", "tests/wire_contract.rs"),
            ]),
        ]}

    def select(self, *paths):
        return HELPERS["narrow_tests"](self.metadata, paths, ROOT)

    def test_docs_select_no_additional_rust_targets(self):
        self.assertEqual(self.select("docs/development.md"), [])

    def test_source_selects_unit_targets_once(self):
        self.assertEqual(self.select("crates/outer/src/lib.rs", "crates/outer/src/other.rs"), [
            HELPERS["cargo_test"]("outer", "--bin", "app"),
            HELPERS["cargo_test"]("outer", "--lib"),
        ])

    def test_nested_package_does_not_select_parent(self):
        self.assertEqual(self.select("crates/outer/inner/src/lib.rs"), [HELPERS["cargo_test"]("inner", "--lib")])

    def test_nested_integration_source_selects_its_binary(self):
        self.assertEqual(self.select("crates/module/tests/it/worker_liveness.rs"), [
            HELPERS["cargo_test"]("synapse-module", "--features", "test-support", "--test", "it")])

    def test_single_file_integration_source_selects_its_binary(self):
        self.assertEqual(self.select("crates/module/tests/wire_contract.rs"), [
            HELPERS["cargo_test"]("synapse-module", "--features", "test-support", "--test", "wire_contract")])

    def test_shared_helper_conservatively_selects_integration_tests(self):
        self.assertEqual(self.select("crates/module/tests/common/mod.rs"), [
            HELPERS["cargo_test"]("synapse-module", "--features", "test-support", "--tests")])

    def test_fast_contract_set_keeps_cross_seat_boundaries(self):
        tests = HELPERS["contract_tests"]()
        self.assertIn(HELPERS["cargo_test"]("synapse-core", "--test", "dev_binary_guard", "--test", "child_process"), tests)
        module = tests[1]
        self.assertEqual(module[:3], ["cargo", "nextest", "run"])
        self.assertEqual(module[module.index("-p") + 1], "synapse-module")
        expression = module[module.index("-E") + 1]
        for boundary in ("binary(wire_contract)", "test(launch_nonce_readers::)", "test(store::)", "test(catalog_migration::)"):
            self.assertIn(boundary, expression)

    def test_full_builds_fixtures_before_module_tests(self):
        gates = HELPERS["full_gates"]()
        self.assertEqual(gates[0], ["ckdev-mutate", "check"])
        self.assertEqual(gates[1][:5], ["cargo", "build", "--locked", "--no-default-features", "-p"])
        self.assertIn("synapse-probe-stub", gates[2])
        self.assertIn("--test", gates[4])
        self.assertIn("skeleton_e2e", gates[4])
        self.assertEqual(gates[-1][1], "clippy")


if __name__ == "__main__":
    unittest.main(verbosity=2)
