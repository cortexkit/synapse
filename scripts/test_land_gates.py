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

    def test_only_apple_test_commands_stay_on_the_mac(self):
        portable = HELPERS["static_gates"]() + HELPERS["full_gates"]() + HELPERS["contract_tests"]()
        apple = [HELPERS["cargo_test"](package, "--lib") for package in (
            "synapse-engine-owned", "synapse-worker-ane", "synapse-worker-ane-direct",
            "synapse-worker-decode", "synapse-worker-llama")]
        remote, native = HELPERS["split_commands"]([tuple(c) for c in portable + apple], "Darwin")
        self.assertEqual(remote, [tuple(c) for c in portable])
        module_native = HELPERS["module_macos_tests"](tuple(HELPERS["full_gates"]()[4]))
        self.assertEqual(native, [tuple(c) for c in module_native + apple])
        remote, native = HELPERS["split_commands"](portable + apple, "Linux")
        self.assertEqual(remote, portable + apple)
        self.assertEqual(native, [])

    def test_module_native_filter_keeps_all_ten_active_mac_cases_only(self):
        command = tuple(HELPERS["full_gates"]()[4])
        native = HELPERS["module_macos_tests"](command)
        self.assertEqual(len(native), 2)
        self.assertEqual(native[0], HELPERS["module_fixtures"]()[1])
        self.assertEqual(native[-1][:3], ["cargo", "nextest", "run"])
        expression = native[-1][-1]
        self.assertEqual(expression.count("test("), 10)
        for case in ("direct_ane_models_reuse_one_runtime_supervisor",
                     "kv_runtime_matrix_proves_warm_reuse_reclamation_and_lifecycle_faults",
                     "ane_probe_placement_ping_runs_outside_the_module_runtime",
                     "owned_gte_inline_embed_batch_throughput_sweep"):
            self.assertIn(f"test({case})", expression)
        self.assertNotIn("--run-ignored", native[-1])

    def test_narrow_module_native_filter_does_not_broaden_target_selection(self):
        for target, count in (("it", 1), ("skeleton_e2e", 6), ("wire_contract", 0)):
            command = HELPERS["cargo_test"]("synapse-module", "--features", "test-support", "--test", target)
            native = HELPERS["module_macos_tests"](command)
            if count:
                self.assertEqual(native[-1][-1].count("test("), count)
                self.assertNotIn("--lib", native[-1])
            else:
                self.assertEqual(native, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
