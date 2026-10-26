"""Exercise landing selection without substituting fakes for the real gates."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import platform
import runpy
import subprocess
import sys
import unittest
from unittest.mock import patch


HELPERS = runpy.run_path(str(Path(__file__).with_name("land-gates.py")))
ROOT = Path("/checkout")


def package(name, directory, targets):
    return {"name": name, "manifest_path": str(ROOT / directory / "Cargo.toml"), "targets": [
        {"name": target, "kind": [kind], "src_path": str(ROOT / directory / path),
         "required-features": list(required)}
        for target, kind, path, *required in targets
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
                ("skeleton_e2e", "test", "tests/skeleton_e2e.rs", "e2e"),
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

    def test_feature_gated_target_gets_its_required_features(self):
        self.assertEqual(self.select("crates/module/tests/skeleton_e2e.rs"), [
            HELPERS["cargo_test"]("synapse-module", "--features", "test-support,e2e", "--test", "skeleton_e2e")])

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
        self.assertEqual(gates[0][:5], ["cargo", "build", "--locked", "--no-default-features", "-p"])
        self.assertIn("synapse-probe-stub", gates[1])
        self.assertIn("--test", gates[3])
        self.assertIn("skeleton_e2e", gates[3])
        self.assertIn("e2e", gates[3][gates[3].index("--features") + 1].split(","))
        self.assertEqual(gates[-1][1], "clippy")

    def plan(self, *flags):
        replies = {"merge-base": "base\n", "diff": "", "ls-files": "", "metadata": '{"packages": []}'}

        def check_output(command, **_):
            return next(reply for word, reply in replies.items() if word in command)

        output = io.StringIO()
        with patch.object(sys, "argv", ["land-gates.py", "--plan", *flags]), \
                patch.object(subprocess, "check_output", side_effect=check_output), \
                patch.object(platform, "system", return_value="Linux"), redirect_stdout(output):
            self.assertEqual(HELPERS["main"](), 0)
        return [list(command) for command in json.loads(output.getvalue())]

    def test_landing_plan_adds_module_builds_tests_and_clippy_only_with_full(self):
        fast, full = self.plan(), self.plan("--full")
        self.assertEqual(fast[0], HELPERS["static_gates"]()[0])
        for gate in HELPERS["full_gates"]():
            self.assertIn(gate, full)
        self.assertNotIn(HELPERS["full_gates"]()[-1], fast)

    def test_only_apple_test_commands_stay_on_the_mac(self):
        portable = HELPERS["static_gates"]() + HELPERS["full_gates"]() + HELPERS["contract_tests"]()
        apple = [HELPERS["cargo_test"](package, "--lib") for package in (
            "synapse-engine-owned", "synapse-worker-ane", "synapse-worker-ane-direct",
            "synapse-worker-decode", "synapse-worker-llama")]
        remote, native = HELPERS["split_commands"]([tuple(c) for c in portable + apple], "Darwin")
        self.assertEqual(remote, [tuple(c) for c in portable])
        module_native = HELPERS["module_macos_tests"](tuple(HELPERS["full_gates"]()[3]))
        self.assertEqual(native, [tuple(c) for c in module_native + apple])
        remote, native = HELPERS["split_commands"](portable + apple, "Linux")
        self.assertEqual(remote, portable + apple)
        self.assertEqual(native, [])

    def test_module_native_filter_keeps_active_mac_cases_only(self):
        command = tuple(HELPERS["full_gates"]()[3])
        native = HELPERS["module_macos_tests"](command)
        self.assertEqual(native[0], HELPERS["module_fixtures"]()[1])
        run = native[-1]
        self.assertEqual(run[:3], ["cargo", "nextest", "run"])
        self.assertIn("e2e", run[run.index("--features") + 1].split(","))
        expression = run[-1]
        for case in ("direct_ane_models_reuse_one_runtime_supervisor",
                     "kv_runtime_matrix_proves_warm_reuse_reclamation_and_lifecycle_faults",
                     "ane_probe_placement_ping_runs_outside_the_module_runtime"):
            self.assertIn(f"test({case})", expression)
        self.assertNotIn("--run-ignored", run)

    def test_module_native_names_are_checked_against_the_listed_targets(self):
        command = tuple(HELPERS["full_gates"]()[3])
        check, run = HELPERS["module_macos_tests"](command)[-2:]
        self.assertEqual(check[:3], ["python3", "scripts/land-gates.py", "--require-tests"])
        names = json.loads(check[3])
        self.assertEqual(" | ".join(f"test({name})" for name in names), run[-1])
        listing = check[check.index("--") + 1:]
        self.assertEqual(listing[:3], ["cargo", "nextest", "list"])
        # The listing must build exactly what the run builds.
        selection = run[run.index("-p"):run.index("--retries")]
        self.assertEqual(listing[listing.index("-p"):listing.index("--message-format")], selection)

    def test_a_name_that_matches_no_listed_test_is_refused(self):
        listing = {"rust-suites": {"synapse-module::it": {"testcases": {
            "kv::kv_runtime_matrix_proves_warm_reuse": {}}}}}
        unmatched = HELPERS["unmatched_tests"]
        self.assertEqual(unmatched(listing, ["kv_runtime_matrix"]), [])
        self.assertEqual(unmatched(listing, ["kv_runtime_matrix", "renamed_away"]), ["renamed_away"])
        self.assertEqual(unmatched({"rust-suites": {}}, ["anything"]), ["anything"])

    def test_require_tests_fails_when_listing_lacks_a_name(self):
        listing = json.dumps({"rust-suites": {"bin": {"testcases": {"mod::present": {}}}}})
        with patch.object(subprocess, "check_output", return_value=listing) as list_tests, \
                redirect_stdout(io.StringIO()), patch.object(sys, "stderr", io.StringIO()):
            self.assertEqual(HELPERS["require_tests"](["present"], ["cargo", "nextest", "list"]), 0)
            self.assertEqual(HELPERS["require_tests"](["present", "gone"], ["cargo", "nextest", "list"]), 1)
        self.assertEqual(list_tests.call_args.args[0], ["cargo", "nextest", "list"])

    def test_narrow_module_native_filter_does_not_broaden_target_selection(self):
        lib_case = "test(direct_ane_models_reuse_one_runtime_supervisor)"
        it_case = "test(kv_runtime_matrix_proves_warm_reuse_reclamation_and_lifecycle_faults)"
        e2e_case = "test(ane_probe_placement_ping_runs_outside_the_module_runtime)"
        for target, present, absent in (("it", [it_case], [lib_case, e2e_case]),
                                        ("skeleton_e2e", [e2e_case], [lib_case, it_case])):
            command = HELPERS["cargo_test"]("synapse-module", "--features", "test-support", "--test", target)
            run = HELPERS["module_macos_tests"](command)[-1]
            self.assertNotIn("--lib", run)
            for case in present:
                self.assertIn(case, run[-1])
            for case in absent:
                self.assertNotIn(case, run[-1])
        command = HELPERS["cargo_test"]("synapse-module", "--features", "test-support", "--test", "wire_contract")
        self.assertEqual(HELPERS["module_macos_tests"](command), [])

if __name__ == "__main__":
    unittest.main(verbosity=2)
