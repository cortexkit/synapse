#!/usr/bin/env python3
"""Release-candidate guards, shared by hosted builds and fixture self-tests."""

import argparse
from contextlib import contextmanager
import errno
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile


class Refused(ValueError):
    """An input cannot be used to build or certify a release candidate."""


@contextmanager
def ckdev_binary(worker):
    """Smoke extracted images without impersonating an installed process."""
    with tempfile.TemporaryDirectory() as scratch:
        name = worker.name[3:] if worker.name.startswith("ck-") else worker.name
        alias = Path(scratch) / ("ckdev-" + name)
        try:
            os.link(worker, alias)
        except OSError as error:
            if error.errno != errno.EXDEV:
                raise
            shutil.copy2(worker, alias)
        yield alias


def require_source(source, head):
    if not re.fullmatch(r"[0-9a-f]{40}", source) or head != source:
        raise Refused(f"source commit mismatch: requested {source}, HEAD {head}")


def require_readme(text):
    if "libcublasLt.so.12" in text:
        raise Refused("README contains the obsolete CUDA 12 soname")


def require_digest(archive, expected):
    if not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise Refused("archive digest must be a committed lowercase SHA-256")
    digest = hashlib.sha256()
    with archive.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != expected:
        raise Refused(f"digest mismatch: {archive}")


def extract_verified(archive, expected, extractor):
    # Never give an extractor unverified vendor bytes.
    require_digest(archive, expected)
    extractor()


def extract_archive(archive, destination, strip_components):
    destination.mkdir(parents=True, exist_ok=True)
    if archive.suffix == ".zip":
        if strip_components:
            raise Refused("strip-components is only supported for tar archives")
        with zipfile.ZipFile(archive) as zipped:
            zipped.extractall(destination)
    else:
        subprocess.run(
            ["tar", "-xJf", str(archive), "-C", str(destination),
             f"--strip-components={strip_components}"],
            check=True,
        )


def require_inventory(inventory, candidate):
    if not candidate["isDraft"]:
        raise Refused("candidate must remain a draft")
    names = {asset["name"] for asset in candidate["assets"]}
    expected = {entry["asset"] for entry in inventory["assets"]}
    expected |= {name + inventory["sidecar_suffix"] for name in expected if name.endswith(".zip")}
    if not expected <= names:
        raise Refused(f"missing candidate assets: {sorted(expected - names)}")
    print(f"checked {len(expected)} candidate assets and sidecars")


def require_spirv(output, core):
    actual = re.search(r"kernel_revision=([0-9a-f]{64})(?:\s|$)", output)
    expected = re.search(r'VULKAN_KERNEL_REVISION: &str =\s*"([0-9a-f]{64})"', core)
    if not actual or not expected or actual.group(1) != expected.group(1):
        raise Refused("SPIR-V set digest differs from synapse-core")


def require_cuda_elf(dynamic):
    needed = re.findall(r"\(NEEDED\).*?\[(.*?)\]", dynamic)
    cuda = sorted(name for name in needed if re.match(r"lib(?:cuda|cublas)", name))
    if cuda != ["libcublas.so.13", "libcublasLt.so.13", "libcudart.so.13"]:
        raise Refused(f"unexpected CUDA DT_NEEDED entries: {cuda}")


def require_missing_runtime(worker):
    with tempfile.TemporaryDirectory() as isolated:
        for directory in os.environ["LD_LIBRARY_PATH"].split(":"):
            for library in Path(directory).glob("*.so*"):
                if not library.name.startswith("libcublasLt.so"):
                    shutil.copy2(library.resolve(), Path(isolated) / library.name)
        result = subprocess.run(
            [str(worker), "--probe-floor", "--model", "gte-modernbert-base"],
            env={**os.environ, "LD_LIBRARY_PATH": isolated},
            capture_output=True, text=True, timeout=30,
        )
        print(result.stderr)
        if result.returncode == 0 or "libcublasLt.so.13" not in result.stderr:
            raise Refused("missing libcublasLt.so.13 did not produce the expected loader failure")


class GuardSelfTests(unittest.TestCase):
    """Each clean control passes; planted violations must reach their refusal arm."""

    def assert_refused(self, message, function, *args):
        with self.assertRaisesRegex(Refused, message):
            function(*args)

    def test_development_image_alias_preserves_binary_and_exe_suffix(self):
        with tempfile.TemporaryDirectory() as scratch:
            built = Path(scratch) / "ck-synapse-worker-vulkan.exe"
            built.write_bytes(b"built image")
            with ckdev_binary(built) as alias:
                self.assertEqual(alias.name, "ckdev-synapse-worker-vulkan.exe")
                self.assertEqual(alias.read_bytes(), b"built image")
                self.assertTrue(os.path.samefile(built, alias))
            self.assertFalse(alias.exists())
            self.assertEqual(built.read_bytes(), b"built image")

    def test_source_commit_equality(self):
        source = "a" * 40
        require_source(source, source)
        self.assert_refused("source commit mismatch", require_source, source, "b" * 40)
        for invalid in ("main", "A" * 40, "a" * 39):
            self.assert_refused("source commit mismatch", require_source, invalid, invalid)

    def test_obsolete_readme_soname(self):
        require_readme("libcudart.so.13 libcublas.so.13 libcublasLt.so.13")
        self.assert_refused("obsolete CUDA 12", require_readme, "requires libcublasLt.so.12")

    def test_digest_before_extraction(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "vendor.zip"
            with zipfile.ZipFile(archive, "w") as zipped:
                zipped.writestr(zipfile.ZipInfo("runtime.so"), b"verified fixture")
            # ZipInfo fixes the timestamp, making this fixture's bytes reproducible.
            expected = "7394e90a058c3010ab8271e3856855d3748cde0c6601bcdd58a71fdf0960aa92"
            destination = root / "extracted"
            extract_verified(archive, expected, lambda: extract_archive(archive, destination, 0))
            self.assertEqual((destination / "runtime.so").read_bytes(), b"verified fixture")
            (destination / "runtime.so").unlink()
            archive.write_bytes(b"corrupt vendor bytes")
            extracted = []
            self.assert_refused("digest mismatch", extract_verified, archive, expected,
                                lambda: extracted.append(True))
            self.assertEqual(extracted, [], "extractor saw corrupt bytes")
            self.assertFalse((destination / "runtime.so").exists())
            self.assert_refused("lowercase SHA-256", require_digest, archive, "vendor-published-hash")

    def test_inventory_completeness(self):
        inventory = {"assets": [{"asset": "worker.zip"}, {"asset": "release-manifest.json"}],
                     "sidecar_suffix": ".sha256"}
        assets = [{"name": name} for name in ("worker.zip", "worker.zip.sha256", "release-manifest.json")]
        require_inventory(inventory, {"isDraft": True, "assets": assets})
        for index in range(len(assets)):
            self.assert_refused("missing candidate assets", require_inventory, inventory,
                                {"isDraft": True, "assets": assets[:index] + assets[index + 1:]})
        self.assert_refused("remain a draft", require_inventory, inventory,
                            {"isDraft": False, "assets": assets})

    def test_spirv_digest_matches_core(self):
        digest = "a" * 64
        core = f'pub const VULKAN_KERNEL_REVISION: &str =\n "{digest}";'
        require_spirv(f"worker kernel_revision={digest}\n", core)
        self.assert_refused("SPIR-V", require_spirv, "kernel_revision=" + "b" * 64, core)
        self.assert_refused("SPIR-V", require_spirv, "missing revision", core)

    def test_cuda_dt_needed(self):
        names = ["libcudart.so.13", "libcublas.so.13", "libcublasLt.so.13"]
        def dynamic(libraries):
            return "\n".join(f"(NEEDED) Shared library: [{name}]" for name in libraries)
        require_cuda_elf(dynamic(names + ["libc.so.6"]))
        violations = [names[:-1], names + ["libcuda.so.1"], names + [names[0]],
                      names[:-1] + ["libcublasLt.so.12"]]
        for libraries in violations:
            self.assert_refused("DT_NEEDED", require_cuda_elf, dynamic(libraries))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    commands = parser.add_subparsers(dest="command")
    source = commands.add_parser("source")
    source.add_argument("--commit", required=True)
    source.add_argument("--readme", type=Path, default=Path("README.md"))
    for command in ("digest", "extract"):
        archive = commands.add_parser(command)
        archive.add_argument("--archive", type=Path, required=True)
        archive.add_argument("--sha256", required=True)
        if command == "extract":
            archive.add_argument("--destination", type=Path, required=True)
            archive.add_argument("--strip-components", type=int, default=0)
    inventory = commands.add_parser("inventory")
    inventory.add_argument("--inventory", type=Path, default=Path("bench/parity/release-assets.json"))
    inventory.add_argument("--candidate", type=Path, required=True)
    spirv = commands.add_parser("spirv")
    spirv.add_argument("--worker", type=Path, required=True)
    spirv.add_argument("--core", type=Path, default=Path("crates/synapse-core/src/worker_engine_names.rs"))
    elf = commands.add_parser("cuda-elf")
    elf.add_argument("--worker", type=Path, required=True)
    args = parser.parse_args()
    if args.self_test:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(GuardSelfTests)
        return 0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1
    try:
        if args.command == "source":
            head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
            require_source(args.commit, head)
            require_readme(args.readme.read_text(encoding="utf-8"))
        elif args.command == "digest":
            require_digest(args.archive, args.sha256)
        elif args.command == "extract":
            extract_verified(args.archive, args.sha256,
                             lambda: extract_archive(args.archive, args.destination, args.strip_components))
        elif args.command == "inventory":
            require_inventory(json.loads(args.inventory.read_text(encoding="utf-8")), json.loads(args.candidate.read_text(encoding="utf-8")))
        elif args.command == "spirv":
            with ckdev_binary(args.worker) as alias:
                output = subprocess.check_output([str(alias), "--version"], text=True)
            print(output)
            require_spirv(output, args.core.read_text(encoding="utf-8"))
        elif args.command == "cuda-elf":
            output = subprocess.check_output(["readelf", "-d", str(args.worker)], text=True)
            print(output)
            require_cuda_elf(output)
            with ckdev_binary(args.worker) as alias:
                require_missing_runtime(alias)
        else:
            parser.error("a command or --self-test is required")
    except (Refused, OSError, subprocess.SubprocessError, ValueError, KeyError) as error:
        print(f"release-candidate refused: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
