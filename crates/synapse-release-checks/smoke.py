"""Exercise packaged workers, including refusal exit codes (not loader aborts)."""
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile


def run(binary, arg, code, token, env=None):
    p = subprocess.run([str(binary), arg], capture_output=True, text=True, env=env)
    output = p.stdout + p.stderr
    print(output)
    if p.returncode != code or token not in output:
        raise ValueError(f'{binary.name}: expected {code}/{token}, got {p.returncode}/{output}')
    return output


def main(root):
    windows = sys.platform == 'win32'
    suffix = '.exe' if windows else ''
    if sys.platform == 'darwin':
        if not (root / 'ck-synapse').is_file():
            raise ValueError('ANE worker must resolve beside ck-synapse')
        run(root / 'ck-synapse-worker-ane-direct', '--version', 0, 'ane')
        return
    for feature in ('cuda', 'vulkan'):
        binary = root / ('ck-synapse-worker-' + feature + suffix)
        output = run(binary, '--version', 0, feature)
        if not re.search(r'\bfeatures=' + feature + r'\b', output):
            raise ValueError('worker feature disabled')
        if not re.search(r'\bmanifest_digest=[0-9a-f]{64}\b', output):
            raise ValueError('missing embedded manifest digest')
    cuda = root / ('ck-synapse-worker-cuda' + suffix)
    if not windows:
        libs = os.environ['LD_LIBRARY_PATH'].split(':')
        if any(list(Path(p).glob('libcuda.so*')) for p in libs):
            raise ValueError('driver must not be bundled')
    run(cuda, '--probe-floor', 2, 'cuda_no_driver')
    if windows:
        with tempfile.TemporaryDirectory() as empty:
            alone = Path(empty) / cuda.name
            alone.write_bytes(cuda.read_bytes())
            run(alone, '--probe-floor', 2, 'cuda_runtime_missing:cublasLt64_13.dll')
    vulkan = root / ('ck-synapse-worker-vulkan' + suffix)
    with tempfile.TemporaryDirectory() as empty:
        env = dict(os.environ, VK_DRIVER_FILES=empty)
        run(vulkan, '--probe-floor', 2, 'vulkan_no_device', env)
    if not windows:
        lavapipe = list(Path('/usr/share/vulkan/icd.d').glob('lvp*.json'))
        if len(lavapipe) != 1:
            raise ValueError('expected exactly one Mesa lavapipe ICD')
        run(vulkan, '--probe-floor', 2, 'vulkan_software_device', dict(os.environ, VK_DRIVER_FILES=str(lavapipe[0])))


if __name__ == '__main__':
    main(Path(sys.argv[1]).resolve())
