"""Local-only output boundaries and shared-Mac resource admission/watchdog."""

import ctypes
import json
import os
import re
import signal
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
TARGET = ROOT / "target" / "stt-bias"


def private_path(path):
    path = Path(path).resolve()
    if not path.is_relative_to(TARGET.resolve()):
        raise ValueError("per-clip inputs/outputs must stay under target/stt-bias")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def footprint():
    """Read current and lifetime-peak phys_footprint, not the RSS counter.

    macOS rusage_info_v4 has a 16-byte UUID followed by uint64 counters;
    phys_footprint and lifetime_max_phys_footprint are at offsets 72 and 240.
    The kernel high-water mark includes peaks between watchdog samples.
    """
    libproc = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
    function = libproc.proc_pid_rusage
    function.argtypes = (ctypes.c_int, ctypes.c_int, ctypes.c_void_p)
    function.restype = ctypes.c_int
    buffer = ctypes.create_string_buffer(512)
    if function(os.getpid(), 4, buffer) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    current = struct.unpack_from("=Q", buffer, 72)[0]
    peak = struct.unpack_from("=Q", buffer, 240)[0]
    if not current or peak < current:
        raise RuntimeError("invalid macOS footprint counters")
    return {"phys_footprint_bytes": current, "peak_phys_footprint_bytes": peak}


def sample():
    import psutil
    swap = subprocess.check_output(["sysctl", "vm.swapusage"], text=True)
    match = re.search(r"used = ([0-9.]+)([MGT])", swap)
    if not match:
        raise RuntimeError("cannot read swap usage")
    swap_bytes = float(match[1]) * {"M": 2**20, "G": 2**30, "T": 2**40}[match[2]]
    vm = subprocess.check_output(["vm_stat"], text=True)
    pageout_match = re.search(r"Pageouts:\s+(\d+)", vm)
    pressure = subprocess.check_output(["memory_pressure", "-Q"], text=True)
    free_match = re.search(r"free percentage:\s+(\d+)%", pressure)
    if pageout_match is None or free_match is None:
        raise RuntimeError("cannot read pageouts or memory pressure")
    pageouts, free = int(pageout_match[1]), int(free_match[1])
    process = psutil.Process()
    rss = process.memory_info().rss
    for child in process.children(recursive=True):
        try:
            rss += child.memory_info().rss
        except psutil.NoSuchProcess:
            pass
    return {"time": time.time(), "swap_bytes": swap_bytes, "pageouts": pageouts,
            "free_percent": free, "rss_bytes": rss,
            "load": list(os.getloadavg()), **footprint()}


def limits_exceeded(first, recent):
    value = recent[-1]
    return (value["rss_bytes"] > 20 * 2**30 or
            value["swap_bytes"] - first["swap_bytes"] > 2**30 or
            value["free_percent"] < 25 or
            value["pageouts"] - recent[0]["pageouts"] > 25000)


class Watchdog:
    def __init__(self, output):
        if sys.platform != "darwin":
            raise RuntimeError("private-audio inference is admitted only on this Mac")
        self.path = private_path(output)
        self.first = sample()
        if self.first["free_percent"] < 25:
            raise RuntimeError("memory_pressure free percentage below 25%")
        self.stopped = threading.Event()
        self.history = []
        self.thread = threading.Thread(target=self.watch, daemon=True)

    def record(self, value):
        self.history.append(value)
        with self.path.open("a") as output:
            output.write(json.dumps(value) + "\n")

    def start(self):
        self.record(self.first)
        self.thread.start()

    def watch(self):
        while not self.stopped.wait(30):
            try:
                value = sample()
                self.record(value)
                recent = self.history[-5:]
                if limits_exceeded(self.first, recent):
                    print("Resource watchdog stopped inference; inspect target telemetry", file=sys.stderr)
                    self.terminate()
                    return
            except Exception:
                self.terminate()
                return

    @staticmethod
    def terminate():
        import psutil
        for child in psutil.Process().children(recursive=True):
            try:
                child.terminate()
            except psutil.NoSuchProcess:
                pass
        os.kill(os.getpid(), signal.SIGTERM)

    def close(self):
        self.stopped.set()
        self.thread.join()
        self.record(sample())
