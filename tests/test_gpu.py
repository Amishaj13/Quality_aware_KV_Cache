"""Test 1 (environment) and Test 5 (memory recording)."""
import platform

import pytest
import torch

from evaluation import memory_metrics as mm


def test_python_works():
    assert int(platform.python_version_tuple()[0]) == 3


def test_torch_imports():
    assert torch.__version__
    assert torch.tensor([1, 2]).sum().item() == 3


def test_gpu_detected_if_available():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA GPU on this machine (expected on CPU-only hosts)")
    assert torch.cuda.device_count() >= 1
    free, total = torch.cuda.mem_get_info(0)
    assert 0 < free <= total
    x = torch.ones(1024, device="cuda")
    assert x.sum().item() == 1024


def test_memory_snapshot_marks_unavailable_on_cpu():
    snap = mm.snapshot()
    if torch.cuda.is_available():
        assert snap["total_bytes"] > 0 and snap["peak_allocated_bytes"] is not None
    else:
        # unavailable measurements are None, never 0 or an estimate
        assert all(v is None for v in snap.values())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_peak_memory_recorded_on_gpu():
    mm.reset_peak()
    before = mm.snapshot()["allocated_bytes"]
    x = torch.empty(64 * 1024**2, dtype=torch.uint8, device="cuda")  # 64 MiB
    snap = mm.snapshot()
    assert snap["peak_allocated_bytes"] - before >= 64 * 1024**2
    assert snap["peak_reserved_bytes"] >= snap["peak_allocated_bytes"]
    del x


def test_cache_bytes_counts_tensors():
    ts = [torch.zeros(2, 3, dtype=torch.float16) for _ in range(4)]
    assert mm.cache_bytes(((ts[0], ts[1]), (ts[2], ts[3]))) == 4 * 12
    assert mm.cache_bytes((ts[0], ts[0])) == 12  # a shared tensor is counted once
    assert mm.cache_bytes(None) is None
    assert mm.cache_bytes(object()) is None
