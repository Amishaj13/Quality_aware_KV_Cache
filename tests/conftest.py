"""Shared fixtures.

By default the model tests use a tiny random-weight Qwen2-architecture model built
offline (utils/tiny_model.py), so the suite runs on CPU without downloads. To test
the real model on SageMaker:

    KVR_TEST_MODEL=Qwen/Qwen2.5-Coder-7B-Instruct pytest tests/ -v
"""
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))


@pytest.fixture(scope="session")
def model_name(tmp_path_factory):
    name = os.environ.get("KVR_TEST_MODEL")
    if name:
        return name
    from utils.tiny_model import build_tiny_model
    return build_tiny_model(tmp_path_factory.mktemp("tiny_model"))


@pytest.fixture(scope="session")
def device_map():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="session")
def cfg(model_name, device_map):
    from utils.config import load_config
    return load_config(overrides={"model": {"name": model_name, "device_map": device_map},
                                  "experiment": {"max_new_tokens": 32, "warmup": False}})


@pytest.fixture(scope="session")
def loaded(cfg):
    from models.model_loader import load_model
    return load_model(cfg)


@pytest.fixture(scope="session")
def samples():
    from utils.dataset import load_samples
    valid, invalid = load_samples("datasets/sample_tasks.json")
    assert not invalid, invalid
    return valid
