"""Inspect the SageMaker environment and save results/server_report.json.

Read-only: installs nothing, creates no AWS resources, never prints credentials.
  python scripts/server_report.py [--model Qwen/Qwen2.5-Coder-7B-Instruct] [--out results/server_report.json]
"""
import argparse
import datetime as dt
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import urllib.request

import _bootstrap  # noqa: F401
from utils.paths import PROJECT_ROOT, resolve

PACKAGES = ["torch", "transformers", "accelerate", "safetensors", "tokenizers", "huggingface_hub", "numpy",
            "pandas", "matplotlib", "scipy", "psutil", "tqdm", "pytest", "networkx", "pyyaml", "bitsandbytes",
            "flash_attn", "boto3", "sagemaker"]


def sh(cmd, timeout=20):
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return (r.stdout or r.stderr).strip()
    except Exception as e:
        return f"unavailable ({type(e).__name__})"


def sagemaker_environment():
    info = {"type": "unknown", "evidence": []}
    meta_path = "/opt/ml/metadata/resource-metadata.json"
    meta = {}
    if os.path.exists(meta_path):
        try:
            meta = json.load(open(meta_path))
        except Exception:
            pass
        info["evidence"].append(meta_path)
    keep = ("AppType", "DomainId", "SpaceName", "ResourceName", "AppImageVersion", "InstanceType")
    info["metadata"] = {k: meta[k] for k in keep if k in meta}
    home = os.path.expanduser("~")
    if os.path.isdir("/home/studio-lab-user") or "studio-lab" in home:
        info["type"] = "SageMaker Studio Lab"
    elif meta.get("AppType") or meta.get("DomainId") or os.environ.get("SAGEMAKER_SPACE_NAME"):
        app = meta.get("AppType", "")
        info["type"] = f"SageMaker Studio ({app or 'app type unknown'})"
    elif os.path.isdir("/home/ec2-user/SageMaker"):
        info["type"] = "SageMaker Notebook Instance"
    elif meta:
        info["type"] = "SageMaker (metadata present)"
    else:
        info["type"] = "not detected as SageMaker (local machine or other cloud)"
    return info


def aws_region():
    for k in ("AWS_REGION", "AWS_DEFAULT_REGION", "REGION_NAME"):
        if os.environ.get(k):
            return os.environ[k], f"env {k}"
    try:
        import boto3
        r = boto3.session.Session().region_name
        if r:
            return r, "boto3 session"
    except Exception:
        pass
    return None, "unavailable"


def instance_type(env_meta):
    if env_meta.get("metadata", {}).get("InstanceType"):
        return env_meta["metadata"]["InstanceType"], "resource-metadata.json"
    if os.environ.get("SAGEMAKER_INSTANCE_TYPE"):
        return os.environ["SAGEMAKER_INSTANCE_TYPE"], "env"
    # EC2 instance metadata (IMDSv2); often blocked inside Studio, so short timeout
    try:
        req = urllib.request.Request("http://169.254.169.254/latest/api/token", method="PUT",
                                     headers={"X-aws-ec2-metadata-token-ttl-seconds": "60"})
        token = urllib.request.urlopen(req, timeout=1).read().decode()
        req = urllib.request.Request("http://169.254.169.254/latest/meta-data/instance-type",
                                     headers={"X-aws-ec2-metadata-token": token})
        return urllib.request.urlopen(req, timeout=1).read().decode(), "EC2 instance metadata"
    except Exception:
        return None, "unavailable (check the Studio space / notebook settings page)"


def package_versions():
    from importlib import metadata
    out = {}
    for p in PACKAGES:
        try:
            out[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            out[p] = None
    return out


def gpu_info():
    out = {"cuda_available": False, "devices": []}
    try:
        import torch
    except ImportError:
        out["error"] = "torch not installed"
        return out
    out["torch"] = torch.__version__
    out["torch_cuda_runtime"] = torch.version.cuda
    out["cudnn"] = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
    out["cuda_available"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        out["bf16_supported"] = torch.cuda.is_bf16_supported()
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            free, total = torch.cuda.mem_get_info(i)
            out["devices"].append({"index": i, "name": p.name, "compute_capability": f"{p.major}.{p.minor}",
                                   "total_vram_gib": round(total / 1024**3, 2), "free_vram_gib": round(free / 1024**3, 2),
                                   "multiprocessors": p.multi_processor_count})
    return out


def nvidia_smi():
    if not shutil.which("nvidia-smi"):
        return {"available": False}
    q = sh("nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used,memory.free,compute_cap "
           "--format=csv,noheader")
    cuda = sh("nvidia-smi | grep -o 'CUDA Version: [0-9.]*'")
    return {"available": True, "query": q, "driver_cuda_version": cuda.replace("CUDA Version: ", ""),
            "full": sh("nvidia-smi")}


def memory_and_disk():
    out = {}
    try:
        import psutil
        vm = psutil.virtual_memory()
        out["ram_total_gib"] = round(vm.total / 1024**3, 2)
        out["ram_available_gib"] = round(vm.available / 1024**3, 2)
        out["cpu_count"] = psutil.cpu_count()
    except ImportError:
        out["ram"] = sh("free -g")
        out["cpu_count"] = os.cpu_count()
    disks = {}
    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    for label, path in (("project", str(PROJECT_ROOT)), ("home", os.path.expanduser("~")), ("hf_cache", hf_home),
                        ("tmp", "/tmp")):
        p = path
        while not os.path.exists(p) and p != "/":
            p = os.path.dirname(p)
        u = shutil.disk_usage(p)
        disks[label] = {"path": path, "total_gib": round(u.total / 1024**3, 1), "free_gib": round(u.free / 1024**3, 1)}
    out["disk"] = disks
    out["df_h"] = sh("df -h")
    return out


def network_checks(model):
    out = {}
    try:
        socket.create_connection(("huggingface.co", 443), timeout=5).close()
        out["internet"] = True
    except Exception as e:
        out["internet"] = False
        out["internet_error"] = type(e).__name__
    out["hf_token_present"] = bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"))
    if out["internet"]:
        try:
            from huggingface_hub import HfApi
            api = HfApi(token=os.environ.get("HF_TOKEN") or None)
            mi = api.model_info(model, files_metadata=True)
            size = sum((s.size or 0) for s in (mi.siblings or []) if s.rfilename.endswith(".safetensors"))
            out["hf_model_accessible"] = True
            out["hf_model_sha"] = mi.sha
            out["hf_model_safetensors_gib"] = round(size / 1024**3, 2) if size else None
            out["hf_model_gated"] = getattr(mi, "gated", None)
        except Exception as e:
            out["hf_model_accessible"] = False
            out["hf_model_error"] = f"{type(e).__name__}: {str(e)[:300]}"
    return out


def build_report(model):
    env = sagemaker_environment()
    region, region_src = aws_region()
    itype, itype_src = instance_type(env)
    return {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "sagemaker_environment": env,
        "aws_region": region, "aws_region_source": region_src,
        "instance_type": itype, "instance_type_source": itype_src,
        "os": {"platform": platform.platform(), "release": sh("cat /etc/os-release | head -2"),
               "kernel": platform.release()},
        "python": {"version": platform.python_version(), "executable": sys.executable},
        "packages": package_versions(),
        "gpu": gpu_info(),
        "nvidia_smi": nvidia_smi(),
        "memory_disk": memory_and_disk(),
        "network": network_checks(model),
        "target_model": model,
    }


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    p.add_argument("--out", default="results/server_report.json")
    a = p.parse_args(argv)
    rep = build_report(a.model)
    out = resolve(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=2, default=str) + "\n")
    g = rep["gpu"]
    print(f"Environment : {rep['sagemaker_environment']['type']}")
    print(f"Region      : {rep['aws_region']}  ({rep['aws_region_source']})")
    print(f"Instance    : {rep['instance_type']}  ({rep['instance_type_source']})")
    print(f"Python      : {rep['python']['version']}   torch {g.get('torch')}  CUDA runtime {g.get('torch_cuda_runtime')}")
    print(f"Driver      : {rep['nvidia_smi'].get('query', 'nvidia-smi unavailable')}")
    for dv in g.get("devices", []):
        print(f"GPU {dv['index']}       : {dv['name']}  {dv['total_vram_gib']} GiB total, {dv['free_vram_gib']} GiB free, "
              f"cc {dv['compute_capability']}")
    md = rep["memory_disk"]
    print(f"RAM         : {md.get('ram_total_gib')} GiB total, {md.get('ram_available_gib')} GiB available")
    print(f"Disk free   : " + ", ".join(f"{k} {v['free_gib']} GiB" for k, v in md["disk"].items()))
    n = rep["network"]
    print(f"Internet    : {n.get('internet')}   HF token set: {n.get('hf_token_present')}   "
          f"model accessible: {n.get('hf_model_accessible')}  ({n.get('hf_model_safetensors_gib')} GiB)")
    print(f"Saved       : {out}")


if __name__ == "__main__":
    main()
