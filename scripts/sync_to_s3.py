"""Upload results (and the dataset/configs) to an EXISTING S3 bucket. Never creates buckets,
never uploads model weights. Credentials come from the SageMaker execution role.

  export KVR_S3_BUCKET=my-existing-bucket          # or set s3.bucket in the config
  export KVR_S3_PREFIX=kv_cache_research            # optional
  python scripts/sync_to_s3.py [--dry-run]
"""
import argparse
import datetime as dt
import os
import sys

import _bootstrap  # noqa: F401

from utils.config import load_config
from utils.paths import PROJECT_ROOT

INCLUDE_DIRS = ["results", "configs", "datasets"]
INCLUDE_FILES = ["requirements-lock.txt"]
SKIP_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".gguf")
MAX_FILE_MB = 200


def files_to_upload():
    for d in INCLUDE_DIRS:
        for root, _, files in os.walk(PROJECT_ROOT / d):
            for f in files:
                p = os.path.join(root, f)
                if f.endswith(SKIP_SUFFIXES) or f == ".gitkeep" or os.path.getsize(p) > MAX_FILE_MB * 1e6:
                    continue
                yield p
    for f in INCLUDE_FILES:
        p = PROJECT_ROOT / f
        if p.exists():
            yield str(p)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment_config.yaml")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--run-label", default=dt.datetime.now().strftime("%Y%m%d-%H%M%S"))
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    s3cfg = cfg.get("s3") or {}
    bucket = os.environ.get("KVR_S3_BUCKET") or s3cfg.get("bucket")
    prefix = (os.environ.get("KVR_S3_PREFIX") or s3cfg.get("prefix") or "kv_cache_research").strip("/")
    if not bucket:
        sys.exit("No bucket set. export KVR_S3_BUCKET=<existing bucket> or set s3.bucket in the config.")
    import boto3
    from botocore.exceptions import ClientError

    s3 = boto3.client("s3")
    try:
        s3.head_bucket(Bucket=bucket)
    except ClientError as e:
        sys.exit(f"Cannot access bucket '{bucket}' ({e.response['Error'].get('Code')}). "
                 "This script does not create buckets; check the name and the execution role's permissions.")
    files = list(files_to_upload())
    dest = f"{prefix}/{a.run_label}"
    print(f"{len(files)} files -> s3://{bucket}/{dest}/")
    for p in files:
        key = f"{dest}/{os.path.relpath(p, PROJECT_ROOT)}"
        if a.dry_run:
            print("  would upload", key)
        else:
            s3.upload_file(p, bucket, key)
    print("done" if not a.dry_run else "dry run only")


if __name__ == "__main__":
    main()
