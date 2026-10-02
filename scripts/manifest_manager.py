"""
scripts/manifest_manager.py
Manages download manifests, SHA-256 checksums, resume checkpoints, and retry backoff:
- Manifest fields:
    dataset, dataset_id, variable, start_datetime, end_datetime,
    bbox, depth_range, output_file, size, checksum, status,
    retry_count, error, timestamp
- Statuses: PLANNED, DOWNLOADING, COMPLETE, FAILED, CORRUPTED, SKIPPED
- SHA-256 checksum recorded in data/checksums.csv
- Existing validated chunks are never re-downloaded (idempotent resume).
"""

import os
import sys
import json
import time
import hashlib
import pandas as pd
from datetime import datetime, timezone

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

MANIFEST_JSON_PATH = "data/manifests/download_manifest.json"
MANIFEST_CSV_PATH = "data/manifests/download_manifest.csv"
CHECKSUMS_CSV_PATH = "data/checksums.csv"

def compute_sha256(filepath):
    """Computes SHA-256 checksum of a file."""
    if not os.path.exists(filepath):
        return None
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(65536), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

class ManifestManager:
    def __init__(self, json_path=MANIFEST_JSON_PATH, csv_path=MANIFEST_CSV_PATH, checksums_path=CHECKSUMS_CSV_PATH):
        self.json_path = json_path
        self.csv_path = csv_path
        self.checksums_path = checksums_path
        self.manifest = self._load_manifest()
        self._ensure_checksums_file()

    def _ensure_checksums_file(self):
        os.makedirs(os.path.dirname(self.checksums_path), exist_ok=True)
        if not os.path.exists(self.checksums_path):
            with open(self.checksums_path, "w") as f:
                f.write("timestamp,filepath,sha256,size_bytes\n")

    def _load_manifest(self):
        os.makedirs(os.path.dirname(self.json_path), exist_ok=True)
        if os.path.exists(self.json_path):
            try:
                with open(self.json_path, "r") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save_manifest(self):
        with open(self.json_path, "w") as f:
            json.dump(self.manifest, f, indent=2)
        # Also export to CSV
        if self.manifest:
            df = pd.DataFrame(list(self.manifest.values()))
            df.to_csv(self.csv_path, index=False)

    def get_chunk_key(self, dataset, variable, start_datetime, end_datetime):
        """Constructs unique chunk identifier."""
        return f"{dataset.upper()}_{variable.upper()}_{start_datetime}_{end_datetime}".replace(":", "-")

    def is_chunk_complete(self, chunk_key):
        """Checks if a chunk was already successfully downloaded and validated."""
        if chunk_key in self.manifest:
            rec = self.manifest[chunk_key]
            out_file = rec.get("output_file")
            if rec.get("status") == "COMPLETE" and out_file and os.path.exists(out_file):
                # Verify file size > 0
                if os.path.getsize(out_file) > 0:
                    return True
        return False

    def record_chunk(self, dataset, dataset_id, variable, start_datetime, end_datetime,
                     bbox, depth_range, output_file, status="PLANNED", error=None):
        chunk_key = self.get_chunk_key(dataset, variable, start_datetime, end_datetime)
        
        existing = self.manifest.get(chunk_key, {})
        retry_count = existing.get("retry_count", 0)
        
        file_size = os.path.getsize(output_file) if output_file and os.path.exists(output_file) else 0
        checksum = compute_sha256(output_file) if output_file and os.path.exists(output_file) and file_size > 0 else None

        record = {
            "chunk_key": chunk_key,
            "dataset": dataset,
            "dataset_id": dataset_id,
            "variable": variable,
            "start_datetime": str(start_datetime),
            "end_datetime": str(end_datetime),
            "bbox": str(bbox),
            "depth_range": str(depth_range),
            "output_file": output_file,
            "size": file_size,
            "checksum": checksum,
            "status": status,
            "retry_count": retry_count,
            "error": str(error) if error else None,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

        self.manifest[chunk_key] = record
        self._save_manifest()

        # If complete, log to checksums.csv
        if status == "COMPLETE" and checksum:
            with open(self.checksums_path, "a") as f:
                f.write(f"{record['timestamp']},{output_file},{checksum},{file_size}\n")

        return record

    def update_status(self, chunk_key, status, error=None, output_file=None):
        if chunk_key in self.manifest:
            rec = self.manifest[chunk_key]
            rec["status"] = status
            rec["timestamp"] = datetime.now(timezone.utc).isoformat()
            if error:
                rec["error"] = str(error)
                rec["retry_count"] = rec.get("retry_count", 0) + 1
            if output_file and os.path.exists(output_file):
                rec["output_file"] = output_file
                rec["size"] = os.path.getsize(output_file)
                rec["checksum"] = compute_sha256(output_file)
                if status == "COMPLETE" and rec["checksum"]:
                    with open(self.checksums_path, "a") as f:
                        f.write(f"{rec['timestamp']},{output_file},{rec['checksum']},{rec['size']}\n")
            self._save_manifest()

def retry_with_backoff(operation_fn, max_retries=3, initial_delay=2.0, backoff_factor=2.0):
    """
    Executes operation_fn with exponential backoff on transient errors (timeouts, network errors).
    """
    delay = initial_delay
    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            return operation_fn()
        except Exception as e:
            last_exc = e
            print(f"[RETRY] Attempt {attempt}/{max_retries} failed: {e}. Retrying in {delay:.1f}s...")
            time.sleep(delay)
            delay *= backoff_factor
    raise last_exc
