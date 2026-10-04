"""
scripts/finish_raw_acquisition.py
Completes the remaining 2020 raw acquisition for OSCAR and CCMP:
- OSCAR: 2020-02-01 to 2020-12-31 (monthly batching via podaac-data-downloader)
- CCMP: 2020-06-04 to 2020-12-31 (concurrent daily downloads via RSS HTTP)
"""

import os
import sys
import time

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from scripts.download_oscar import download_oscar_period
from scripts.download_ccmp import download_ccmp_period

def main():
    print("=" * 70)
    print("FINISHING REMAINING 2020 RAW DATA ACQUISITION")
    print("=" * 70)
    t0 = time.time()

    print("\n>>> PHASE 1: OSCAR Surface Currents (2020-02-01 to 2020-12-31) <<<")
    try:
        oscar_files = download_oscar_period("2020-02-01", "2020-12-31")
        print(f"[OSCAR COMPLETE] Total OSCAR files processed: {len(oscar_files)}")
    except Exception as e:
        print(f"[OSCAR ERROR] {e}")
        sys.exit(1)

    print("\n>>> PHASE 2: CCMP Surface Winds (2020-06-04 to 2020-12-31) <<<")
    try:
        ccmp_files = download_ccmp_period("2020-06-04", "2020-12-31", max_workers=6)
        print(f"[CCMP COMPLETE] Total CCMP files processed: {len(ccmp_files)}")
    except Exception as e:
        print(f"[CCMP ERROR] {e}")
        sys.exit(1)

    elapsed = time.time() - t0
    print("\n" + "=" * 70)
    print(f"RAW ACQUISITION FINISHED SUCCESSFULLY in {elapsed/60:.1f} minutes")
    print("=" * 70)

if __name__ == "__main__":
    main()
