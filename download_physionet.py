"""
download_physionet.py
=====================
Download all PhysioNet 2019 Sepsis Challenge PSV files directly from the
directory listings, then build the Phase 1 processed dataset.

Uses concurrent downloads (ThreadPoolExecutor) for speed.
"""

import os
import re
import sys
import time
import urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── Paths ────────────────────────────────────────────────────────────────────
RAW_DIR = Path("e:/DMSRDE INTERNSHIP PROJECT/health-risk-triage/data/raw/physionet2019")

# ── PhysioNet directory URLs ─────────────────────────────────────────────────
BASE = "https://physionet.org/files/challenge-2019/1.0.0/training"
SETS = {
    "training_setA": f"{BASE}/training_setA/",
    "training_setB": f"{BASE}/training_setB/",
}

MAX_WORKERS = 20  # concurrent download threads


def get_psv_filenames(directory_url: str) -> list[str]:
    """Scrape the PhysioNet HTML directory listing for .psv file names."""
    print(f"  [LIST] Fetching file listing from {directory_url} ...")
    resp = urllib.request.urlopen(directory_url, timeout=30)
    html = resp.read().decode("utf-8")
    # PhysioNet lists files as links like <a href="p000001.psv">
    filenames = re.findall(r'href="(p\d+\.psv)"', html)
    print(f"         Found {len(filenames):,} PSV files")
    return filenames


def download_one(url: str, dest: Path) -> bool:
    """Download a single file. Returns True on success."""
    if dest.exists():
        return True  # skip already downloaded
    try:
        urllib.request.urlretrieve(url, dest)
        return True
    except Exception:
        return False


def download_set(set_name: str, dir_url: str):
    """Download all PSV files for one training set."""
    print(f"\n{'='*60}")
    print(f"  Downloading {set_name}")
    print(f"{'='*60}")

    filenames = get_psv_filenames(dir_url)
    if not filenames:
        print(f"  [ERR] No files found for {set_name}!")
        return 0

    # Check how many are already downloaded
    already = sum(1 for f in filenames if (RAW_DIR / f).exists())
    to_download = len(filenames) - already
    print(f"  Already present: {already:,}  |  To download: {to_download:,}")

    if to_download == 0:
        print(f"  [OK] All files already downloaded for {set_name}")
        return len(filenames)

    # Build download tasks
    tasks = []
    for fname in filenames:
        dest = RAW_DIR / fname
        if not dest.exists():
            tasks.append((f"{dir_url}{fname}", dest))

    # Download concurrently
    success = already
    failed = 0
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(download_one, url, dest): (url, dest) for url, dest in tasks}
        for i, future in enumerate(as_completed(futures), 1):
            if future.result():
                success += 1
            else:
                failed += 1
                url, dest = futures[future]
                print(f"  [WARN] Failed: {dest.name}")

            if i % 1000 == 0 or i == len(tasks):
                elapsed = time.time() - start_time
                rate = i / elapsed if elapsed > 0 else 0
                print(f"  ... {i:,}/{len(tasks):,} done  ({rate:.0f} files/sec)")

    print(f"  [OK] {set_name}: {success:,} OK, {failed} failed")
    return success


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    total = 0
    for set_name, url in SETS.items():
        total += download_set(set_name, url)

    print(f"\n{'='*60}")
    actual = len(list(RAW_DIR.glob("*.psv")))
    print(f"[DONE] Total PSV files in raw dir: {actual:,}")
    print(f"       Location: {RAW_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
