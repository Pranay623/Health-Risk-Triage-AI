"""
retry_download.py
=================
Retry downloading missing PSV files from PhysioNet training_setB.
Uses lower concurrency and retries to handle server throttling.
"""

import re
import time
import urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

RAW_DIR = Path("e:/DMSRDE INTERNSHIP PROJECT/health-risk-triage/data/raw/physionet2019")
BASE = "https://physionet.org/files/challenge-2019/1.0.0/training"
MAX_WORKERS = 8  # reduced concurrency to avoid throttling
MAX_RETRIES = 3


def get_psv_filenames(directory_url: str) -> list[str]:
    """Scrape the PhysioNet HTML directory listing for .psv file names."""
    print(f"  [LIST] Fetching file listing from {directory_url} ...")
    resp = urllib.request.urlopen(directory_url, timeout=30)
    html = resp.read().decode("utf-8")
    filenames = re.findall(r'href="(p\d+\.psv)"', html)
    print(f"         Found {len(filenames):,} PSV files in listing")
    return filenames


def download_one_with_retry(url: str, dest: Path, retries: int = MAX_RETRIES) -> bool:
    """Download a single file with retries. Returns True on success."""
    for attempt in range(retries):
        try:
            urllib.request.urlretrieve(url, dest)
            return True
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(1 * (attempt + 1))  # backoff
            else:
                return False
    return False


def main():
    # Get the full listing of what SHOULD be in setB
    setB_url = f"{BASE}/training_setB/"
    all_setB = get_psv_filenames(setB_url)

    # Find which files we're missing
    missing = [f for f in all_setB if not (RAW_DIR / f).exists()]
    print(f"\n  Total setB files in listing: {len(all_setB):,}")
    print(f"  Already downloaded:          {len(all_setB) - len(missing):,}")
    print(f"  Missing (to retry):          {len(missing):,}")

    if not missing:
        print("  [OK] All files present!")
        return

    # First, verify a sample of "missing" files actually exist on the server
    print(f"\n  [CHECK] Verifying a sample of missing files exist on server...")
    test_files = missing[:5]
    real_missing = []
    for f in test_files:
        url = f"{setB_url}{f}"
        try:
            req = urllib.request.Request(url, method="HEAD")
            resp = urllib.request.urlopen(req, timeout=10)
            print(f"    {f} -> EXISTS (status {resp.status})")
            real_missing.append(f)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                print(f"    {f} -> DOES NOT EXIST (404)")
            else:
                print(f"    {f} -> ERROR ({e.code})")
                real_missing.append(f)
        except Exception as e:
            print(f"    {f} -> ERROR ({e})")
            real_missing.append(f)

    if not real_missing and len(test_files) > 0:
        print("\n  [INFO] Sample files all returned 404 - they don't exist on PhysioNet.")
        print("  [INFO] The directory listing contained phantom entries.")
        print("  [INFO] The dataset may simply have fewer files than listed.")

        # Count actual vs phantom
        print("\n  [CHECK] Verifying more files to find boundary...")
        exists_count = 0
        not_exists_count = 0
        for f in missing[:50]:
            url = f"{setB_url}{f}"
            try:
                req = urllib.request.Request(url, method="HEAD")
                resp = urllib.request.urlopen(req, timeout=10)
                exists_count += 1
            except:
                not_exists_count += 1
        print(f"    Of 50 tested: {exists_count} exist, {not_exists_count} don't exist")

        if exists_count == 0:
            total_in_dir = len(list(RAW_DIR.glob("*.psv")))
            print(f"\n  [DONE] All real files are downloaded. Total: {total_in_dir:,}")
            return

    # If files do exist, retry downloading them
    print(f"\n  [DL] Retrying {len(missing):,} files with {MAX_WORKERS} workers...")
    tasks = [(f"{setB_url}{f}", RAW_DIR / f) for f in missing]

    success = 0
    failed = 0
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(download_one_with_retry, url, dest): (url, dest) for url, dest in tasks}
        for i, future in enumerate(as_completed(futures), 1):
            if future.result():
                success += 1
            else:
                failed += 1

            if i % 500 == 0 or i == len(tasks):
                elapsed = time.time() - start_time
                rate = i / elapsed if elapsed > 0 else 0
                print(f"  ... {i:,}/{len(tasks):,} done  ({rate:.0f} files/sec)  OK={success} FAIL={failed}")

    total = len(list(RAW_DIR.glob("*.psv")))
    print(f"\n  [DONE] Retry complete. Success={success}, Failed={failed}")
    print(f"  Total PSV files in raw dir: {total:,}")


if __name__ == "__main__":
    main()
