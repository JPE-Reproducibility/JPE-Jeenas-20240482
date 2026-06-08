# Python script to iterate over "DD_work_run.py" and set up parallelized workflow locally

import sys
import time
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# Define the root directory, robust to interactive or file execution
try:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
except NameError:
    PROJECT_ROOT = Path.cwd()

N = 60
MAX_WORKERS = 5          # with 6 cores, consider 5
RETRIES = 1              # re-run failed jobs once
SCRIPT = PROJECT_ROOT / "code" / "Python" / "DD_work_run.py"

# Create log directory
LOG_DIR = Path(PROJECT_ROOT / "interim_output/_DD_par_logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

def run_one(i: int) -> tuple[int, int, float, str]:
    t0 = time.time()
    log_path = LOG_DIR / f"DDgen_output_{i:02d}.log"   
    cmd = [sys.executable, "-u", SCRIPT, str(i)]
    with open(log_path, "w") as f:
        p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    return (i, p.returncode, time.time() - t0, log_path)

def main():
    jobs = list(range(N))
    attempts = {i: 0 for i in jobs}
    failed = set(jobs)

    while failed:
        batch = sorted(failed)
        failed.clear()

        print(f"Launching {len(batch)} jobs with {MAX_WORKERS} workers...")
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futs = {ex.submit(run_one, i): i for i in batch}
            for fut in as_completed(futs):
                i, rc, dt, log = fut.result()
                if rc == 0:
                    print(f"[OK]  {i:02d}  {dt:8.1f}s  log={log}")
                else:
                    attempts[i] += 1
                    print(f"[FAIL]{i:02d}  {dt:8.1f}s  rc={rc}  log={log}")
                    if attempts[i] <= RETRIES:
                        failed.add(i)

        if failed:
            print(f"Retrying failed jobs: {sorted(failed)}")

    print("All jobs completed.")

if __name__ == "__main__":
    main()
