# src/experiments/run_all.py
import os
import sys
import time
import subprocess
from pathlib import Path

# 1. Locate the project root dynamically (searches for 'data' or 'src' folders)
CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR
for _ in range(3):
    if (ROOT_DIR / "data").exists() or (ROOT_DIR / "src").exists():
        break
    ROOT_DIR = ROOT_DIR.parent

# 2. Setup the environment to include ROOT_DIR in PYTHONPATH dynamically
env = os.environ.copy()
env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(ROOT_DIR), env.get("PYTHONPATH", "")]))

# 3. List of experiment scripts to run in order
EXPERIMENTS = [
    "scaling.py",
    "adaboost_scaling.py",
    "rf_scaling.py",
    "head_to_head.py",
    "noise_robustness.py",
    "bias_variance.py",
    "unsupervised_analysis.py",
]

def run_experiment(script_name: str) -> bool:
    script_path = CURRENT_DIR / script_name
    if not script_path.exists():
        print(f"\n[!] WARNING: Script not found: {script_name} at {script_path}")
        return False

    print("\n" + "=" * 60)
    print(f"RUNNING: {script_name}")
    print("=" * 60)

    start_time = time.time()

    # Run the script as a separate Python subprocess with correct python executable and environment
    process = subprocess.run(
        [sys.executable, str(script_path)],
        env=env,
        cwd=str(ROOT_DIR)  # Execute from project root to ensure clean context
    )

    elapsed = time.time() - start_time

    if process.returncode == 0:
        print(f"\n[+] SUCCESS: {script_name} finished in {elapsed:.2f} seconds.")
        return True
    else:
        print(f"\n[-] FAILED: {script_name} exited with code {process.returncode} after {elapsed:.2f} seconds.")
        return False

def main():
    print(f"Initializing Master Run...")
    print(f"Project Root: {ROOT_DIR}")
    print(f"Experiments Directory: {CURRENT_DIR}")
    print(f"Total planned experiments: {len(EXPERIMENTS)}\n")

    overall_start = time.time()
    results = {}

    for script in EXPERIMENTS:
        success = run_experiment(script)
        results[script] = "SUCCESS" if success else "FAILED"

    overall_elapsed = time.time() - overall_start

    # Print Final Summary Report
    print("\n" + "#" * 60)
    print("                      FINAL SUMMARY REPORT                      ")
    print("#" * 60)
    print(f"Total Execution Time: {overall_elapsed / 60:.2f} minutes\n")

    for script, status in results.items():
        icon = "Success" if status == "SUCCESS" else "❌"
        print(f" {icon} {script:30s} : {status}")
    print("#" * 60)

if __name__ == "__main__":
    main()