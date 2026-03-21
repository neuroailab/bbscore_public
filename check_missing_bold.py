"""Check which subjects are missing BOLD files in the route-learning dataset."""

import os
import glob

BIDS_DIR = os.environ.get(
    "ROUTE_LEARNING_DATA",
    os.path.join(os.getcwd(), "route-learning"),
)
EXPECTED_RUNS = 14
TASK = "routelearning"


def check_subject(subject_dir):
    sub = os.path.basename(subject_dir)
    func_dir = os.path.join(subject_dir, "func")
    pattern = os.path.join(func_dir, f"{sub}_task-{TASK}_run-*_bold.nii.gz")
    bold_files = sorted(glob.glob(pattern))

    present_runs = set()
    for f in bold_files:
        base = os.path.basename(f)
        run_str = base.split("_run-")[1].split("_")[0]
        present_runs.add(int(run_str))

    expected = set(range(1, EXPECTED_RUNS + 1))
    missing = sorted(expected - present_runs)
    return len(bold_files), missing


def main():
    subject_dirs = sorted(glob.glob(os.path.join(BIDS_DIR, "sub-*")))
    if not subject_dirs:
        print(f"No subjects found in {BIDS_DIR}")
        return

    incomplete = []
    for sd in subject_dirs:
        sub = os.path.basename(sd)
        count, missing = check_subject(sd)
        if count == EXPECTED_RUNS:
            continue
        incomplete.append((sub, count, missing))

    print(f"Checked {len(subject_dirs)} subjects in {BIDS_DIR}\n")

    if not incomplete:
        print("All subjects have 14 BOLD files.")
        return

    print(f"{len(incomplete)} subject(s) with missing BOLD files:\n")
    print(f"  {'Subject':<16} {'Found':>5} {'Missing runs'}")
    print(f"  {'-'*16} {'-'*5} {'-'*30}")
    for sub, count, missing in incomplete:
        runs_str = ", ".join(str(r) for r in missing)
        print(f"  {sub:<16} {count:>5}  {runs_str}")

    print(f"\n{len(subject_dirs) - len(incomplete)} subject(s) are complete (14/14).")


if __name__ == "__main__":
    main()
