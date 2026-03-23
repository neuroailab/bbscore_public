"""
Run early_vs_late_learning.py for all Experiment 1 subject pairs.

Subjects 1-20 (all Exp1 subjects). By default, only subjects that have
task-routelearning BOLD under ROUTE_LEARNING_DATA (see early_vs_late_learning)
are included, so pairs are not wasted on missing preproc. Set
RUN_ALL_EXP1_NO_BOLD_FILTER=1 to use the full 1–20 list anyway.

Iterates over all unique pairs, saves per-pair JSON results,
then aggregates into a single heatmap-ready JSON file.

Parallelism: subprocesses run concurrently (default worker count from CPU count,
capped at 16). Override with EXP1_PARALLEL_JOBS, e.g. EXP1_PARALLEL_JOBS=4.

Child stdout is captured by default so parallel runs do not interleave lines (all
7 LORO folds still run; mixed "Fold k/7" blocks in the terminal were from
multiple pairs writing at once). Set RUN_ALL_EXP1_TEE_CHILD=1 to stream child
output live again. Set RUN_ALL_EXP1_PAIR_LOGS=1 to save full stdout/stderr per
pair under results/pair_logs/.

results/ must be writable by the user running the child processes. If the repo
or results tree is owned by root, either:

  sudo chown -R "$USER:$USER" path/to/bbscore_public/results

or run the driver with sudo using this project’s venv interpreter (so numpy,
nilearn, etc. resolve), e.g. from bbscore_public:

  sudo "$(pwd)/.venv/bin/python" run_all_exp1.py

Plain sudo python3 often misses the venv and will fail on imports.
"""

import subprocess
import sys
import os
import json
import itertools
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
REGION = "hippocampus"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

CANDIDATE_SUBJECTS = [f"sub-Exp1s{i:02d}" for i in (1, 3, 5, 7, 8, 10, 12, 14, 16, 18, 20)]


def _filter_subjects_with_routelearning_bold(candidates):
    import early_vs_late_learning as evl

    root = os.environ.get("ROUTE_LEARNING_DATA", evl.BIDS_DIR)
    ok, skipped = [], []
    for s in candidates:
        try:
            evl.find_bold_files(s, bids_dir=root)
            ok.append(s)
        except FileNotFoundError:
            skipped.append(s)
    return ok, skipped, root


_no_filter = os.environ.get("RUN_ALL_EXP1_NO_BOLD_FILTER", "").strip().lower() in (
    "1",
    "true",
    "yes",
)

if _no_filter:
    SUBJECTS = list(CANDIDATE_SUBJECTS)
    _skipped_subjects = []
    print(
        "RUN_ALL_EXP1_NO_BOLD_FILTER set: using all candidate subjects "
        "(pairs may fail if BOLD is missing).\n"
    )
else:
    SUBJECTS, _skipped_subjects, _data_root = _filter_subjects_with_routelearning_bold(
        CANDIDATE_SUBJECTS
    )
    if _skipped_subjects:
        print(
            f"No routelearning BOLD under {_data_root} — skipping "
            f"{len(_skipped_subjects)} subject(s): {', '.join(_skipped_subjects)}\n"
        )

if len(SUBJECTS) < 2:
    print(
        "Need at least 2 subjects with routelearning BOLD. "
        f"Found: {SUBJECTS}",
        file=sys.stderr,
    )
    sys.exit(1)

def _ensure_results_dir_writable():
    """Exit early if child scripts cannot write per-pair JSON here."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    probe = os.path.join(RESULTS_DIR, ".write_probe_delete_me")
    try:
        with open(probe, "w") as f:
            f.write("ok")
        os.remove(probe)
    except OSError as e:
        print(
            f"ERROR: cannot write to results directory:\n  {RESULTS_DIR}\n"
            f"  ({e})\n"
            "Fix ownership, e.g.:\n"
            f"  sudo chown -R $USER:$USER {RESULTS_DIR}\n"
            "Or re-run this script with sudo using the project venv Python, e.g.:\n"
            f"  sudo {os.path.join(SCRIPT_DIR, '.venv', 'bin', 'python')} "
            f"{os.path.join(SCRIPT_DIR, 'run_all_exp1.py')}\n",
            file=sys.stderr,
        )
        sys.exit(1)


_ensure_results_dir_writable()


def _max_workers(n_pairs: int) -> int:
    env = os.environ.get("EXP1_PARALLEL_JOBS")
    if env is not None:
        try:
            w = int(env.strip())
            return max(1, min(w, n_pairs))
        except ValueError:
            pass
    cpus = os.cpu_count() or 4
    return max(1, min(cpus, n_pairs, 16))


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes")


def _run_pair(idx: int, total: int, sub_a: str, sub_b: str) -> tuple:
    t0 = time.time()
    cmd = [
        sys.executable,
        "early_vs_late_learning.py",
        "--subject-a",
        sub_a,
        "--subject-b",
        sub_b,
    ]
    tee = _env_truthy("RUN_ALL_EXP1_TEE_CHILD")
    save_logs = _env_truthy("RUN_ALL_EXP1_PAIR_LOGS")

    if tee:
        proc = subprocess.run(cmd, cwd=SCRIPT_DIR)
        out, err = "", ""
    else:
        proc = subprocess.run(
            cmd, cwd=SCRIPT_DIR, capture_output=True, text=True
        )
        out = proc.stdout or ""
        err = proc.stderr or ""
        if save_logs:
            log_dir = os.path.join(RESULTS_DIR, "pair_logs")
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(log_dir, f"{sub_a}__{sub_b}.log")
            with open(log_path, "w") as f:
                f.write(out)
                if err:
                    f.write("\n--- stderr ---\n")
                    f.write(err)

    elapsed = time.time() - t0
    return idx, total, sub_a, sub_b, proc.returncode, elapsed, out, err


pairs = list(itertools.combinations(SUBJECTS, 2))
total = len(pairs)
workers = _max_workers(total)

print(f"Subjects ({len(SUBJECTS)}): {', '.join(SUBJECTS)}")
print(f"Total pairs: {total}")
print(f"Parallel workers: {workers}\n")

failed = []
_log_lock = threading.Lock()

with ThreadPoolExecutor(max_workers=workers) as executor:
    futures = {
        executor.submit(_run_pair, idx, total, sub_a, sub_b): (sub_a, sub_b)
        for idx, (sub_a, sub_b) in enumerate(pairs, 1)
    }
    for future in as_completed(futures):
        idx, _, sub_a, sub_b, rc, elapsed, out, err = future.result()
        with _log_lock:
            if rc != 0:
                print(
                    f"  *** FAILED pair {idx}/{total}: {sub_a} x {sub_b} "
                    f"(exit {rc}) ***"
                )
                failed.append((sub_a, sub_b))
                if out.strip():
                    print(out, end="" if out.endswith("\n") else "\n")
                if err.strip():
                    print(err, end="" if err.endswith("\n") else "\n", file=sys.stderr)
            else:
                print(f"  OK {idx}/{total}: {sub_a} x {sub_b} ({elapsed:.1f}s)")

# ── Aggregate per-pair JSONs into a heatmap-ready summary ──
print(f"\n{'='*60}")
print("Aggregating results for heatmap...")

heatmap = {
    "subjects": SUBJECTS,
    "region": REGION,
    "a_to_b_change": {},
    "b_to_a_change": {},
    "a_to_b_early": {},
    "a_to_b_late": {},
    "b_to_a_early": {},
    "b_to_a_late": {},
}

for sub_a, sub_b in pairs:
    pair_file = os.path.join(
        RESULTS_DIR, f"early_vs_late_{sub_a}_{sub_b}_{REGION}.json"
    )
    if not os.path.exists(pair_file):
        continue

    with open(pair_file) as f:
        r = json.load(f)

    key = f"{sub_a}__{sub_b}"
    heatmap["a_to_b_change"][key] = r["a_to_b"]["change"]
    heatmap["b_to_a_change"][key] = r["b_to_a"]["change"]
    heatmap["a_to_b_early"][key] = r["a_to_b"]["early"]["median_r"]
    heatmap["a_to_b_late"][key] = r["a_to_b"]["late"]["median_r"]
    heatmap["b_to_a_early"][key] = r["b_to_a"]["early"]["median_r"]
    heatmap["b_to_a_late"][key] = r["b_to_a"]["late"]["median_r"]

heatmap_file = os.path.join(RESULTS_DIR, "exp1_heatmap.json")
with open(heatmap_file, "w") as f:
    json.dump(heatmap, f, indent=2)

print(f"Heatmap data saved to {heatmap_file}")
print(f"\n{'='*60}")
print(f"ALL DONE: {total - len(failed)}/{total} pairs succeeded")
if failed:
    print(f"\nFailed pairs ({len(failed)}):")
    for a, b in failed:
        print(f"  {a}  x  {b}")
print(f"{'='*60}")
