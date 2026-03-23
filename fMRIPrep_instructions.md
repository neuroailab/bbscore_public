### fMRIPrep setup for RouteLearning (M1 Mac, 8 GB RAM): faster + scientifically valid baseline

This guide is optimized for:
- Apple M1 (2021 Mac)
- 8 GB memory
- RouteLearning workflow that uses MNI-space BOLD + confounds + Harvard-Oxford mask

The key speed optimization is **skipping FreeSurfer reconstruction** with:
- `--fs-no-reconall`

For this specific pipeline, that is usually a good tradeoff.

---

## 1) Prerequisites

1. Install and start Docker Desktop.
2. Confirm Docker works:

```bash
docker --version
docker run --rm hello-world
```

3. Make sure the BIDS dataset exists at:
- `bbscore_public/route-learning/`

---

## 2) Docker Desktop settings (important on 8 GB RAM)

In Docker Desktop settings, start with:
- CPUs: **4**
- Memory: **6 GB** (or 5 GB if your Mac struggles)
- Swap: leave default

Why: fMRIPrep can otherwise pressure memory and become unstable on 8 GB systems.

---

## 3) Create output/work directories

From repo root (`/Users/sudharsansundar/bbscore_public`):

```bash
mkdir -p route-learning/derivatives/fmriprep
mkdir -p route-learning/derivatives/fmriprep-work
```

---

## 4) Set paths

```bash
BIDS_DIR="/Users/sudharsansundar/bbscore_public/route-learning"
OUT_DIR="/Users/sudharsansundar/bbscore_public/route-learning/derivatives/fmriprep"
WORK_DIR="/Users/sudharsansundar/bbscore_public/route-learning/derivatives/fmriprep-work"
FS_LICENSE="/absolute/path/to/license.txt"
```

Note: with `--fs-no-reconall`, FreeSurfer is skipped, but keeping the license mount/path is harmless and can avoid version-specific surprises.

---

## 5) Recommended command (fast + valid for your use case)

Run one subject first:

```bash
docker run --rm -it \
  -v "${BIDS_DIR}":/data:ro \
  -v "${OUT_DIR}":/out \
  -v "${WORK_DIR}":/work \
  -v "${FS_LICENSE}":/opt/freesurfer/license.txt:ro \
  nipreps/fmriprep:23.2.0 \
  /data /out participant \
  --participant-label Exp1s01 \
  --output-spaces MNI152NLin2009cAsym \
  --fs-no-reconall \
  --nthreads 4 \
  --omp-nthreads 2 \
  --work-dir /work \
  --fs-license-file /opt/freesurfer/license.txt \
  --skip-bids-validation
```

### Why these flags
- `--output-spaces MNI152NLin2009cAsym`: matches your loader expectation.
- `--fs-no-reconall`: major runtime reduction.
- `--nthreads 4 --omp-nthreads 2`: safer default for 8 GB RAM.
- `--skip-bids-validation`: small speedup.

---

## 6) Process more subjects (recommended: one at a time)

On an 8 GB machine, process subjects individually by changing:
- `--participant-label Exp1s01` to e.g. `Exp1s03`, `Exp1s05`, etc.

This is slower wall-clock than a large batch but much safer for memory/disk.

---

## 7) Verify required outputs exist

For each processed subject, confirm files like:

- `route-learning/derivatives/fmriprep/sub-Exp1s01/func/sub-Exp1s01_task-routelearning_run-01_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz`
- `route-learning/derivatives/fmriprep/sub-Exp1s01/func/sub-Exp1s01_task-routelearning_run-01_space-MNI152NLin2009cAsym_desc-confounds_timeseries.tsv`

If those exist for both source and target subjects, you can run RouteLearning benchmarks.

---

## 8) Disk and runtime tips (M1/8GB)

- fMRIPrep can take hours per subject.
- `route-learning/derivatives/fmriprep-work/` is temporary and can become large.
- After a subject succeeds, you can usually delete the work dir:

```bash
rm -rf "/Users/sudharsansundar/bbscore_public/route-learning/derivatives/fmriprep-work"
mkdir -p "/Users/sudharsansundar/bbscore_public/route-learning/derivatives/fmriprep-work"
```

---

## 9) Run your benchmark

Example:

```bash
python run.py --benchmark RouteLearningExp1s01toExp1s02Hippo --metric ridge --model None --layer dummy
```

`--layer` is still required by `run.py` argument parsing, even though RouteLearning benchmarks do not use model features.

---

## 10) Troubleshooting (M1 + 8 GB)

### A) Container exits early / process gets killed (likely out-of-memory)

Symptoms:
- Docker container stops unexpectedly
- `Killed` or memory-related errors in logs

What to do:
1. Reduce thread load:
   - `--nthreads 2 --omp-nthreads 1`
2. In Docker Desktop, lower CPU count and keep memory around 5–6 GB.
3. Run one subject at a time (no batching).
4. Close other heavy apps while fMRIPrep runs.

---

### B) “No space left on device” / disk fills up

Symptoms:
- write failures during preprocessing
- Docker or fMRIPrep reports no space

What to do:
1. Delete temporary work files between subjects:

```bash
rm -rf "/Users/sudharsansundar/bbscore_public/route-learning/derivatives/fmriprep-work"
mkdir -p "/Users/sudharsansundar/bbscore_public/route-learning/derivatives/fmriprep-work"
```

2. Keep only subjects you need in `derivatives/fmriprep`.
3. In Docker Desktop, run “Clean / Prune” if image/cache bloat is large.

---

### C) Command finishes, but expected files are missing

Symptoms:
- no `*_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz`
- no `*_desc-confounds_timeseries.tsv`

Checks:
1. Confirm `--output-spaces MNI152NLin2009cAsym` was used.
2. Confirm participant label format is correct:
   - use `Exp1s01`, not `sub-Exp1s01`
3. Check output path:
   - should be `route-learning/derivatives/fmriprep/sub-Exp.../func/`
4. Check logs for run-level failures in the container output.

---

### D) fMRIPrep runs but is very slow

What to do:
1. Keep `--fs-no-reconall`.
2. Process fewer subjects first (source + target only).
3. Keep one output space only (`MNI152NLin2009cAsym`).
4. Ensure your `BIDS_DIR` mount is local SSD (not cloud-synced/network).

---

### E) Docker mount/path errors

Symptoms:
- `No such file or directory`
- permission or mount path errors

What to do:
1. Re-check variable values:

```bash
echo "$BIDS_DIR"
echo "$OUT_DIR"
echo "$WORK_DIR"
echo "$FS_LICENSE"
```

2. Ensure each path exists on host:

```bash
ls "$BIDS_DIR"
ls "$OUT_DIR"
ls "$WORK_DIR"
ls "$FS_LICENSE"
```

3. Re-run command after correcting paths.

---

### F) RouteLearning loader still says derivatives not found

What to do:
1. Verify exact filename pattern exists:
   - `sub-Exp1s01_task-routelearning_run-01_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz`
2. Ensure files are under:
   - `route-learning/derivatives/fmriprep/sub-Exp.../func/`
3. Confirm subject you processed matches your benchmark source/target.

---

### G) Sanity check command

Before running a full benchmark, check one subject has expected outputs:

```bash
ls "/Users/sudharsansundar/bbscore_public/route-learning/derivatives/fmriprep/sub-Exp1s01/func/" | head
```

If you see `desc-preproc_bold.nii.gz` and `desc-confounds_timeseries.tsv` with `space-MNI152NLin2009cAsym`, preprocessing is set up correctly.

---

## 11) Expected runtime per subject (M1, 8 GB RAM)

These are practical ballpark ranges for the recommended command in this guide:
- `--fs-no-reconall`
- `--output-spaces MNI152NLin2009cAsym`
- one subject at a time
- moderate thread settings (e.g., `--nthreads 4 --omp-nthreads 2`)

| Scenario | Approx runtime per subject |
|---|---|
| Best case (clean run, low system load) | ~1.5 to 3 hours |
| Typical | ~3 to 6 hours |
| Worst case (memory pressure, throttling, retries) | ~6 to 10+ hours |

### What shifts runtime up/down the most
- **Up (slower):** high memory pressure, too many threads, heavy background apps, thermal throttling, low free disk.
- **Down (faster):** `--fs-no-reconall`, one output space only, one subject at a time, enough free disk, minimal background load.