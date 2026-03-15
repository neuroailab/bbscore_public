import os
import glob
import shutil
from data.LeBel2023 import LeBel2023TRAssembly

subjects = ["UTS01", "UTS02", "UTS03"]
a = LeBel2023TRAssembly(subjects=subjects)

print("Data root:", a.data_dir)

for s in subjects:
    # Remove stale local copies across all supported layout variants.
    for p in a._candidate_data_dirs(s):
        if os.path.exists(p):
            shutil.rmtree(p)

    try:
        files = a._ensure_data_downloaded(s)
        print(s, "hf5 files:", len(files))
    except Exception as e:
        print(s, "download failed:", e)

print("\nFinal counts:")
for s in subjects:
    files = a._collect_hf5_files(s)
    print(s, len(files))
    if files:
        print("  sample:", files[0])