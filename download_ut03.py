import os
import glob
import shutil
from data.LeBel2023 import LeBel2023TRAssembly

subjects = ["UTS01", "UTS02", "UTS03"]
a = LeBel2023TRAssembly(subjects=subjects)

print("Data dir:", a.data_dir)

for s in subjects:
    subj_path = os.path.join(a.data_dir, s)
    if os.path.exists(subj_path):
        shutil.rmtree(subj_path)

    a.fetch(
        source=f"s3://openneuro.org/ds003020/derivative/preprocessed_data/{s}/",
        target_dir=a.data_dir,
        filename=s,
        method="s3",
        anonymous=True,
        force_download=True,
    )

    n = len(glob.glob(os.path.join(subj_path, "**", "*.hf5"), recursive=True))
    print(s, "hf5 files:", n)



import os
import boto3
from botocore import UNSIGNED
from botocore.config import Config

root = os.environ["SCIKIT_LEARN_DATA"]
base = os.path.join(root, "LeBel2023TRAssembly", "ds003020", "derivative", "preprocessed_data")
subjects = ["UTS01", "UTS02", "UTS03"]

s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
bucket = "openneuro.org"

for subj in subjects:
    prefix = f"ds003020/derivative/preprocessed_data/{subj}/"
    out_dir = os.path.join(base, subj)
    os.makedirs(out_dir, exist_ok=True)

    n = 0
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("/"):
                continue
            rel = key[len(prefix):]
            if not rel.endswith(".hf5"):
                continue
            local = os.path.join(out_dir, rel)
            os.makedirs(os.path.dirname(local), exist_ok=True)
            s3.download_file(bucket, key, local)
            n += 1
    print(subj, "downloaded hf5:", n)