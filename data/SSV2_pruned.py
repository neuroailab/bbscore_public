import os
import csv
from typing import Optional, Callable, Tuple, List

import av
import torch
from PIL import Image
from sklearn.datasets import get_data_home
import numpy as np
from data.base import BaseDataset
from data.augmentations import rand_augment_transform, _FILL, randaugment_parallel, randaugment_threaded
import shutil


class SSV2Pruned(BaseDataset):
    """Dataset for the pruned SomethingSomethingV2 videos, using fetch_and_extract."""

    BUCKET = "gs://bbscore_datasets/SSV2_pruned"

    # Preset list of original class IDs (sparse)
    ORIGINAL_CLASSES = [
        0, 1, 2, 3, 4, 5, 6, 8, 10, 12,
        13, 14, 15, 16, 22, 23, 31, 33, 38,
        39, 46, 51, 53, 90, 91, 92, 122,
        124, 130, 138, 139, 141, 143, 144,
        150, 151, 156, 170, 171, 173
    ]

    # Mapping from original class ID to dense 0..N-1 index
    CLASS_TO_IDX = {orig: idx for idx, orig in enumerate(ORIGINAL_CLASSES)}

    def __init__(
        self,
        root_dir: Optional[str] = None,
        overwrite: bool = False,
        preprocess: Optional[Callable] = None,
        apply_randaugment: bool = False,
        randaugment_config_str: str = 'rand-m7-n2-mstd0.3',
        train: bool = True,
    ):
        super().__init__(root_dir)
        self.overwrite = overwrite
        self.preprocess = preprocess
        self.train = train
        self.apply_randaugment = apply_randaugment
        self.randaugment_transform = None

        if self.apply_randaugment and self.train:
            hparams_vid = {"img_mean": _FILL,
                           "translate_const": 50, "magnitude_std": 0.3}
            self.randaugment_transform = rand_augment_transform(
                randaugment_config_str, hparams_vid)

        self.stimulus_data: List[str] = []
        self.labels: List[int] = []
        self.target_fps = 12.0
        self.video_path_base = "videos"
        self.video_folder_name = "ssv2_train_videos" if train else "ssv2_test_videos"
        self.video_path = os.path.join(
            self.root_dir, self.video_path_base, self.video_folder_name)
        self._prepare_videos()

    def _download_ssv2_data(self):
        """Download (and unzip) the pruned CSV & video ZIP via fetch_and_extract."""
        os.makedirs(self.root_dir, exist_ok=True)

        csv_fname = "train_classes_map.csv" if self.train else "test_classes_map.csv"
        zip_fname = "ssv2_train_videos.zip" if self.train else "ssv2_test_videos.zip"
        local_csv = os.path.join(self.root_dir, csv_fname)

        # Download CSV
        if self.overwrite or not os.path.isfile(local_csv):
            print(f"Fetching CSV to {local_csv}...")
            try:
                self.fetch_and_extract(
                    source=f"{self.BUCKET}/{csv_fname}",
                    target_dir=self.root_dir,
                    filename=csv_fname,
                    extract=False,
                    method="gcs",
                    force_download=self.overwrite,
                )
                print(f"Successfully downloaded CSV: {csv_fname}")
            except Exception as e:
                print(f"Error downloading CSV {csv_fname}: {e}")
                raise
        else:
            print(f"Found existing CSV at {local_csv}, skipping download.")

        # Set up video paths
        self.videos_dir = os.path.join(self.root_dir, self.video_path_base)
        expected_video_content_path = os.path.join(
            self.videos_dir, self.video_folder_name)

        # Download and extract videos
        if self.overwrite or not os.path.isdir(expected_video_content_path):
            if self.overwrite and os.path.isdir(expected_video_content_path):
                print(
                    f'Removing existing video directory: {expected_video_content_path}')
                shutil.rmtree(expected_video_content_path)

            # Ensure parent directories exist
            os.makedirs(self.videos_dir, exist_ok=True)

            zip_path = os.path.join(self.root_dir, zip_fname)

            # Download ZIP file
            print(f"Downloading {zip_fname} from {self.BUCKET}...")
            try:
                self._fetch_gcs(
                    source=f"{self.BUCKET}/{zip_fname}",
                    filepath=zip_path
                )
                print(f"Successfully downloaded ZIP to {zip_path}")

                # Verify ZIP file exists and has content
                if not os.path.isfile(zip_path):
                    raise FileNotFoundError(
                        f"ZIP file not found after download: {zip_path}")

                zip_size = os.path.getsize(zip_path)
                print(f"ZIP file size: {zip_size / (1024*1024):.2f} MB")

                if zip_size == 0:
                    raise ValueError(
                        f"Downloaded ZIP file is empty: {zip_path}")

            except Exception as e:
                print(f"Error downloading ZIP {zip_fname}: {e}")
                raise

            # Extract ZIP file
            print(f"Extracting {zip_path} to {self.videos_dir}...")
            try:
                self.extract(
                    filepath=zip_path,
                    extract_dir=self.videos_dir,
                    format="zip",
                    delete_archive=False
                )
                print(f"Successfully extracted videos to {self.videos_dir}")

                # Verify extraction worked
                if not os.path.isdir(expected_video_content_path):
                    # List contents to debug
                    contents = os.listdir(self.videos_dir) if os.path.isdir(
                        self.videos_dir) else []
                    print(f"Contents of {self.videos_dir}: {contents}")
                    raise FileNotFoundError(
                        f"Video content directory not found after extraction: {expected_video_content_path}"
                    )

                # Check if there are actually video files
                video_files = os.listdir(expected_video_content_path)
                print(f"Found {len(video_files)} items in video directory")
                if len(video_files) == 0:
                    print("WARNING: No video files found in extracted directory")

            except Exception as e:
                print(f"Error extracting ZIP {zip_fname}: {e}")
                raise

            self.video_path = expected_video_content_path

        else:
            self.video_path = expected_video_content_path
            print(
                f"Found existing video directory at {self.video_path}, skipping download.")

    def _prepare_videos(self):
        """Load label mapping from CSV and build the sorted list of video paths."""
        self._download_ssv2_data()

        csv_path = os.path.join(
            self.root_dir,
            "train_classes_map.csv" if self.train else "test_classes_map.csv"
        )

        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        print(f"Loading labels from {csv_path}...")

        with open(csv_path, newline="") as fin:
            reader = csv.DictReader(fin)
            for row in reader:
                full_path = row["filename"]
                orig_cls = int(row["class"])

                # Check if the original class is in our mapping
                if orig_cls not in self.CLASS_TO_IDX:
                    print(
                        f"WARNING: Class {orig_cls} not found in CLASS_TO_IDX mapping")
                    continue

                video_file_path = os.path.join(self.video_path, full_path)

                # Check if video file actually exists
                if not os.path.isfile(video_file_path):
                    print(f"WARNING: Video file not found: {video_file_path}")
                    continue

                self.stimulus_data.append(video_file_path)
                # Remap to dense index
                self.labels.append(self.CLASS_TO_IDX[orig_cls])

        print(f"Loaded {len(self.stimulus_data)} video samples")

        if len(self.stimulus_data) == 0:
            raise ValueError("No valid video samples found!")

    def __len__(self) -> int:
        return len(self.stimulus_data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        video_path = self.stimulus_data[idx]
        frames = self._load_data(video_path)
        if self.randaugment_transform and self.train:
            frames = randaugment_threaded(
                frames, self.randaugment_transform, num_workers=8)
        label = self.labels[idx]
        return self.preprocess(frames), label

    def _load_data(self, video_path: str) -> List[Image.Image]:
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        try:
            container = av.open(video_path)
            stream = container.streams.video[0]
            orig_fps = float(stream.average_rate)
            duration_s = (container.duration or 0) / 1e6
            total_frames = int(orig_fps * duration_s)
            interval = max(int(round(orig_fps / self.target_fps)), 1)
            idxs = list(range(0, total_frames, interval))
            max_count = int(self.target_fps * 4)
            if len(idxs) > max_count:
                idxs = idxs[:max_count]

            frames: List[Image.Image] = []
            frame_idx = 0
            next_ptr = 0
            for packet in container.demux(stream):
                for frm in packet.decode():
                    if next_ptr < len(idxs) and frame_idx == idxs[next_ptr]:
                        frames.append(frm.to_image())
                        next_ptr += 1
                        if next_ptr >= len(idxs):
                            break
                    frame_idx += 1
                if next_ptr >= len(idxs):
                    break

            container.close()

            if frames and len(frames) < max_count:
                frames += [frames[-1]] * (max_count - len(frames))

            return frames

        except Exception as e:
            print(f"Error loading video {video_path}: {e}")
            raise


class SSV2PrunedStimulusTrainSet(SSV2Pruned):
    def __init__(
        self,
        root_dir: Optional[str] = None,
        overwrite: bool = False,
        preprocess: Optional[Callable] = None
    ):
        if root_dir is None:
            root_dir = os.path.join(get_data_home(), SSV2Pruned.__name__)
        super().__init__(root_dir=root_dir, overwrite=overwrite,
                         preprocess=preprocess, train=True)


class AugmentedSSV2PrunedStimulusTrainSet(SSV2Pruned):
    def __init__(
        self,
        root_dir: Optional[str] = None,
        overwrite: bool = False,
        preprocess: Optional[Callable] = None
    ):
        if root_dir is None:
            root_dir = os.path.join(get_data_home(), SSV2Pruned.__name__)
        super().__init__(root_dir=root_dir, overwrite=overwrite, preprocess=preprocess, train=True,
                         apply_randaugment=True)


class SSV2PrunedStimulusTestSet(SSV2Pruned):
    def __init__(
        self,
        root_dir: Optional[str] = None,
        overwrite: bool = False,
        preprocess: Optional[Callable] = None
    ):
        if root_dir is None:
            root_dir = os.path.join(get_data_home(), SSV2Pruned.__name__)
        super().__init__(root_dir=root_dir, overwrite=overwrite,
                         preprocess=preprocess, train=False)
