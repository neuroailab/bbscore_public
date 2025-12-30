from data.augmentations import rand_augment_transform, _FILL, randaugment_parallel, randaugment_threaded
import shutil
from data.base import BaseDataset
from sklearn.datasets import get_data_home
from PIL import Image
import os
import csv
from typing import Optional, Callable, Tuple, List, Union

import cv2
import torch
import xarray as xr
import decord
decord.bridge.set_bridge('torch')


class PhysionPrediction(BaseDataset):

    BUCKET = "gs://bbscore_datasets/PhysionPrediction"

    def __init__(
        self,
        root_dir: Optional[str] = None,
        overwrite: bool = False,
        preprocess: Optional[Callable] = None,
        apply_randaugment: bool = False,
        randaugment_config_str: str = 'rand-m7-n2-mstd0.3',
        train: bool = True,
        placement: bool = False,
        intrageneralization: bool = False
    ):
        super().__init__(root_dir)
        self.overwrite = overwrite
        self.preprocess = preprocess
        self.train = train
        self.placement = placement
        self.intrageneralization = intrageneralization
        self.apply_randaugment = apply_randaugment
        self.randaugment_transform = None

        if self.apply_randaugment and self.train:
            hparams_vid = {"img_mean": _FILL,
                           "translate_const": 50, "magnitude_std": 0.3}
            self.randaugment_transform = rand_augment_transform(
                randaugment_config_str, hparams_vid)

        self.stimulus_data: List[str] = []
        self.labels: List[float] = []
        self.behavioral_targets: List[float] = []  # Only for test set
        self.target_fps = 25.0
        self.video_path_base = "videos"
        self.video_folder_name = "stimulus_PhysionGlobalPrediction2024"
        self.video_path = os.path.join(
            self.root_dir, self.video_path_base, self.video_folder_name)

        print(
            f"PhysionPrediction '{self.split_name()}' initialized. RandAugment: {self.apply_randaugment and self.train}.")
        self._prepare_videos()

    def split_name(self):  # Helper for print
        s = "train" if self.train else "test"
        if self.intrageneralization:
            s += "_intra"
        if self.placement:
            s += "_placement"
        else:
            s += "_contact"
        return s

    def _download_physion_data(self):  # Ensure shutil is imported
        os.makedirs(self.root_dir, exist_ok=True)
        csv_fname = "stimulus_PhysionGlobalPrediction2024.csv"
        zip_fname = "stimulus_PhysionGlobalPrediction2024.zip"

        behavioral_data_fname_key = "PhysionHumanContactPrediction2024.nc" if not self.placement else "PhysionHumanPlacementPrediction2024.nc"
        # For Physion, behavioral data is only on the test set.
        if not self.train:
            local_behavioral_nc = os.path.join(
                self.root_dir, behavioral_data_fname_key)
            self.local_behavioral_nc = local_behavioral_nc
            if self.overwrite or not os.path.isfile(local_behavioral_nc):
                print(f"Fetching behavioral data to {local_behavioral_nc}...")
                self.fetch_and_extract(
                    source=f"{self.BUCKET}/{behavioral_data_fname_key}",
                    target_dir=self.root_dir,
                    filename=behavioral_data_fname_key,
                    extract=False,
                    method="gcs",
                    force_download=self.overwrite,
                )
            else:
                print(
                    f"Found existing human data at {local_behavioral_nc}, skipping download.")

        local_csv = os.path.join(self.root_dir, csv_fname)
        if self.overwrite or not os.path.isfile(local_csv):
            print(f"Fetching CSV to {local_csv}...")
            self.fetch_and_extract(
                source=f"{self.BUCKET}/{csv_fname}", target_dir=self.root_dir, filename=csv_fname,
                extract=False, method="gcs", force_download=self.overwrite)
        else:
            print(f"Found existing CSV at {local_csv}, skipping download.")

        self.videos_dir = os.path.join(self.root_dir, self.video_path_base)
        expected_video_content_path = os.path.join(
            self.videos_dir, self.video_folder_name)

        if self.overwrite or not os.path.isdir(expected_video_content_path):
            if self.overwrite and os.path.isdir(self.videos_dir):
                shutil.rmtree(self.videos_dir)
            os.makedirs(self.videos_dir, exist_ok=True)

            zip_path = os.path.join(self.root_dir, zip_fname)
            # Fetch GCS needs filepath for download destination of zip
            self._fetch_gcs(
                source=f"{self.BUCKET}/{zip_fname}", filepath=zip_path)
            print(f"Fetched ZIP to {zip_path}")

            # Extract into self.videos_dir. The zip should contain the self.video_folder_name directory.
            self.extract(filepath=zip_path, extract_dir=self.videos_dir,
                         format="zip", delete_archive=True)  # Delete zip after extraction
            self.video_path = expected_video_content_path  # This should now exist
            print(f"Extracted videos into {self.video_path}")
            if not os.path.isdir(self.video_path):
                raise FileNotFoundError(
                    f"Video content directory not found after extraction: {self.video_path}")
        else:
            self.video_path = expected_video_content_path
            print(
                f"Found existing video directory at {self.video_path}, skipping download.")

    def __len__(self):
        return len(self.stimulus_data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Union[float, Tuple[str, float, float]], int]:
        video_file_path_relative = self.stimulus_data[idx]
        video_full_path = os.path.join(
            self.video_path, video_file_path_relative)

        pil_frames = self._load_data(
            video_full_path)  # Returns List[PIL.Image]

        if self.randaugment_transform and self.train:
            pil_frames = randaugment_threaded(
                pil_frames, self.randaugment_transform, num_workers=8)

        if self.preprocess is None:
            raise ValueError(
                "preprocess is not set for PhysionPrediction dataset.")

        # preprocess should handle List[PIL.Image] -> Tensor (e.g. (C,T,H,W))
        preprocessed_video_tensor = self.preprocess(
            pil_frames)  # , self.target_fps)  # Pass list of PILs

        label = self.labels[idx]

        if not self.train:
            behavioral_target = self.behavioral_targets[idx]
            return preprocessed_video_tensor, (os.path.basename(video_file_path_relative), label, behavioral_target)
        else:
            return preprocessed_video_tensor, label

    def _prepare_videos(self):
        self._download_physion_data()
        if not self.train:
            ds = xr.open_dataset(
                self.local_behavioral_nc,
                engine="h5netcdf"
            )
            target_key = "choice" if self.placement else "responseBool"
            self.behavioral_data = ds[target_key].groupby(
                "stimulus_id").mean(dim="presentation")

        csv_path = os.path.join(
            self.root_dir, "stimulus_PhysionGlobalPrediction2024.csv")
        with open(csv_path, newline="") as fin:
            reader = csv.DictReader(fin)
            for row in reader:
                # This should be relative like "test/00001.mp4"
                relative_path = row["filename"]
                cls = row["contacts"] if self.placement else row["label"]
                if float(cls) == -1.0 and self.placement:
                    continue

                stim_id_base = row["stimulus_id"][:-8]  # remove _sim_XXX

                # Filter logic
                passes_filter = False
                if self.train:
                    if self.intrageneralization:
                        if int(row["train"]) == 1 and int(row["intra_generalizability"]) == 1:
                            passes_filter = True
                    else:
                        if int(row["train"]) == 1:
                            passes_filter = True
                else:  # test
                    if not stim_id_base in self.behavioral_data['stimulus_id'].values:
                        continue  # Skip if no behavioral data
                    if self.intrageneralization:
                        if int(row["train"]) == 0 and int(row["intra_generalizability"]) == 1:
                            passes_filter = True
                    else:
                        if int(row["train"]) == 0:
                            passes_filter = True

                if passes_filter:
                    self.stimulus_data.append(
                        relative_path)  # Store relative path
                    self.labels.append(float(cls))
                    if not self.train:
                        self.behavioral_targets.append(
                            self.behavioral_data.sel(stimulus_id=stim_id_base).item())

    def _load_data(self, video_path: str) -> List[Image.Image]:
        # , num_threads=8)
        decord.bridge.set_bridge('torch')
        vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
        orig_fps = vr.get_avg_fps()
        interval = max(int(round(orig_fps / self.target_fps)), 1)
        # build the list of frame indices you want
        idxs = list(range(0, len(vr), interval))
        # quickly pull them all as a tensor [N,H,W,3]
        frames = vr.get_batch(idxs)
        # convert to PIL if you must, or leave as a tensor for your preprocess
        if frames.shape[0] > int(self.target_fps * 0.45):
            frames = frames[:int(self.target_fps * 0.45)]
        frames = [Image.fromarray(frame.numpy()) for frame in frames]
        while len(frames) < int(self.target_fps * 0.45):
            frames += [frames[-1]]
        return frames


class PhysionContactPredictionTrain(PhysionPrediction):
    def __init__(self,
                 root_dir: Optional[str] = None,
                 overwrite: bool = False,
                 preprocess: Optional[Callable] = None,
                 ):
        root = os.path.join(get_data_home(), PhysionPrediction.__name__)
        super().__init__(root_dir=root, overwrite=overwrite, preprocess=preprocess,
                         train=True, placement=False, intrageneralization=False)


class PhysionContactPredictionAugmentedTrain(PhysionPrediction):
    def __init__(self,
                 root_dir: Optional[str] = None,
                 overwrite: bool = False,
                 preprocess: Optional[Callable] = None,
                 ):
        root = os.path.join(get_data_home(), PhysionPrediction.__name__)
        super().__init__(root_dir=root, overwrite=overwrite, preprocess=preprocess,
                         apply_randaugment=True, randaugment_config_str='rand-m7-n2-mstd0.3',
                         train=True, placement=False, intrageneralization=False)


class PhysionContactPredictionTest(PhysionPrediction):
    """Dataset class for the testing set of the Physion Contact Prediction Stimulus."""

    def __init__(
        self,
        root_dir: Optional[str] = None,
        overwrite: bool = False,
        preprocess: Optional[Callable] = None,
    ):
        root = os.path.join(
            get_data_home(), PhysionPrediction.__name__
        )
        super().__init__(root_dir=root, overwrite=overwrite, preprocess=preprocess,
                         train=False, placement=False, intrageneralization=False)


class PhysionIntraContactPredictionTrain(PhysionPrediction):
    """Dataset class for the training set of the Physion Contact Prediction Stimulus for the Intra-configuration generalization Task."""

    def __init__(
        self,
        root_dir: Optional[str] = None,
        overwrite: bool = False,
        preprocess: Optional[Callable] = None,
    ):
        root = os.path.join(
            get_data_home(), PhysionPrediction.__name__
        )
        super().__init__(root_dir=root, overwrite=overwrite, preprocess=preprocess,
                         train=True, intrageneralization=True)


class PhysionIntraContactPredictionAugmentedTrain(PhysionPrediction):
    """Dataset class for the training set of the Physion Contact Prediction Stimulus for the Intra-configuration generalization Task."""

    def __init__(
        self,
        root_dir: Optional[str] = None,
        overwrite: bool = False,
        preprocess: Optional[Callable] = None,
    ):
        root = os.path.join(
            get_data_home(), PhysionPrediction.__name__
        )
        super().__init__(root_dir=root, overwrite=overwrite, preprocess=preprocess,
                         apply_randaugment=True, randaugment_config_str='rand-m7-n2-mstd0.3',
                         train=True, intrageneralization=True)


class PhysionIntraContactPredictionTest(PhysionPrediction):
    """Dataset class for the testing set of the Physion Contact Prediction Stimulus for the Intra-configuration generalization Task."""

    def __init__(
        self,
        root_dir: Optional[str] = None,
        overwrite: bool = False,
        preprocess: Optional[Callable] = None,
    ):
        # Initialize with train=True to load the training videos
        root = os.path.join(
            get_data_home(), PhysionPrediction.__name__
        )
        super().__init__(root_dir=root, overwrite=overwrite,
                         preprocess=preprocess, train=False, intrageneralization=True)


class PhysionPlacementPredictionTrain(PhysionPrediction):
    """Dataset class for the training set of the Physion Placement Prediction Stimulus."""

    def __init__(
        self,
        root_dir: Optional[str] = None,
        overwrite: bool = False,
        preprocess: Optional[Callable] = None,
    ):
        root = os.path.join(
            get_data_home(), PhysionPrediction.__name__
        )
        super().__init__(root_dir=root, overwrite=overwrite,
                         preprocess=preprocess, train=True, placement=True)


class PhysionPlacementPredictionAugmentedTrain(PhysionPrediction):
    """Dataset class for the training set of the Physion Placement Prediction Stimulus."""

    def __init__(
        self,
        root_dir: Optional[str] = None,
        overwrite: bool = False,
        preprocess: Optional[Callable] = None,
    ):
        root = os.path.join(
            get_data_home(), PhysionPrediction.__name__
        )
        super().__init__(root_dir=root, overwrite=overwrite,
                         apply_randaugment=True, randaugment_config_str='rand-m7-n2-mstd0.3',
                         preprocess=preprocess, train=True, placement=True)


class PhysionPlacementPredictionTest(PhysionPrediction):
    """Dataset class for the testing set of the Physion Placement Prediction Stimulus."""

    def __init__(
        self,
        root_dir: Optional[str] = None,
        overwrite: bool = False,
        preprocess: Optional[Callable] = None
    ):
        root = os.path.join(
            get_data_home(), PhysionPrediction.__name__
        )
        super().__init__(root_dir=root, overwrite=overwrite,
                         preprocess=preprocess, train=False, placement=True)


class PhysionIntraPlacementPredictionTrain(PhysionPrediction):
    """Dataset class for the training set of the Physion Placement Prediction Stimulus for the Intra-configuration generalization Task."""

    def __init__(
        self,
        root_dir: Optional[str] = None,
        overwrite: bool = False,
        preprocess: Optional[Callable] = None,
    ):
        root = os.path.join(
            get_data_home(), PhysionPrediction.__name__
        )
        super().__init__(root_dir=root, overwrite=overwrite, preprocess=preprocess,
                         train=True, intrageneralization=True, placement=True)


class PhysionIntraPlacementPredictionAugmentedTrain(PhysionPrediction):
    """Dataset class for the training set of the Physion Placement Prediction Stimulus for the Intra-configuration generalization Task."""

    def __init__(
        self,
        root_dir: Optional[str] = None,
        overwrite: bool = False,
        preprocess: Optional[Callable] = None,
    ):
        root = os.path.join(
            get_data_home(), PhysionPrediction.__name__
        )
        super().__init__(root_dir=root, overwrite=overwrite, preprocess=preprocess,
                         apply_randaugment=True, randaugment_config_str='rand-m7-n2-mstd0.3',
                         train=True, intrageneralization=True, placement=True)


class PhysionIntraPlacementPredictionTest(PhysionPrediction):
    """Dataset class for the testing set of the Physion Placement Prediction Stimulus for the Intra-configuration generalization Task."""

    def __init__(
        self,
        root_dir: Optional[str] = None,
        overwrite: bool = False,
        preprocess: Optional[Callable] = None
    ):
        root = os.path.join(
            get_data_home(), PhysionPrediction.__name__
        )
        super().__init__(root_dir=root, overwrite=overwrite, preprocess=preprocess,
                         train=False, intrageneralization=True, placement=True)
