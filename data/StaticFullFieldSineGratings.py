import os
import torch
import glob
import numpy as np

from PIL import Image
from typing import Optional, Callable
from sklearn.datasets import get_data_home

from data.base import BaseDataset


class StaticFullFieldSineGratingsStimulusSet(BaseDataset):
    """
    A dataset of static full-field sine gratings in various orientations,
    phases, spatial frequencies, and color.
    """

    def __init__(
        self,
        preprocess: Optional[Callable] = None,
        overwrite: bool = False,
    ):
        root_dir = os.path.join(
            get_data_home(), "StaticFullFieldSineGratingsStimulusSet")
        super().__init__(root_dir)
        self.overwrite = overwrite
        self.preprocess = preprocess
        self.image_paths = self._load_image_paths()

    def _check_files_exists(self, *paths):
        return all(os.path.exists(path) for path in paths)

    def _load_image_paths(self):
        stimulus_path = os.path.join(self.root_dir, 'images.zip')

        if not self._check_files_exists(stimulus_path) \
           or self.overwrite:
            direc = "gs://bbscore_datasets/StaticFullFieldSineGratingsStimulusSet"
            gcs_path = f"{direc}/{os.path.basename(stimulus_path)}"

            download_path = self.fetch(
                source=gcs_path,
                force_download=self.overwrite,
            )

            self.extract(
                filepath=download_path,
                extract_dir=self.root_dir,
                format="zip",
                delete_archive=False,
            )

        assert self._check_files_exists(stimulus_path)

        return sorted(glob.glob(f"{self.root_dir}/images/*.jpg"))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        if self.preprocess:
            image = self.preprocess(image)

        # extract label
        parts = image_path.split("/")[-1].split("_")
        angle = float(parts[1][:-3])
        sf = float(parts[2][:-2])
        phase = float(parts[3][:-5])
        color_string = parts[4].split(".jpg")[0]
        color = 0.0 if color_string == "bw" else 1.0

        label = np.array([angle, sf, phase, color])

        return image, label
