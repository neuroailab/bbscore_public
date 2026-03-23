import os
import torch
import numpy as np
import h5py

from PIL import Image
from typing import Optional, Callable, Union, List
from sklearn.datasets import get_data_home

from data.base import BaseDataset
from data.utils import split_half_consistency, one_vs_all_consistency


class TVSDStimulusSet(BaseDataset):
    """
    THINGS Ventral Spiking Dataset
    """

    def __init__(
        self,
        root_dir: Optional[str] = None,
        preprocess: Optional[Callable] = None,
        overwrite: bool = False,
        train: bool = True,
        region: str = None,
    ):
        super().__init__(root_dir)
        self.overwrite = overwrite
        self.preprocess = preprocess
        self.train = train
        self.image_paths = self._load_image_paths()
        self.assembly, self.neural_data, self.ceiling = None, None, None
        if region is not None:
            self.assembly = TVSDAssembly(root_dir=root_dir, region=region,
                                         timebin_length='default')
            self.neural_data, self.ceiling = self.assembly.get_assembly(train)

    def _check_files_exists(self, *paths):
        return all(os.path.exists(path) for path in paths)

    def _get_image_paths_for_monkey(self, metadata):
        path = 'train_imgs' if self.train else 'test_imgs'
        refs = metadata[path]['things_path'][:].flatten()
        decoded_strings = []
        for ref in refs:
            arr = metadata[path][ref][()]
            arr = arr.flatten()
            s = ''.join(map(chr, arr))
            decoded_strings.append(
                os.path.join(self.root_dir, "images", s.replace('\\', '/'))
            )
        return decoded_strings

    def _load_image_paths(self):
        stimulus_path = os.path.join(self.root_dir, 'images.zip')
        metadata_path_F = os.path.join(
            self.root_dir, 'things_imgs_monkeyF.mat'
        )
        metadata_path_N = os.path.join(
            self.root_dir, 'things_imgs_monkeyN.mat'
        )

        if not self._check_files_exists(stimulus_path, metadata_path_F,
                                        metadata_path_N) \
           or self.overwrite:
            for path in [stimulus_path, metadata_path_F, metadata_path_N]:
                direc = "gs://bbscore_datasets/TVSDStimulusSet"
                gcs_path = f"{direc}/{os.path.basename(path)}"

                download_path = self.fetch(
                    source=gcs_path,
                    force_download=self.overwrite,
                )

                if path == stimulus_path:
                    self.extract(
                        filepath=download_path,
                        extract_dir=self.root_dir,
                        format="zip",
                        delete_archive=False,
                    )

        assert self._check_files_exists(stimulus_path, metadata_path_F,
                                        metadata_path_N)

        metadataF = h5py.File(metadata_path_F, 'r')
        return self._get_image_paths_for_monkey(metadataF)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        if self.preprocess:
            image = self.preprocess(image)
        if self.neural_data is not None:
            return image, self.neural_data[idx]
        return image


class TVSDStimulusTrainSet(TVSDStimulusSet):
    """Dataset class for the training set of the BMD Stimulus."""

    def __init__(
        self,
        root_dir: Optional[str] = None,
        overwrite: bool = False,
        preprocess: Optional[Callable] = None
    ):
        # Initialize with train=True to load the training videos
        root = os.path.join(
            get_data_home(), TVSDStimulusSet.__name__
        )
        super().__init__(root_dir=root, overwrite=overwrite,
                         preprocess=preprocess, train=True)


class TVSDStimulusTestSet(TVSDStimulusSet):
    """Dataset class for the testing set of the BMD Stimulus."""

    def __init__(
        self,
        root_dir: Optional[str] = None,
        overwrite: bool = False,
        preprocess: Optional[Callable] = None
    ):
        # Initialize with train=False to load the test videos
        root = os.path.join(
            get_data_home(), TVSDStimulusSet.__name__
        )
        super().__init__(root_dir=root, overwrite=overwrite,
                         preprocess=preprocess, train=False)


class TVSDFullStimulusTrainSet(TVSDStimulusSet):
    """Dataset class for the training set of the BMD Stimulus."""

    def __init__(
        self,
        root_dir: Optional[str] = None,
        overwrite: bool = False,
        preprocess: Optional[Callable] = None
    ):
        # Initialize with train=True to load the training videos
        root = os.path.join(
            get_data_home(), TVSDStimulusSet.__name__
        )
        super().__init__(root_dir=root, overwrite=overwrite,
                         preprocess=preprocess, train=True, region='Full')


class TVSDFullStimulusTestSet(TVSDStimulusSet):
    """Dataset class for the training set of the BMD Stimulus."""

    def __init__(
        self,
        root_dir: Optional[str] = None,
        overwrite: bool = False,
        preprocess: Optional[Callable] = None
    ):
        # Initialize with train=True to load the training videos
        root = os.path.join(
            get_data_home(), TVSDStimulusSet.__name__
        )
        super().__init__(root_dir=root, overwrite=overwrite,
                         preprocess=preprocess, train=False, region='Full')


class TVSDV1StimulusTrainSet(TVSDStimulusSet):
    """Dataset class for the training set of the BMD Stimulus."""

    def __init__(
        self,
        root_dir: Optional[str] = None,
        overwrite: bool = False,
        preprocess: Optional[Callable] = None
    ):
        # Initialize with train=True to load the training videos
        root = os.path.join(
            get_data_home(), TVSDStimulusSet.__name__
        )
        super().__init__(root_dir=root, overwrite=overwrite,
                         preprocess=preprocess, train=True, region='V1')


class TVSDV1StimulusTestSet(TVSDStimulusSet):
    """Dataset class for the training set of the BMD Stimulus."""

    def __init__(
        self,
        root_dir: Optional[str] = None,
        overwrite: bool = False,
        preprocess: Optional[Callable] = None
    ):
        # Initialize with train=True to load the training videos
        root = os.path.join(
            get_data_home(), TVSDStimulusSet.__name__
        )
        super().__init__(root_dir=root, overwrite=overwrite,
                         preprocess=preprocess, train=False, region='V1')


class TVSDV4StimulusTrainSet(TVSDStimulusSet):
    """Dataset class for the training set of the BMD Stimulus."""

    def __init__(
        self,
        root_dir: Optional[str] = None,
        overwrite: bool = False,
        preprocess: Optional[Callable] = None
    ):
        # Initialize with train=True to load the training videos
        root = os.path.join(
            get_data_home(), TVSDStimulusSet.__name__
        )
        super().__init__(root_dir=root, overwrite=overwrite,
                         preprocess=preprocess, train=True, region='V4')


class TVSDV4StimulusTestSet(TVSDStimulusSet):
    """Dataset class for the training set of the BMD Stimulus."""

    def __init__(
        self,
        root_dir: Optional[str] = None,
        overwrite: bool = False,
        preprocess: Optional[Callable] = None
    ):
        # Initialize with train=True to load the training videos
        root = os.path.join(
            get_data_home(), TVSDStimulusSet.__name__
        )
        super().__init__(root_dir=root, overwrite=overwrite,
                         preprocess=preprocess, train=False, region='V4')


class TVSDITStimulusTrainSet(TVSDStimulusSet):
    """Dataset class for the training set of the BMD Stimulus."""

    def __init__(
        self,
        root_dir: Optional[str] = None,
        overwrite: bool = False,
        preprocess: Optional[Callable] = None
    ):
        # Initialize with train=True to load the training videos
        root = os.path.join(
            get_data_home(), TVSDStimulusSet.__name__
        )
        super().__init__(root_dir=root, overwrite=overwrite,
                         preprocess=preprocess, train=True, region='IT')


class TVSDITStimulusTestSet(TVSDStimulusSet):
    """Dataset class for the training set of the BMD Stimulus."""

    def __init__(
        self,
        root_dir: Optional[str] = None,
        overwrite: bool = False,
        preprocess: Optional[Callable] = None
    ):
        # Initialize with train=True to load the training videos
        root = os.path.join(
            get_data_home(), TVSDStimulusSet.__name__
        )
        super().__init__(root_dir=root, overwrite=overwrite,
                         preprocess=preprocess, train=False, region='IT')


class TVSDAssembly(BaseDataset):
    def __init__(
        self,
        root_dir: Optional[str] = None,
        region: str = 'V1',
        overwrite: bool = False,
        timebin_length: Union[str, int] = 'default',
        mapping_procedure: str = 'split_half',
        # MODIFIED: Added monkey argument
        monkey: Optional[Union[str, List[str]]] = None,
        repetition: bool = False
    ):
        super().__init__(root_dir)
        self.region = region
        self.overwrite = overwrite
        self.timebin_length = timebin_length
        self.mapping_procedure = mapping_procedure
        self.train_data, self.train_ceiling = None, None
        self.test_data, self.test_ceiling = None, None
        self.repetition = repetition
        # leaving threshold very low to not break flow of experiments
        self.ceiling_threshold = -1000.0

        # --- MODIFIED: Process the monkey argument ---
        if monkey is None:
            # Default to using both monkeys if none are specified
            self.monkeys = ['monkeyF', 'monkeyN']
        elif isinstance(monkey, str):
            if monkey not in ['monkeyF', 'monkeyN']:
                raise ValueError(
                    "Invalid monkey identifier. Must be 'monkeyF' or 'monkeyN'.")
            self.monkeys = [monkey]
        elif isinstance(monkey, list):
            if not all(m in ['monkeyF', 'monkeyN'] for m in monkey):
                raise ValueError(
                    "Invalid monkey identifier in list. Must be one of 'monkeyF' or 'monkeyN'.")
            self.monkeys = monkey
        else:
            raise TypeError(
                "'monkey' must be a string, a list of strings, or None.")
        # --- END MODIFICATION ---

    def _check_files_exists(self, *paths):
        return all(os.path.exists(path) for path in paths)

    def _get_channels_for_region(self, monkey):
        regions_to_channels = {
            'monkeyN': {
                'V1': (0, 512),
                'V4': (512, 768),
                'IT': (768, 1024),
                'Full': (0, 1024),
            },
            'monkeyF': {
                'V1': (0, 512),
                'IT': (512, 832),
                'V4': (832, 1024),
                'Full': (0, 1024)
            }
        }
        return regions_to_channels[monkey][self.region]

    def _get_timebin_monkey_data(self, timebin, monkey, assembly_path, train):
        if isinstance(timebin, int):
            path = os.path.join(assembly_path, str(timebin), monkey,
                                'THINGS_normMUA.mat')
        else:
            path = os.path.join(assembly_path, monkey, 'THINGS_normMUA.mat')
        start, end = self._get_channels_for_region(monkey)
        if not self.repetition:
            col = 'train_MUA' if train else 'test_MUA'
            data = h5py.File(path, 'r')[col][:, start:end]
        else:
            if train:
                data = h5py.File(path, 'r')['train_MUA'][:, start:end]
            else:
                data = h5py.File(path, 'r')['test_MUA_reps'][:, :, start:end]
                data = np.moveaxis(data, 0, 1)  # reps should be axis=1
        ceiling = h5py.File(path, 'r')['test_MUA_reps'][:, :, start:end]
        return data, ceiling

    def prepare_data(self, train: bool = True):
        localdir = 'defaultBins' if self.timebin_length == "default" \
            else f"{self.timebin_length}msBins"

        assembly_path = os.path.join(self.root_dir, localdir)

        compute_ceiling = split_half_consistency \
            if self.mapping_procedure == 'split_half' \
            else one_vs_all_consistency

        if not self._check_files_exists(assembly_path) \
           or self.overwrite:
            direc = "gs://bbscore_datasets/TVSDAssembly"
            gcs_path = f"{direc}/{os.path.basename(assembly_path)}.zip"
            self.fetch_and_extract(
                source=gcs_path,
                force_download=self.overwrite,
                target_dir=self.root_dir,
                format="zip",
                delete_archive=False,
            )

        assert self._check_files_exists(assembly_path)

        if self.timebin_length == 'default':
            timebins = ['']
        else:
            timebins = range(25, 200, self.timebin_length)

        for train in [True, False]:
            data, ceiling = [], []
            for timebin in timebins:
                timebin_data, timebin_ceiling = [], []
                # --- MODIFIED: Iterate over the specified monkeys ---
                for monkey in self.monkeys:
                    # --- END MODIFICATION ---
                    tmd, tmc = self._get_timebin_monkey_data(timebin,
                                                             monkey,
                                                             assembly_path,
                                                             train)
                    timebin_data.append(
                        np.expand_dims(tmd, axis=0)
                    )
                    timebin_ceiling.append(
                        np.expand_dims(compute_ceiling(tmc), axis=0)
                    )
                data.append(np.concatenate(timebin_data, axis=-1))
                ceiling.append(np.concatenate(timebin_ceiling, axis=-1))

            # concat across timebins & squeeze singleton dims
            data_arr = np.concatenate(data, axis=0).squeeze()
            ceiling_arr = np.concatenate(ceiling, axis=0).squeeze()

            # --- apply ceiling threshold ---
            if not train:
                mask = ceiling_arr >= self.ceiling_threshold
            # assign to train or test
            if train:
                self.train_data = data_arr
                self.train_ceiling = ceiling_arr
            else:
                self.test_data = data_arr
                self.test_ceiling = ceiling_arr

        if self.repetition:
            self.train_data = self.train_data[:, None, ...]
            self.train_data, self.train_ceiling = self.train_data[:,
                                                                  :, mask], self.train_ceiling[mask]
            self.test_data, self.test_ceiling = self.test_data[:,
                                                               :, mask], self.test_ceiling[mask]
        else:
            if self.train_data.ndim != 3:
                self.train_data, self.train_ceiling = self.train_data[:,
                                                                      mask], self.train_ceiling[mask]
                self.test_data, self.test_ceiling = self.test_data[:,
                                                                   mask], self.test_ceiling[mask]
            if self.train_data.ndim == 3:
                self.train_data = self.train_data.transpose(1, 2, 0)
                self.test_data = self.test_data.transpose(1, 2, 0)

    def get_assembly(self, train: bool = True):
        if self.train_data is None or self.train_ceiling is None \
           or self.test_data is None or self.test_ceiling is None:
            self.prepare_data(train)
        if train:
            if self.train_ceiling.ndim == 2:
                self.train_ceiling = self.train_ceiling.T
            return self.train_data, self.train_ceiling
        if self.test_ceiling.ndim == 2:
            self.test_ceiling = self.test_ceiling.T
        return self.test_data, self.test_ceiling

    def __len__(self):
        return self.data.shape[1]

    def __getitem__(self, idx):
        raise NotImplementedError("Indexing not supported.")

# The subclasses below will automatically pass the `monkey` argument
# to the parent TVSDAssembly class thanks to `**kwargs`.


class TVSDAssemblyFull(TVSDAssembly):
    def __init__(self, **kwargs):
        root = os.path.join(
            get_data_home(), TVSDAssembly.__name__
        )
        super().__init__(root_dir=root, region='Full',
                         timebin_length='default',
                         **kwargs)


class TVSDAssemblyFull10msBins(TVSDAssembly):
    def __init__(self, **kwargs):
        root = os.path.join(
            get_data_home(), TVSDAssembly.__name__
        )
        super().__init__(root_dir=root, region='Full',
                         timebin_length='default',
                         **kwargs)


class TVSDAssemblyV1(TVSDAssembly):
    def __init__(self, **kwargs):
        root = os.path.join(
            get_data_home(), TVSDAssembly.__name__
        )
        super().__init__(root_dir=root, region='V1',
                         timebin_length='default',
                         **kwargs)


class TVSDAssemblyV4(TVSDAssembly):
    def __init__(self, **kwargs):
        root = os.path.join(
            get_data_home(), TVSDAssembly.__name__
        )
        super().__init__(root_dir=root, region='V4',
                         timebin_length='default', **kwargs)


class TVSDAssemblyIT(TVSDAssembly):
    def __init__(self, **kwargs):
        root = os.path.join(
            get_data_home(), TVSDAssembly.__name__
        )
        super().__init__(root_dir=root, region='IT',
                         timebin_length='default', **kwargs)


class TVSDAssemblyV110msBins(TVSDAssembly):
    def __init__(self, **kwargs):
        root = os.path.join(
            get_data_home(), TVSDAssembly.__name__
        )
        super().__init__(root_dir=root, region='V1',
                         timebin_length=10, **kwargs)


class TVSDAssemblyV410msBins(TVSDAssembly):
    def __init__(self, **kwargs):
        root = os.path.join(
            get_data_home(), TVSDAssembly.__name__
        )
        super().__init__(root_dir=root, region='V4',
                         timebin_length=10, **kwargs)


class TVSDAssemblyIT10msBins(TVSDAssembly):
    def __init__(self, **kwargs):
        root = os.path.join(
            get_data_home(), TVSDAssembly.__name__
        )
        super().__init__(root_dir=root, region='IT',
                         timebin_length=10, **kwargs)


class TVSDAssemblyV1OneVsAll(TVSDAssembly):
    def __init__(self, **kwargs):
        root = os.path.join(
            get_data_home(), TVSDAssembly.__name__
        )
        super().__init__(root_dir=root, region='V1',
                         timebin_length='default', repetition=True,
                         mapping_procedure='one_vs_all', **kwargs)


class TVSDAssemblyV4OneVsAll(TVSDAssembly):
    def __init__(self, **kwargs):
        root = os.path.join(
            get_data_home(), TVSDAssembly.__name__
        )
        super().__init__(root_dir=root, region='V4',
                         timebin_length='default', repetition=True,
                         mapping_procedure='one_vs_all', **kwargs)


class TVSDAssemblyITOneVsAll(TVSDAssembly):
    def __init__(self, **kwargs):
        root = os.path.join(
            get_data_home(), TVSDAssembly.__name__
        )
        super().__init__(root_dir=root, region='IT',
                         timebin_length='default', repetition=True,
                         mapping_procedure='one_vs_all', **kwargs)


class TVSDAssemblyFullOneVsAll(TVSDAssembly):
    def __init__(self, **kwargs):
        root = os.path.join(
            get_data_home(), TVSDAssembly.__name__
        )
        super().__init__(root_dir=root, region='Full', repetition=True,
                         timebin_length='default', mapping_procedure='one_vs_all',
                         **kwargs)


class TVSDAssemblyFull10msBinsOneVsAll(TVSDAssembly):
    def __init__(self, **kwargs):
        root = os.path.join(
            get_data_home(), TVSDAssembly.__name__
        )
        super().__init__(root_dir=root, region='Full', repetition=True,
                         timebin_length='default', mapping_procedure='one_vs_all',
                         **kwargs)


class TVSDAssemblyV110msBinsOneVsAll(TVSDAssembly):
    def __init__(self, **kwargs):
        root = os.path.join(
            get_data_home(), TVSDAssembly.__name__
        )
        super().__init__(root_dir=root, region='V1', repetition=True,
                         timebin_length=10, mapping_procedure='one_vs_all',
                         **kwargs)


class TVSDAssemblyV410msBinsOneVsAll(TVSDAssembly):
    def __init__(self, **kwargs):
        root = os.path.join(
            get_data_home(), TVSDAssembly.__name__
        )
        super().__init__(root_dir=root, region='V4', repetition=True,
                         timebin_length=10, mapping_procedure='one_vs_all',
                         **kwargs)


class TVSDAssemblyIT10msBinsOneVsAll(TVSDAssembly):
    def __init__(self, **kwargs):
        root = os.path.join(
            get_data_home(), TVSDAssembly.__name__
        )
        super().__init__(root_dir=root, region='IT', repetition=True,
                         timebin_length=10, mapping_procedure='one_vs_all',
                         **kwargs)


class TVSDAssemblyMonkeyFV1(TVSDAssembly):
    def __init__(self, **kwargs):
        root = os.path.join(get_data_home(), TVSDAssembly.__name__)
        super().__init__(root_dir=root, region='V1', monkey='monkeyF', **kwargs)


class TVSDAssemblyMonkeyFV4(TVSDAssembly):
    def __init__(self, **kwargs):
        root = os.path.join(get_data_home(), TVSDAssembly.__name__)
        super().__init__(root_dir=root, region='V4', monkey='monkeyF', **kwargs)


class TVSDAssemblyMonkeyFIT(TVSDAssembly):
    def __init__(self, **kwargs):
        root = os.path.join(get_data_home(), TVSDAssembly.__name__)
        super().__init__(root_dir=root, region='IT', monkey='monkeyF', **kwargs)


class TVSDAssemblyMonkeyFFull(TVSDAssembly):
    def __init__(self, **kwargs):
        root = os.path.join(get_data_home(), TVSDAssembly.__name__)
        super().__init__(root_dir=root, region='Full', monkey='monkeyF', **kwargs)


class TVSDAssemblyMonkeyFV110msBins(TVSDAssembly):
    def __init__(self, **kwargs):
        root = os.path.join(get_data_home(), TVSDAssembly.__name__)
        super().__init__(root_dir=root, region='V1',
                         monkey='monkeyF', timebin_length=10, **kwargs)


class TVSDAssemblyMonkeyFV410msBins(TVSDAssembly):
    def __init__(self, **kwargs):
        root = os.path.join(get_data_home(), TVSDAssembly.__name__)
        super().__init__(root_dir=root, region='V4',
                         monkey='monkeyF', timebin_length=10, **kwargs)


class TVSDAssemblyMonkeyFIT10msBins(TVSDAssembly):
    def __init__(self, **kwargs):
        root = os.path.join(get_data_home(), TVSDAssembly.__name__)
        super().__init__(root_dir=root, region='IT',
                         monkey='monkeyF', timebin_length=10, **kwargs)


class TVSDAssemblyMonkeyFFull10msBins(TVSDAssembly):
    def __init__(self, **kwargs):
        root = os.path.join(get_data_home(), TVSDAssembly.__name__)
        super().__init__(root_dir=root, region='Full',
                         monkey='monkeyF', timebin_length=10, **kwargs)


class TVSDAssemblyMonkeyNV1(TVSDAssembly):
    def __init__(self, **kwargs):
        root = os.path.join(get_data_home(), TVSDAssembly.__name__)
        super().__init__(root_dir=root, region='V1', monkey='monkeyN', **kwargs)


class TVSDAssemblyMonkeyNV4(TVSDAssembly):
    def __init__(self, **kwargs):
        root = os.path.join(get_data_home(), TVSDAssembly.__name__)
        super().__init__(root_dir=root, region='V4', monkey='monkeyN', **kwargs)


class TVSDAssemblyMonkeyNIT(TVSDAssembly):
    def __init__(self, **kwargs):
        root = os.path.join(get_data_home(), TVSDAssembly.__name__)
        super().__init__(root_dir=root, region='IT', monkey='monkeyN', **kwargs)


class TVSDAssemblyMonkeyNFull(TVSDAssembly):
    def __init__(self, **kwargs):
        root = os.path.join(get_data_home(), TVSDAssembly.__name__)
        super().__init__(root_dir=root, region='Full', monkey='monkeyN', **kwargs)


class TVSDAssemblyMonkeyNV110msBins(TVSDAssembly):
    def __init__(self, **kwargs):
        root = os.path.join(get_data_home(), TVSDAssembly.__name__)
        super().__init__(root_dir=root, region='V1',
                         monkey='monkeyN', timebin_length=10, **kwargs)


class TVSDAssemblyMonkeyNV410msBins(TVSDAssembly):
    def __init__(self, **kwargs):
        root = os.path.join(get_data_home(), TVSDAssembly.__name__)
        super().__init__(root_dir=root, region='V4',
                         monkey='monkeyN', timebin_length=10, **kwargs)


class TVSDAssemblyMonkeyNIT10msBins(TVSDAssembly):
    def __init__(self, **kwargs):
        root = os.path.join(get_data_home(), TVSDAssembly.__name__)
        super().__init__(root_dir=root, region='IT',
                         monkey='monkeyN', timebin_length=10, **kwargs)


class TVSDAssemblyMonkeyNFull10msBins(TVSDAssembly):
    def __init__(self, **kwargs):
        root = os.path.join(get_data_home(), TVSDAssembly.__name__)
        super().__init__(root_dir=root, region='Full',
                         monkey='monkeyN', timebin_length=10, **kwargs)


class TVSDRepeatedStimulusTestSet(TVSDStimulusSet):
    """
    Dataset class for a test set where each image is repeated to form a
    video-like sequence. For each image, 18 data points are generated,
    with each data point being a repetition of the image 2*n+3 times, where
    n is the index of the data point (0 to 17).
    """

    def __init__(
        self,
        root_dir: Optional[str] = None,
        overwrite: bool = False,
        preprocess: Optional[Callable] = None,
    ):
        root = os.path.join(get_data_home(), TVSDStimulusSet.__name__)
        super().__init__(
            root_dir=root,
            overwrite=overwrite,
            preprocess=preprocess,
            train=False
        )
        self.num_repetitions = 18

    def __len__(self):
        return len(self.image_paths) * self.num_repetitions

    def __getitem__(self, idx: int) -> Union[torch.Tensor, tuple]:
        original_idx = idx // self.num_repetitions
        repetition_n = idx % self.num_repetitions
        num_frames = 2 * repetition_n + 3

        image_path = self.image_paths[original_idx]
        image = Image.open(image_path).convert("RGB")
        video = [image] * num_frames

        if self.preprocess:
            video = self.preprocess(video)

        if self.neural_data is not None:
            return video, self.neural_data[original_idx]
        return video


class TVSDRepeatedAssemblyBase(TVSDAssembly):
    """
    Base class for repeated assemblies.

    - Train split: uses default timebin ("defaultBins") -> usual 2D array (num_images, num_neurons).
    - Test split: uses 10 ms timebins ("10msBins"), then flattens
      (num_images, num_timebins, num_neurons) -> (num_images * num_timebins, num_neurons).
    """

    def __init__(self, **kwargs):
        super().__init__(timebin_length='default', **kwargs)  # train uses default

        # This will get updated after loading test data
        self.num_repetitions = 18

    def prepare_data(self, train: bool = True):
        # mapping_procedure logic as in parent
        compute_ceiling = (
            split_half_consistency
            if self.mapping_procedure == 'split_half'
            else one_vs_all_consistency
        )

        def ensure_dir(localdir: str) -> str:
            assembly_path = os.path.join(self.root_dir, localdir)
            if not self._check_files_exists(assembly_path) or self.overwrite:
                direc = "gs://bbscore_datasets/TVSDAssembly"
                gcs_path = f"{direc}/{localdir}.zip"
                self.fetch_and_extract(
                    source=gcs_path,
                    force_download=self.overwrite,
                    target_dir=self.root_dir,
                    format="zip",
                    delete_archive=False,
                )
            assert self._check_files_exists(assembly_path)
            return assembly_path

        def load_split(
            assembly_path: str,
            timebins,
            train_flag: bool,
        ):
            """
            Replicates the core logic of TVSDAssembly.prepare_data for one split
            and one timebin configuration.
            Returns:
                data_arr, ceiling_arr
            """
            data_list, ceiling_list = [], []
            for timebin in timebins:
                tb_data_list, tb_ceiling_list = [], []
                for monkey in self.monkeys:
                    tmd, tmc = self._get_timebin_monkey_data(
                        timebin, monkey, assembly_path, train_flag
                    )
                    tb_data_list.append(np.expand_dims(tmd, axis=0))
                    tb_ceiling_list.append(
                        np.expand_dims(compute_ceiling(tmc), axis=0)
                    )
                data_list.append(np.concatenate(tb_data_list, axis=-1))
                ceiling_list.append(np.concatenate(tb_ceiling_list, axis=-1))

            data_arr = np.concatenate(data_list, axis=0).squeeze()
            ceiling_arr = np.concatenate(ceiling_list, axis=0).squeeze()
            return data_arr, ceiling_arr

        # -----------------------
        # 1) TRAIN: defaultBins
        # -----------------------
        assembly_path_default = ensure_dir('defaultBins')
        # default timebin -> effectively a single bin
        train_data_default, train_ceiling_default = load_split(
            assembly_path_default, [''], train_flag=True
        )
        # train_data_default: (num_images, num_neurons)
        # train_ceiling_default: (num_neurons,)

        # -----------------------
        # 2) TEST: 10msBins
        # -----------------------
        assembly_path_10ms = ensure_dir('10msBins')
        timebins_10ms = range(25, 200, 10)  # 18 bins of 10 ms

        test_data_10ms, test_ceiling_10ms = load_split(
            assembly_path_10ms, timebins_10ms, train_flag=False
        )
        # test_data_10ms: (num_timebins, num_images, num_neurons)
        # test_ceiling_10ms: (num_timebins, num_neurons)

        # -----------------------
        # 3) Ceiling + mask
        # -----------------------
        # Aggregate ceiling over timebins to get a per-neuron value
        if test_ceiling_10ms.ndim == 2:
            per_neuron_ceiling = test_ceiling_10ms.mean(axis=0)
        else:  # already 1D
            per_neuron_ceiling = test_ceiling_10ms

        mask = per_neuron_ceiling >= self.ceiling_threshold
        # (In practice, threshold is very low -> mask ~ all True)

        # -----------------------
        # 4) Apply mask & set train
        # -----------------------
        # Train stays at default timebin resolution
        if train_data_default.ndim == 2:
            train_data_default = train_data_default[:, mask]
        elif train_data_default.ndim == 3:
            # Safety branch; should not happen for defaultBins.
            train_data_default = train_data_default.transpose(1, 2, 0)
            train_data_default = train_data_default[:, mask, :]

        self.train_data = train_data_default
        self.train_ceiling = per_neuron_ceiling[mask]

        # -----------------------
        # 5) Reshape test: 10ms → flattened
        # -----------------------
        # test_data_10ms is (timebins, images, neurons)
        if test_data_10ms.ndim == 3:
            test_data_10ms = test_data_10ms[:, :, mask]  # (T, I, N_mask)

            # Reorder to (images, timebins, neurons)
            test_data_10ms = np.transpose(test_data_10ms, (1, 0, 2))
            num_images, num_timebins, num_neurons = test_data_10ms.shape

            # Save how many repetitions (i.e., timebins of 10 ms)
            self.num_repetitions = num_timebins  # should be 18

            # Flatten (images, timebins) -> (images * timebins, neurons)
            self.test_data = test_data_10ms.reshape(
                num_images * num_timebins, num_neurons
            )
            self.test_ceiling = per_neuron_ceiling[mask]
        else:
            # Fallback: if for some reason it's already 2D
            self.test_data = test_data_10ms[:, mask]
            self.test_ceiling = per_neuron_ceiling[mask]


class TVSDRepeatedAssemblyFull(TVSDRepeatedAssemblyBase):
    """Repeated Assembly for the Full brain region (default train, 10ms test)."""

    def __init__(self, **kwargs):
        root = os.path.join(get_data_home(), TVSDAssembly.__name__)
        super().__init__(
            root_dir=root,
            region='Full',
            **kwargs
        )


class TVSDRepeatedAssemblyV1(TVSDRepeatedAssemblyBase):
    """Repeated Assembly for the V1 brain region (default train, 10ms test)."""

    def __init__(self, **kwargs):
        root = os.path.join(get_data_home(), TVSDAssembly.__name__)
        super().__init__(
            root_dir=root,
            region='V1',
            **kwargs
        )


class TVSDRepeatedAssemblyV4(TVSDRepeatedAssemblyBase):
    """Repeated Assembly for the V4 brain region (default train, 10ms test)."""

    def __init__(self, **kwargs):
        root = os.path.join(get_data_home(), TVSDAssembly.__name__)
        super().__init__(
            root_dir=root,
            region='V4',
            **kwargs
        )


class TVSDRepeatedAssemblyIT(TVSDRepeatedAssemblyBase):
    """Repeated Assembly for the IT brain region (default train, 10ms test)."""

    def __init__(self, **kwargs):
        root = os.path.join(get_data_home(), TVSDAssembly.__name__)
        super().__init__(
            root_dir=root,
            region='IT',
            **kwargs
        )
