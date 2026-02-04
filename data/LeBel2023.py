import os
import glob
import numpy as np
import h5py
import re
from typing import Optional, List, Union, Callable
from data.base import BaseDataset


class LeBel2023StimulusSet(BaseDataset):
    """
    Stimulus set for the LeBel et al. (2023) dataset (OpenNeuro ds003020).
    Consists of natural language narratives.
    """

    def __init__(self, root_dir: Optional[str] = None, preprocess: Optional[Callable] = None):
        super().__init__(root_dir)
        self.preprocess = preprocess
        self.stimuli = []
        self.stimuli_ids = []

        self.dataset_dir = os.path.join(self.root_dir, "ds003020")
        self.textgrid_dir = os.path.join(
            self.dataset_dir, "derivative", "TextGrids")

        self._prepare_stimuli()

    def _prepare_stimuli(self):
        s3_source = "s3://openneuro.org/ds003020/derivative/TextGrids/"

        # Download TextGrids if not present
        if not os.path.exists(self.textgrid_dir) or not os.listdir(self.textgrid_dir):
            try:
                print(f"Downloading TextGrids from {s3_source}...")
                self.fetch(
                    source=s3_source,
                    target_dir=os.path.dirname(self.textgrid_dir),
                    filename="TextGrids",
                    method="s3",
                    anonymous=True
                )
            except Exception as e:
                print(f"Error downloading TextGrids: {e}")

        # Parse TextGrids
        tg_files = sorted(
            glob.glob(os.path.join(self.textgrid_dir, "*.TextGrid")))
        print(f"DEBUG: Found {len(tg_files)} TextGrid files.")
        if not tg_files:
            raise FileNotFoundError(
                f"No .TextGrid files found in {self.textgrid_dir}")

        for tg_file in tg_files:
            story_name = os.path.basename(tg_file).replace(".TextGrid", "")
            try:
                text = self._parse_textgrid(tg_file)
                self.stimuli.append(text)
                self.stimuli_ids.append(story_name)
            except Exception as e:
                print(f"Failed to parse {tg_file}: {e}")

    def _parse_textgrid(self, filepath):
        """
        Simple TextGrid parser to extract words from the 'words' tier.
        Assumes standard Praat short/long text format.
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        # Find the 'words' tier
        # This is a heuristic parser. A robust one would use a library.
        # We look for: name = "words" ... intervals [x]: ... text = "the"

        words = []
        # Split by intervals usually works if we find the right tier
        # But file structure can vary.
        # Let's try a regex approach for "text =" fields inside the words tier.

        # Locate words tier
        tier_match = re.search(
            r'name = "words"(.*?)(name = |$)', content, re.DOTALL)
        if tier_match:
            tier_content = tier_match.group(1)
            # Extract text = "..."
            # Note: Praat saves text = "..." or text = ""
            matches = re.findall(r'text = "(.*?)"', tier_content)
            # Filter out empty strings (silences) if needed, or keep them?
            # Usually we want the transcript.
            filtered_words = [w for w in matches if w.strip()]
            return " ".join(filtered_words)

        return ""

    def __len__(self):
        return len(self.stimuli)

    def __getitem__(self, idx):
        text = self.stimuli[idx]
        if self.preprocess:
            return self.preprocess(text)
        return text


class LeBel2023Assembly(BaseDataset):
    """
    fMRI Assembly for LeBel et al. (2023).
    """

    def __init__(self, root_dir: Optional[str] = None, subjects: Union[str, List[str]] = ['UTS01']):
        super().__init__(root_dir)
        if isinstance(subjects, str):
            subjects = [subjects]
        self.subjects = subjects
        self.dataset_dir = os.path.join(self.root_dir, "ds003020")
        self.data_dir = os.path.join(
            self.dataset_dir, "derivative", "preprocessed_data")

    def get_assembly(self):
        """
        Returns:
            fmri_data: (n_stories, n_voxels) - AVERAGED over time for now to match benchmark structure
                       OR concatenated time series if benchmark supports it.
            noise_ceiling: (n_voxels,)
        """
        s3_base = "s3://openneuro.org/ds003020/derivative/preprocessed_data/"

        all_subject_data = []

        # We assume the StimulusSet has loaded stories in sorted order of filenames.
        # We must load fMRI files in the SAME order.
        # StimulusSet loads *.TextGrid sorted.
        # We should load matching *.hf5 files.

        # Get list of stories from the TextGrid directory to ensure alignment
        tg_dir = os.path.join(self.dataset_dir, "derivative", "TextGrids")
        if not os.path.exists(tg_dir):
            # Try to instantiate stimulus set to trigger download?
            # Or just trust it exists if Benchmark calls stimulus first.
            # We'll just assume sorted glob of hf5 matches sorted glob of TextGrid
            pass

        for subj in self.subjects:
            subj_path = os.path.join(self.data_dir, subj)

            # Check for existing data
            hf5_files = sorted(glob.glob(os.path.join(subj_path, "*.hf5")))

            # Check recursive if needed
            if not hf5_files and os.path.exists(subj_path):
                hf5_files = sorted(glob.glob(os.path.join(
                    subj_path, "**", "*.hf5"), recursive=True))

            # Validate count (we expect 84 stories)
            if len(hf5_files) < 84:
                print(
                    f"Found {len(hf5_files)} files for {subj}, expected 84. Redownloading...")
                import shutil
                if os.path.exists(subj_path):
                    shutil.rmtree(subj_path)
                hf5_files = []  # Force download

            if not hf5_files:
                # If directory exists but empty (or contains wrong stuff), clean it up to ensure fetch runs
                if os.path.exists(subj_path):
                    try:
                        # Check if it has the weird hash dir and move files?
                        # Or just nuke it to be safe and redownload (safer for reproducibility)
                        # But user might have slow connection.
                        # Let's check for subdirectories and flatten if possible?
                        # Too complex. Let's just remove empty dir.
                        if not os.listdir(subj_path):
                            os.rmdir(subj_path)
                    except OSError:
                        pass

                # Download subject data
                try:
                    print(f"Downloading fMRI data for {subj}...")
                    self.fetch(
                        source=f"{s3_base}{subj}/",
                        target_dir=self.data_dir,
                        filename=subj,
                        method="s3",
                        anonymous=True
                    )
                except Exception as e:
                    print(f"Error downloading data for {subj}: {e}")

            # Reload file list
            hf5_files = sorted(glob.glob(os.path.join(subj_path, "*.hf5")))

            # If still no files, check if they are nested (e.g. from previous bad download)
            if not hf5_files and os.path.exists(subj_path):
                nested_hf5 = sorted(glob.glob(os.path.join(
                    subj_path, "**", "*.hf5"), recursive=True))
                if nested_hf5:
                    hf5_files = nested_hf5

            print(f"DEBUG: Found {len(hf5_files)} HF5 files for {subj}.")

            if not hf5_files:
                raise FileNotFoundError(f"No .hf5 files found for {subj}")

            subj_data_list = []
            for f in hf5_files:
                # Load data from HDF5
                # Key inside hf5 is typically 'data' or the story name
                try:
                    with h5py.File(f, 'r') as hf:
                        # Inspect keys
                        keys = list(hf.keys())
                        # Usually 'data' or 'dset'
                        # Based on inspection of similar datasets, it's often 'data'
                        # We'll try common keys
                        dset = None
                        for k in ['data', 'dset', 'roi', 'rep']:
                            if k in keys:
                                dset = hf[k][:]
                                break
                        if dset is None:
                            # Fallback: take the first key
                            dset = hf[keys[0]][:]

                        # dset shape: (time, voxels)
                        # To match the "Story" granularity of StimulusSet (1 string),
                        # we might need to average over time or similar.
                        # OR we assume the pipeline handles (1, T, V).

                        # For now, let's just take the MEAN over time to get (1, voxels)
                        # This creates a "story vector".

                        subj_data_list.append(np.nanmean(dset, axis=0))

                except Exception as e:
                    print(f"Error loading {f}: {e}")

            # Stack stories: (n_stories, n_voxels)
            if subj_data_list:
                print(
                    f"DEBUG: Stacking {len(subj_data_list)} stories for {subj}")
                all_subject_data.append(np.stack(subj_data_list, axis=0))

        if not all_subject_data:
            raise ValueError("No data loaded.")

        # Check if shapes match for averaging/stacking
        shapes = [d.shape for d in all_subject_data]

        if len(all_subject_data) > 1:
            # If shapes differ, we MUST concatenate (cannot average)
            if len(set(shapes)) > 1:
                print(
                    f"Warning: Subject shapes differ {shapes}. Concatenating features.")
                fmri_data = np.concatenate(all_subject_data, axis=1)
            else:
                # If shapes match, we usually average for shared benchmarks,
                # BUT since we know these subjects aren't aligned, averaging is geometrically wrong.
                # Concatenation is safer to preserve information.
                print(
                    f"Note: Multiple subjects loaded. Concatenating {len(all_subject_data)} subjects.")
                fmri_data = np.concatenate(all_subject_data, axis=1)
        else:
            fmri_data = all_subject_data[0]

        # Handle NaNs
        fmri_data = np.nan_to_num(fmri_data)

        print(f"DEBUG: Final fmri_data shape: {fmri_data.shape}")

        # Fake ceiling
        ncsnr = np.ones(fmri_data.shape[1], dtype=np.float32)

        return fmri_data, ncsnr

    def __len__(self):
        # Placeholder or compute
        # Since get_assembly loads data, we can call it or use a default if known
        # In this dataset, there are 27 stories/files.
        return 27

    def __getitem__(self, idx):
        # This dataset usually accessed via get_assembly() for the whole matrix.
        # But for BaseDataset compatibility:
        # We can lazy load or just return a placeholder if not loaded yet.
        # But to be safe, we can try to load just that one file if possible.
        # However, BaseDataset structure here seems to focus on get_assembly for the benchmark.
        # We'll return a dummy value if get_assembly hasn't been called, or look it up.
        # Given the complexity, let's just return None or raise NotImplementedError
        # unless we cache the data in __init__ which is expensive.
        # BUT, the Scorer might not use __getitem__ for Assembly if it uses get_assembly.
        # So we just need to satisfy the abstract class.
        return None
