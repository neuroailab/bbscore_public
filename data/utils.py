"""
Code mostly borrowed from https://github.com/brain-score/
"""
import xarray as xr
import numpy as np

from collections import OrderedDict
from xarray import IndexVariable
from scipy.stats import pearsonr

from torch.utils.data._utils.collate import default_collate


def array_is_element(arr, element):
    return len(arr) == 1 and arr[0] == element


def get_metadata_before_2022_06(assembly, dims=None, names_only=False,
                                include_coords=True,
                                include_indexes=True,
                                include_multi_indexes=False,
                                include_levels=True):
    """
    Return coords and/or indexes or index levels from an assembly, yielding
    either `name` or `(name, dims, values)`.
    """
    def what(name, dims, values, names_only):
        if names_only:
            return name
        else:
            return name, dims, values
    if dims is None:
        dims = assembly.dims + (None,)  # all dims plus dimensionless coords
    for name in assembly.coords.variables:
        values = assembly.coords.variables[name]
        is_subset = values.dims and (set(values.dims) <= set(dims))
        is_dimless = (not values.dims) and None in dims
        if is_subset or is_dimless:
            is_index = isinstance(values, IndexVariable)
            if is_index:
                if values.level_names:  # it's a MultiIndex
                    if include_multi_indexes:
                        yield what(name, values.dims, values.values,
                                   names_only)
                    if include_levels:
                        for level in values.level_names:
                            level_values = assembly.coords[level]
                            yield what(level, level_values.dims,
                                       level_values.values, names_only)
                else:  # it's an Index
                    if include_indexes:
                        yield what(name, values.dims, values.values,
                                   names_only)
            else:
                if include_coords:
                    yield what(name, values.dims, values.values, names_only)


def get_metadata_after_2022_06(assembly, dims=None, names_only=False,
                               include_coords=True,
                               include_indexes=True,
                               include_multi_indexes=False,
                               include_levels=True):
    """
    Return coords and/or indexes or index levels from an assembly, yielding
    either `name` or `(name, dims, values)`.
    """
    def what(name, dims, values, names_only):
        if names_only:
            return name
        else:
            return name, dims, values
    if dims is None:
        dims = assembly.dims + (None,)  # all dims plus dimensionless coords
    for name, values in assembly.coords.items():
        none_but_keep = (not values.dims) and None in dims
        shared = not (set(values.dims).isdisjoint(set(dims)))
        if none_but_keep or shared:
            if name in assembly.indexes:  # it's an index
                index = assembly.indexes[name]
                if len(index.names) > 1:  # it's a MultiIndex or level
                    if name in index.names:  # it's a level
                        if include_levels:
                            yield what(name, values.dims, values.values,
                                       names_only)
                    else:  # it's a MultiIndex
                        if include_multi_indexes:
                            yield what(name, values.dims, values.values,
                                       names_only)
                else:  # it's a single Index
                    if include_indexes:
                        yield what(name, values.dims, values.values,
                                   names_only)
            else:  # it's a coord
                if include_coords:
                    yield what(name, values.dims, values.values, names_only)


def get_metadata(assembly, dims=None, names_only=False, include_coords=True,
                 include_indexes=True, include_multi_indexes=False,
                 include_levels=True):
    try:
        xr.DataArray().stack(create_index=True)
        yield from get_metadata_after_2022_06(assembly, dims, names_only,
                                              include_coords,
                                              include_indexes,
                                              include_multi_indexes,
                                              include_levels)
    except TypeError:
        yield from get_metadata_before_2022_06(assembly, dims, names_only,
                                               include_coords,
                                               include_indexes,
                                               include_multi_indexes,
                                               include_levels)


def coords_for_dim(assembly, dim):
    result = OrderedDict()
    meta = get_metadata(assembly, dims=(
        dim,), include_indexes=False, include_levels=False)
    for name, dims, values in meta:
        result[name] = values
    return result


def walk_coords(assembly):
    """
    walks through coords and all levels, just like the `__repr__` function,
    yielding `(name, dims, values)`.
    """
    yield from get_metadata(assembly)


class MultiCoord:
    def __init__(self, values):
        self.values = tuple(values) if isinstance(values, list) else values

    def __eq__(self, other):
        return len(self.values) == len(other.values) and \
            all(v1 == v2 for v1, v2 in zip(self.values, other.values))

    def __lt__(self, other):
        return self.values < other.values

    def __hash__(self):
        return hash(self.values)

    def __repr__(self):
        return repr(self.values)

    def __getitem__(self, index):
        return self.values[index]


def multi_groupby(assembly: xr.DataArray, group_coord_names, *args, **kwargs):
    if len(group_coord_names) < 2:
        return assembly.groupby(group_coord_names[0], *args, **kwargs)

    # Get the single dimension all coords share
    dimses = [assembly.coords[c].dims for c in group_coord_names]
    dims = [dim for dim_tuple in dimses for dim in dim_tuple]
    if len(set(dims)) != 1:
        raise ValueError(
            "All coordinates must be associated with the same dimension.")
    dim = dims[0]

    # Create a new MultiCoord index
    group_coords = [assembly.coords[c].values.tolist()
                    for c in group_coord_names]
    multi_group_coord = [MultiCoord(coords) for coords in zip(*group_coords)]

    # Assign the multi_group coordinate and set as index
    multi_group_name = "multi_group"
    tmp_assy = assembly.copy()
    tmp_assy.coords[multi_group_name] = dim, multi_group_coord
    # Perform groupby
    return tmp_assy.groupby(multi_group_name, *args, **kwargs)


def average_repetition(assembly):
    def avg_repr(assembly):
        presentation_coords = [
            coord for coord, dims, values in walk_coords(assembly)
            if array_is_element(dims, 'presentation') and coord != 'repetition'
        ]
        assembly = multi_groupby(assembly, presentation_coords).mean(
            dim='presentation', skipna=True)
        return assembly

    return apply_keep_attrs(assembly, avg_repr)


def apply_keep_attrs(assembly, fnc):  # workaround to keeping attrs
    attrs = assembly.attrs
    assembly = fnc(assembly)
    assembly.attrs = attrs
    return assembly


def split_half_consistency(data, n_splits=50, aggregate=np.mean, rng=None):
    """
    Optimized, vectorized implementation with a provided RNG for reproducibility.
    """
    n_trials, n_conditions, n_units = data.shape
    assert n_trials >= 2, "Need at least 2 trials to split."
    if rng is None:
        rng = np.random

    half = n_trials // 2
    scores = np.zeros((n_splits, n_units), dtype=np.float64)
    for s in range(n_splits):
        idx = rng.permutation(n_trials)

        S1 = data[idx[:half], :, :].mean(axis=0)
        S2 = data[idx[half:], :, :].mean(axis=0)

        μ1 = S1.mean(axis=0)
        μ2 = S2.mean(axis=0)

        C1 = S1 - μ1[np.newaxis, :]
        C2 = S2 - μ2[np.newaxis, :]

        num = np.sum(C1 * C2, axis=0)
        denom = np.sqrt(np.sum(C1 * C1, axis=0) * np.sum(C2 * C2, axis=0))

        corr = num / denom
        corr = np.nan_to_num(corr)
        sb = 2.0 * corr / (1.0 + corr)

        scores[s, :] = sb
    if aggregate is not None:
        return aggregate(scores, axis=0)
    else:
        return scores


def one_vs_all_consistency(data, aggregate=np.mean):
    """
    Optimized, vectorized “one‐versus‐all” consistency (Spearman–Brown‐corrected)
    for data shaped (n_trials, n_conditions, n_units).

    Returns either:
    - an (n_units,) array if `aggregate` is not None (default np.mean across trials), or
    - an (n_trials, n_units) array if `aggregate=None`.
    """
    n_trials, n_conditions, n_units = data.shape
    assert n_trials >= 2, "Need at least 2 trials"

    # Preallocate (n_trials × n_units)
    scores = np.zeros((n_trials, n_units), dtype=np.float64)

    # Compute total sum across trials for each (condition, unit)
    total_sum = data.sum(axis=0)

    for t in range(n_trials):
        # “One‐versus‐all”: current trial vs. mean of others
        # shape: (n_conditions, n_units)
        S1 = data[t, :, :]
        # shape: (n_conditions, n_units)
        S2 = (total_sum - S1) / (n_trials - 1)

        # Center each unit's response across conditions
        mu1 = S1.mean(axis=0)                           # shape: (n_units,)
        mu2 = S2.mean(axis=0)                           # shape: (n_units,)
        # (n_conditions, n_units)
        C1 = S1 - mu1[np.newaxis, :]
        # (n_conditions, n_units)
        C2 = S2 - mu2[np.newaxis, :]

        # Vectorized Pearson numerator / denominator
        num = np.sum(C1 * C2, axis=0)                   # (n_units,)
        denom = np.sqrt(np.sum(C1 * C1, axis=0) *
                        np.sum(C2 * C2, axis=0))        # (n_units,)
        corr = num / denom
        corr = np.nan_to_num(corr)                      # replace 0/0 with 0

        # Spearman–Brown correction: r_sb = 2r / (1 + r)
        sb = 2.0 * corr / (1.0 + corr)
        scores[t, :] = sb

    return aggregate(scores, axis=0) if aggregate is not None else scores


def merge_list_of_dicts(list_of_dicts):
    if not list_of_dicts:
        return {}

    # Initialize the result dict with empty lists for each key
    merged = {key: [] for key in list_of_dicts[0].keys()}

    # Iterate over each dict in the list and extend the corresponding lists
    for d in list_of_dicts:
        for key, value_list in d.items():
            merged[key].append(value_list)
    return merged


def custom_collate(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    first = batch[0]
    if isinstance(first, tuple) and isinstance(first[0], dict):
        # Extract list_of_dicts and labels as before
        list_of_dicts = [item[0] for item in batch]
        list_of_labels = [item[1] for item in batch]
        # Merge all dicts into one
        merged_dict = merge_list_of_dicts(list_of_dicts)
        # Stack labels in the usual way
        stacked_labels = default_collate(list_of_labels)
        return merged_dict, stacked_labels
    if isinstance(first, dict):
        list_of_dicts = [item for item in batch]
        merged_dict = merge_list_of_dicts(list_of_dicts)
        return merged_dict
    return default_collate(batch)
