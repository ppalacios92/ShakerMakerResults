"""
export_service.py
=================
HDF5 writers that take a (possibly windowed) ``ShakerMakerData`` instance
and produce a self-contained ``.h5drm`` file.

The writer copies metadata + node geometry verbatim and only re-slices
the time-series datasets along the column axis when a window mask is
active. That way ``model.get_window(t0, t1).write_h5drm()`` produces a
drop-in file usable by the downstream DRM pipeline.
"""

from __future__ import annotations

import os

import h5py
import numpy as np


def write_h5drm(model, name=None):
    """Write the active time window of ``model`` to a new ``.h5drm`` file.

    Parameters
    ----------
    model : ShakerMakerData
        Source reader. If a window mask is attached (set by
        :meth:`ShakerMakerData.get_window`), only the masked columns are
        copied; otherwise the file is essentially a copy.
    name : str, optional
        Output filename. Defaults to ``<orig_stem>_t<start>_<end>.h5drm``
        placed next to the original file.

    Returns
    -------
    str
        Absolute path of the written file.

    Notes
    -----
    A ``tqdm`` progress bar is shown while copying the (potentially huge)
    time-series datasets. ``tqdm`` is imported lazily so users who never
    write files don't pay the import cost.
    """
    from tqdm import tqdm

    orig_dir = os.path.dirname(model.filename)
    orig_base = os.path.splitext(os.path.basename(model.filename))[0]
    if name is None:
        t0_str = f"{model.time[0]:.1f}".replace(".", "p")
        t1_str = f"{model.time[-1]:.1f}".replace(".", "p")
        name = f"{orig_base}_t{t0_str}_{t1_str}.h5drm"
    out_path = os.path.join(orig_dir, name)
    print(f"Writing: {out_path}")

    data_grp = model._data_grp
    meta_grp = model._meta_grp
    qa_grp = model._qa_grp

    if hasattr(model, "_window_mask"):
        col_idx = np.where(model._window_mask)[0]
    else:
        col_idx = np.arange(model._n_time_data)
    c0, c1 = int(col_idx[0]), int(col_idx[-1]) + 1

    with h5py.File(model.filename, "r") as fin, h5py.File(out_path, "w") as fout:
        fin.copy(meta_grp, fout)
        fout[f"{meta_grp}/dt"][()] = model._dt_orig
        fout[f"{meta_grp}/tstart"][()] = float(model.time[0])
        fout[f"{meta_grp}/tend"][()] = float(model.time[-1])

        for key in fin.keys():
            if key not in (data_grp, meta_grp, qa_grp or ""):
                fin.copy(key, fout)

        orig_vel_shape = fin[f"{data_grp}/velocity"].shape
        fout.create_group(data_grp)

        ts_keys = []
        for key in fin[data_grp].keys():
            ds = fin[f"{data_grp}/{key}"]
            if len(ds.shape) == 2 and ds.shape[1] == orig_vel_shape[1]:
                ts_keys.append(key)
            else:
                fout[f"{data_grp}/{key}"] = ds[:]

        if qa_grp and qa_grp in fin:
            fout.create_group(qa_grp)
            for key in fin[qa_grp].keys():
                ds = fin[f"{qa_grp}/{key}"]
                if len(ds.shape) == 2 and ds.shape[1] == orig_vel_shape[1]:
                    fout[f"{qa_grp}/{key}"] = ds[:, c0:c1]
                else:
                    fout[f"{qa_grp}/{key}"] = ds[:]

        for key in tqdm(ts_keys, desc="write_h5drm", unit="dataset"):
            ds = fin[f"{data_grp}/{key}"]
            fout[data_grp].create_dataset(
                key,
                data=ds[:, c0:c1],
                dtype=ds.dtype,
                chunks=True,
            )

    size_gb = os.path.getsize(out_path) / 1e9
    print(f"Done. {size_gb:.2f} GB")
    return out_path
