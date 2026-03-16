"""
comparison.py
=============
Quantitative comparison functions for ShakerMakerData and StationRead objects.

Functions compute standard signal-similarity metrics (GoF, Peak Error,
Pearson correlation, RMSE) and spectral metrics, printing a structured
summary to stdout.

Author: Patricio Palacios B.
"""

import numpy as np
from .newmark import NewmarkSpectrumAnalyzer

# ---------------------------------------------------------------------------
# Internal helpers  (shared by both comparison functions)
# ---------------------------------------------------------------------------

def _is_station(obj):
    return hasattr(obj, 'z_v') and not hasattr(obj, 'internal')


def _resolve_node(node_id, model_index, n_models):
    if not isinstance(node_id, list):
        return node_id
    if isinstance(node_id[0], list):
        return node_id[model_index][0]
    if len(node_id) == n_models:
        return node_id[model_index]
    return node_id[0]


def _get_signals(obj, node_idx, data_type, filtered=False):
    """Return [Z, E, N] arrays from ShakerMakerData or StationRead."""
    if _is_station(obj):
        if data_type in ('accel', 'acceleration'):
            z, e, n = obj.acceleration_filtered if filtered else obj.acceleration
        elif data_type in ('vel', 'velocity'):
            z, e, n = obj.velocity_filtered if filtered else obj.velocity
        else:
            z, e, n = obj.displacement_filtered if filtered else obj.displacement
        return [z, e, n]

    if node_idx in ('QA', 'qa') or (
            isinstance(node_idx, int) and node_idx >= len(obj.xyz)):
        d = obj.get_qa_data(data_type)
    else:
        d = obj.get_node_data(node_idx, data_type)
    return [d[2], d[0], d[1]]   # Z, E, N


def _get_time(obj):
    return obj.t if _is_station(obj) else obj.time


def _get_name(obj):
    if _is_station(obj):
        return obj.name if obj.name else "Station"
    return obj.model_name


def _metrics(sig_ref, sig_test):
    """Return (GoF, peak_error_pct, pearson_corr, rmse)."""
    diff    = sig_ref - sig_test
    num     = np.sum(diff ** 2)
    den     = np.sum(sig_ref ** 2 + sig_test ** 2)
    gof     = float(1 - np.sqrt(num / den)) if den > 0 else 0.0
    max_ref = np.max(np.abs(sig_ref))
    peak    = float(np.max(np.abs(diff)) / max_ref * 100) if max_ref > 0 else 0.0
    corr    = float(np.corrcoef(sig_ref, sig_test)[0, 1])
    rmse    = float(np.sqrt(np.mean(diff ** 2)))
    return gof, peak, corr, rmse


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compare_node_response(models, node_id, data_type='vel',
                          reference_index=0, filtered=False):
    """Compare time-history signals from multiple models against a reference.

    Signals are interpolated to a common time grid before computing metrics.

    Parameters
    ----------
    models : list of ShakerMakerData or StationRead
        Mixed lists are supported.
    node_id : int, str, or list
        Node index for ShakerMakerData objects. Ignored for StationRead.
        Supports: scalar, per-model list, or list of lists.
    data_type : {'vel', 'accel', 'disp'}, default ``'vel'``
    reference_index : int, default ``0``
        Index of the reference model in ``models``.
    filtered : bool, default ``False``
        Use filtered data for StationRead objects.

    Returns
    -------
    dict
        Nested ``{model_name: {component: {metric: value}}}`` dictionary.
    """
    n          = len(models)
    components = ['Z', 'E', 'N']

    data_all  = []
    times_all = []
    nids_used = []

    for i, obj in enumerate(models):
        nid = _resolve_node(node_id, i, n)
        nids_used.append('Station' if _is_station(obj) else nid)
        data_all.append(_get_signals(obj, nid, data_type, filtered))
        times_all.append(_get_time(obj))

    ref_data = data_all[reference_index]
    ref_time = times_all[reference_index]
    ref_name = _get_name(models[reference_index])

    results = {}
    print("=" * 70)
    print(f"COMPARISON vs Reference: {ref_name}  |  data_type={data_type}")
    print(f"Reference node: {nids_used[reference_index]}")
    print("=" * 70)

    for i in range(n):
        if i == reference_index:
            continue

        name      = _get_name(models[i])
        test_data = data_all[i]
        test_time = times_all[i]

        print(f"\nModel: {name}  |  node: {nids_used[i]}")
        model_res = {}

        for ic, comp in enumerate(components):
            t_common = np.linspace(
                max(ref_time[0],  test_time[0]),
                min(ref_time[-1], test_time[-1]),
                min(len(ref_time), len(test_time)))

            s_ref  = np.interp(t_common, ref_time,  ref_data[ic])
            s_test = np.interp(t_common, test_time, test_data[ic])

            gof, peak, corr, rmse = _metrics(s_ref, s_test)
            model_res[comp] = {'gof': gof, 'peak_err': peak,
                                'corr': corr, 'rmse': rmse}
            print(f"  {comp}: GoF={gof:.4f}  PeakErr={peak:.2f}%  "
                  f"Corr={corr:.4f}  RMSE={rmse:.6f}")

        results[name] = model_res

    print("=" * 70)
    return results


def compare_spectra(models, node_id, data_type='accel',
                    spectral_type='PSa', reference_index=0,
                    filtered=False):
    """Compare Newmark response spectra from multiple models against a reference.

    Parameters
    ----------
    models : list of ShakerMakerData or StationRead
        Mixed lists are supported.
    node_id : int, str, or list
        Node index for ShakerMakerData objects. Ignored for StationRead.
    data_type : {'accel', 'vel', 'disp'}, default ``'accel'``
    spectral_type : {'PSa', 'Sa', 'PSv', 'Sv', 'Sd'}, default ``'PSa'``
        Spectral quantity to compare.
    reference_index : int, default ``0``
    filtered : bool, default ``False``
        Use filtered data for StationRead objects.

    Returns
    -------
    dict
        Nested ``{model_name: {component: {metric: value}}}`` dictionary.
    """
    n          = len(models)
    components = [('Z', 0), ('E', 1), ('N', 2)]
    scale      = 1.0 / 9.81 if data_type == 'accel' else 1.0

    specs_all = []
    nids_used = []

    for i, obj in enumerate(models):
        nid = _resolve_node(node_id, i, n)
        nids_used.append('Station' if _is_station(obj) else nid)

        if _is_station(obj):
            # StationRead: use its built-in Newmark cache
            sp = obj.get_newmark(filtered=filtered)
            T  = sp['T']
            specs_all.append({
                'T':   T,
                'Z':   sp['PSa_z'],
                'E':   sp['PSa_e'],
                'N':   sp['PSa_n'],
            })
        else:
            sigs = _get_signals(obj, nid, data_type)
            dt   = obj.time[1] - obj.time[0]
            raw  = [NewmarkSpectrumAnalyzer.compute(sig * scale, dt)
                    for sig in sigs]
            T    = raw[0]['T']
            specs_all.append({
                'T': T,
                'Z': raw[0][spectral_type],
                'E': raw[1][spectral_type],
                'N': raw[2][spectral_type],
            })

    ref_spec = specs_all[reference_index]
    ref_T    = ref_spec['T']
    ref_name = _get_name(models[reference_index])

    results = {}
    print("=" * 70)
    print(f"SPECTRA COMPARISON vs Reference: {ref_name}")
    print(f"  data_type={data_type}  |  spectral_type={spectral_type}")
    print(f"  Reference node: {nids_used[reference_index]}")
    print("=" * 70)

    for i in range(n):
        if i == reference_index:
            continue

        name      = _get_name(models[i])
        test_spec = specs_all[i]
        test_T    = test_spec['T']

        print(f"\nModel: {name}  |  node: {nids_used[i]}")
        model_res = {}

        for comp, _ in components:
            T_common = np.linspace(
                max(ref_T[0],  test_T[0]),
                min(ref_T[-1], test_T[-1]),
                min(len(ref_T), len(test_T)))

            s_ref  = np.interp(T_common, ref_T,  ref_spec[comp])
            s_test = np.interp(T_common, test_T, test_spec[comp])

            gof, peak, corr, rmse = _metrics(s_ref, s_test)
            model_res[comp] = {'gof': gof, 'peak_err': peak,
                                'corr': corr, 'rmse': rmse}
            print(f"  {comp}: GoF={gof:.4f}  PeakErr={peak:.2f}%  "
                  f"Corr={corr:.4f}  RMSE={rmse:.6f}")

        results[name] = model_res

    print("=" * 70)
    return results
