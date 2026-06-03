# Examples

Worked notebooks and scripts live in the [`examples/`](https://github.com/ppalacios92/ShakerMakerResults/tree/main/examples)
directory of the repository. They cover the full path from a ShakerMaker run to
post-processed figures.

## Notebooks

| Example | What it shows |
|---|---|
| [`ShakerMakerResults.ipynb`](https://github.com/ppalacios92/ShakerMakerResults/blob/main/examples/ShakerMakerResults.ipynb) | End-to-end tour of the reader, plotting, and comparison API |
| [`drm_FSR_STG.ipynb`](https://github.com/ppalacios92/ShakerMakerResults/blob/main/examples/drm_FSR_STG.ipynb) | DRM workflow for the STG case (FSR parameters) |
| [`STG_Surface/STG_surface.ipynb`](https://github.com/ppalacios92/ShakerMakerResults/blob/main/examples/STG_Surface/STG_surface.ipynb) | Surface-grid results for the STG case |
| [`Gaussian_Surface/Gaussian_Surface.ipynb`](https://github.com/ppalacios92/ShakerMakerResults/blob/main/examples/Gaussian_Surface/Gaussian_Surface.ipynb) | Half-space surface grid, nearest-GF reuse |
| [`Gaussian_Perturbation_topo/Gaussian_Perturbation_topo.ipynb`](https://github.com/ppalacios92/ShakerMakerResults/blob/main/examples/Gaussian_Perturbation_topo/Gaussian_Perturbation_topo.ipynb) | Gaussian topographic perturbation |
| [`Compare_sw4_shaker/STG_surface.ipynb`](https://github.com/ppalacios92/ShakerMakerResults/blob/main/examples/Compare_sw4_shaker/STG_surface.ipynb) | Comparing ShakerMaker against SW4 |

## Generation scripts

These are the upstream ShakerMaker scripts that *produce* the HDF5 inputs (run
with MPI). ShakerMakerResults reads what they write.

| Script | Stage |
|---|---|
| [`STG_Surface/00_ffsp/main_rup_bl_1_1rl_FFSP.py`](https://github.com/ppalacios92/ShakerMakerResults/blob/main/examples/STG_Surface/00_ffsp/main_rup_bl_1_1rl_FFSP.py) | FFSP stochastic rupture |
| [`STG_Surface/01_surface/01_surface.py`](https://github.com/ppalacios92/ShakerMakerResults/blob/main/examples/STG_Surface/01_surface/01_surface.py) | DRM / surface-grid run |
| [`Gaussian_Surface/01_Gaussian_Surface/main_HalfSpace_SURFACE_nearest.py`](https://github.com/ppalacios92/ShakerMakerResults/blob/main/examples/Gaussian_Surface/01_Gaussian_Surface/main_HalfSpace_SURFACE_nearest.py) | half-space surface, nearest-GF |

!!! note "Large result files"
    The HDF5 results themselves are not committed. Some folders contain a
    `link_dowload.txt` with a download link for the corresponding `.h5drm` /
    `*_gf.h5` files.

## A self-contained snippet

```python
from ShakerMakerResults import ShakerMakerData, plot_models_response

model  = ShakerMakerData("surface_case.h5drm")
window = model.get_window(t_start=0.0, t_end=40.0)

# Single model
window.plot_node_response(node_id=10, data_type="vel")
window.plot_surface(time=12.5, component="z", data_type="vel")

# Two models, overlaid
plot_models_response(models=[window, window.resample(dt=0.01)],
                     node_ids=[10, 10], data_type="vel")
```
