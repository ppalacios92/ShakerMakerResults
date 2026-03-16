# DRM_DATA

A Python-based tool for reading, analyzing, and visualizing Domain Reduction Method (DRM) data stored in HDF5 format.

---

## Features

- Read DRM data from HDF5 files
- Extract node coordinates, Green's functions, and response spectra
- Visualize DRM domains in 3D
- Plot time histories and Fourier spectra
- Compare multiple DRM models
- Compute Newmark response spectra

---

## Requirements

- Python 3.8+
- numpy, h5py, matplotlib, scipy, numba

---

## Installation
```bash
git clone https://github.com/ppalacios92/DRM_DATA.git
cd DRM_DATA
pip install -e .
```

---

## Repository Structure
```bash
DRM_DATA/
├── drm_data/
│   ├── __init__.py
│   ├── drm.py
│   ├── plotting.py
│   └── newmark.py
├── pyproject.toml
├── README.md
└── LICENSE
```

---

## 🛑 Disclaimer

This tool is provided as-is, without any guarantees of accuracy or suitability for specific engineering tasks.

---

## Author

Developed by Patricio Palacios B.
GitHub: @ppalacios92

---

## How to Cite
```bibtex
@misc{palacios2025drmdata,
  author       = {Patricio Palacios B.},
  title        = {DRM_DATA: A Python-based DRM data reader and visualization tool},
  year         = {2025},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/ppalacios92/DRM_DATA}}
}
```

---

## License

MIT License

---

**LICENSE**
```
MIT License

Copyright (c) 2025 Patricio Palacios B.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```