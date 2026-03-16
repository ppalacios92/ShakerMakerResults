# ShakerMakerResults

A Python-based tool for reading, analyzing, and visualizing ShakerMaker simulation results stored in HDF5 format. Supports both DRM (Domain Reduction Method) and station outputs.

---

## Features

- Read ShakerMaker results from HDF5 files (DRM and station outputs)
- Extract node coordinates, velocity, acceleration and displacement
- Visualize domains in 3D
- Plot time histories and Fourier spectra
- Compare multiple models
- Compute Newmark response spectra

---

## Requirements

- Python 3.8+
- numpy, h5py, matplotlib, scipy, numba

---

## Installation
```bash
git clone https://github.com/ppalacios92/ShakerMakerResults.git
cd ShakerMakerResults
pip install -e .
```

---

## Repository Structure
```bash
ShakerMakerResults/
├── src/
│   └── shakermaker_results/
│       ├── __init__.py
│       ├── shakermaker_data.py
│       ├── plotting.py
│       └── newmark.py
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
@misc{palacios2025shakermakerresults,
  author       = {Patricio Palacios B.},
  title        = {ShakerMakerResults: A Python-based ShakerMaker results reader and visualization tool},
  year         = {2025},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/ppalacios92/ShakerMakerResults}}
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