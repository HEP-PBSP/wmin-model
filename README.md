# wmin-model
![Tests bagde](https://github.com/HEP-PBSP/wmin-model/actions/workflows/tests.yml/badge.svg)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![codecov](https://codecov.io/gh/HEP-PBSP/wmin-model/graph/badge.svg?token=uYUy3rXCWK)](https://codecov.io/gh/HEP-PBSP/wmin-model)

Weight Minimisation PDF-model of Colibri

## wmin-model Installation
From your base conda environment run:
```
conda env create -f environment.yml
```
this will create a conda environment called `wmin-model-dev` that has a `wmin` executable and all the needed dependencies (e.g. `colibri`). 
To use a different environment name, one should do
```
conda env create -n myenv -f environment.yml
```

## Colibri development mode
The above procedure installs the model in editable mode, but colibri is not. If developing colibri as well, a further simple step is required.
Activate the environment, go to the colibri repository and install it in editable mode:
```
cd /path/to/colibri/colibri
pip install -e .
```

