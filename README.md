# wmin-model
![Tests bagde](https://github.com/HEP-PBSP/wmin-model/actions/workflows/tests.yml/badge.svg)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![codecov](https://codecov.io/gh/HEP-PBSP/wmin-model/graph/badge.svg?token=uYUy3rXCWK)](https://codecov.io/gh/HEP-PBSP/wmin-model)

wmin-model is a [Colibri](https://github.com/HEP-PBSP/colibri) PDF-model that implements the POD parametrisation presented 
in (TODO: arxiv id).

## Installation

There are several ways to install wmin-model, see also [colibri installation instructions](https://hep-pbsp.github.io/colibri/get-started/installation.html), however perhaps the easiest way
is to clone the repository first and then use the provided `environment.yml` file:

```bash
git clone git@github.com:HEP-PBSP/wmin-model.git
cd wmin-model
```

from your conda base environment run 

```bash
conda env create -f environment.yml

```

This will create a `wmin-model-dev` environment installed in development mode.
If you want to use a different environment name you can run:

```bash
conda env create -n myenv -f environment.yml

```

## Usage

The wmin-model code can be mainly used for two applications

- Generating a POD basis set -> ...
- Running a POD fit -> ...