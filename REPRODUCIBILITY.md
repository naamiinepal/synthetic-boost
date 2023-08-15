
# Reproducing the results

## Prerequisites
- Python version 3.10 (preferable).
- Download datasets as instructed below.
- Download pretrained checkpoints as instructed below.


You can download a copy of all the files in this repository by cloning the
[git](https://git-scm.com/) repository:

```sh
git clone https://github.com/naamiinepal/synthetic-boost
```

or downloading a zip archive from GitHub for the latest commit.

All source code for the paper are in the [src](src) folder and the related scripts are in [scripts](scripts) folder.
The [Jupyter](https://jupyter.org/)
notebooks used for the evalutation and visualization are in [notebooks](notebooks) folder.

The LaTeX sources for the manuscript text and figures are in [paper](paper) folder.

## Setting up your environment

### Setting up Python

You'll need a working Python 3 environment to run the code.
Although we used version 3.10 to produce the results in the paper, the code should work with any Python 3.8+ version.

### Setting up Virtual Environment

Assuming you have setup required python version, you can create a virtual environment using python's built-in `venv` module.

```sh
    python -m venv .venv
```

`python` is the path to your python executable.
You can find it by running `which python` or `which python3` in your terminal.

This will create a `.venv` folder in your current directory.
You can activate the virtual environment by running:

```sh
    source .venv/bin/activate
```

### Installing dependencies

Once you have activated the virtual environment, you can install the required dependencies by running:

```sh
    pip install -r requirements.txt
```

## Downloading the data

You need to download the data from the following sources:

- [CAMUS](http://humanheart-project.creatis.insa-lyon.fr/database/#collection/6373703d73e9f0047faa1bc8)
- [SDM CAMUS](https://zenodo.org/record/7921055#.ZFyqd9LMLmE)

## Downloading Models

### CLIPSeg

CLIPSeg are automatically downloaded when you run the scripts from [Huggingface Model Hub](https://huggingface.co/CIDAS/clipseg-rd64-refined).

### CRIS

#### Downloading CLIP

The Resnet 50 CLIP model needed for CRIS can be downloaded from [here](https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt) using the following command.

```sh
    wget https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt -O pretrain/RN50.pt
```

#### Downloading CRIS

We have not yet found the official link to download the CRIS model.
We have used the model from [CRIS's repo's issue](https://github.com/DerrickWang005/CRIS.pytorch/issues/3) using [this OneDrive link](https://polimi365-my.sharepoint.com/:f:/g/personal/10524166_polimi_it/Ej-lkQiFHU1ArDG68PP-u3kBJL_UBvvn1scRU7Ps5fiIOw?e=KzFowg).

#### Converting CRIS Model

The downloaded CRIS model needs to be loaded in `DataParallel` mode.
To convert the model, run the following command after saving the downloaded model to `pretrain/cris_best.pt`:

```sh
    python scripts/convert_cris_model.py
```

## Training VLSMs

### CAMUS
Finetune the VLSMs with CAMUS dataset:
```sh
    bash scripts/camus.sh
```

### SDM CAMUS
Finetune the VLSMs with SDM CAMUS dataset:
```sh
    bash scripts/sdm_camus.sh
```

### PT-FT Strategy
Finetune the VLSMs with CAMUS dataset, already finetuned on SDM CAMUS dataset:
```sh
    bash scripts/pt_ft.sh
```