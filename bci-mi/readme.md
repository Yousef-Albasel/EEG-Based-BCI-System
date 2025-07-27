# Instructions to run the environment

A requirement.txt and dependencies folder were provided, the trainning was done in the same number of epochs as provided in main.py, all checkpoints and saved data were also provided, all you have to do for inference is running the code, specifying your path in config, additional information is about the python version is 3.13.1 and torch version being 2.6.0 but the trainning was using the cuda version cu118

python version used is python 3.13.1
Torch version: 2.6.0+cu118

Recommended steps are:

1. Create a Virtual Environment
On Windows:
```bash
python -m venv .venv
.venv\Scripts\activate
```
2. Install dependencies

using 

```bash
pip install --no-index --find-links=dependencies -r requirements.txt
```
or 

`pip install -r requirements.txt`

as for the Cuda version of pytorch 

`pip install torch==2.6.0+cu118 torchvision --extra-index-url https://download.pytorch.org/whl/cu118`


The inference functions and trainning functions are in main.py, i figured it would be easier to see the whole pipeline, comment out unneeded parts,
just running main.py will do the inference if specified the paths for test data
