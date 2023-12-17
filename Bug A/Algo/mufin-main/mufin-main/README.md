# mscthesis

## Requirements

Required non-python packages:
`dos2unix`

To install the required python packages run:
```bash
pip install -r src/requirements.txt
```

## Setup Original Datasets

The original datasets are located in the [data](data) directory. They are registered as git submodules.

To retrieve them, you must init and update them:
```bash
git submodule init
git submodule update
```
