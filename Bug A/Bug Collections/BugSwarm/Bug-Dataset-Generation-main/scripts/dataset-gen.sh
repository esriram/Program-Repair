#!/usr/bin/env bash

python3 src/pretraining/dataset_gen.py --download_dataset True
python3 src/pretraining/dataset_gen.py --export_functions True
python3 src/pretraining/dataset_gen.py --augment_functions True
