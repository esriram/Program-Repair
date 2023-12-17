#!/usr/bin/env bash

# A script that downloads files that are changed between buggy versions and successful versions of bugs in bugswarm

python3 download_projects.py

python3 download_patches.py

python3 parse_patch.py

