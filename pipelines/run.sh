#!/bin/bash
poetry env use python
poetry run python pipelines/qclinear.py pipelines/config.yml
