#!/bin/bash

cd pdes
python makedata.py
python train.py
cd ..
