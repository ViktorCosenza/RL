#!/bin/bash

python -m pip install virtualenv
python -m virtualenv env
source ./env/bin/activate

python -m pip install torch==1.10.0+cpu torchvision==0.11.1+cpu torchaudio==0.10.0+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html
python -m pip install -r requirements.txt

python -m jupyter lab . 
