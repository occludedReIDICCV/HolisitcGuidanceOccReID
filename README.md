# HolisitcGuidanceOccReID
Code for ICCV submission Holistic Guidance


    #Installation
    conda create --name holistic-reid python=3.7
    conda activate holistic-reid

    # install dependencies
    pip install -r requirements.txt

    # install torch and torchvision (select the proper cuda version to suit your machine)
    conda install pytorch torchvision cudatoolkit=9.0 -c pytorch

    # install torchreid (don't need to re-build it if you modify the source code)
    python setup.py develop
    
Steps to reproduce results:
1. Create data directory paths and add data
Under root create a directory reid-data Add data Occluded-Duke, Occluded ReID data from 
https://github.com/lightas/Occluded-DukeMTMC-Dataset, Follow instructions
Download market1501 data and follow instructions from 
https://github.com/lightas/Occluded-DukeMTMC-Dataset

2. Train backbone with 
source_training.py

3. Use saved model from 2. in the appropriate log folder to train Teacher -Student model using 
occluded_reid_train.py

Running "occluded_reid_train.py" will train and validate to produce the results required for the given data set. Dataset define in the Key Word "target=" in occluded_reid_train.py

