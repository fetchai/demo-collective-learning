An implementation of the collective learning scheme (one worker trains, the others vote on the update) in PySyft. To run the notebook with X-ray classification the data needs to be downloaded from Kaggle here: https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia. (This requires making a kaggle account). Unzip the downloaded data and put it in this directory.


- driver.py is the colearn demo (with MNIST data). It creates n workers, selects a random one to train, then coordinates the voting. 
- Collective_Learning_with_PySyft.ipynb is driver.py in notebook form
- driver_xray.py demonstrates colearn with X-ray data
- Collective_Learning_with_PySyft X-ray.ipynb is driver_xray.py in notebook form
