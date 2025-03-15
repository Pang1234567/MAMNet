### Requirements
python >= 3.9.7 
torch >= 2.0.1
torchaudio >= 2.0.2
torchvision >= 0.15.2

### Useage

--Folder "data":
Folder for storing datasets.

--Folder "Logs":
Folder for storing trained models.

--Folder "test_data":
Storage folder for small demonstration datasets.This folder contains 110 test images from the EndoVis2018 dataset and the results of running the model in 'Logs'. After downloading, you can directly use test. py to run the model in "Logs" and test it on the images in "test_data".

--train.py:
Code used for training networks.

--test.py:
Code for testing trained models.
