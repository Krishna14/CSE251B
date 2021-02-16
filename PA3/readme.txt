# README

1. Baseline
In order to run the baseline, just execute starter.py after entering into the baseline directory:

cd 3 baseline 
python starter.py

Hyperparameters such as learning rate, batch size, etc. can be tuned if needed in starter.py 
The resulting loss plot, segmentation overlayed on first test image and the classwise iou of validation set (in csv format) will get generated and saved in the same directory. 

2. Data Augmentation
To run this experiment, go inside the corresponding directory and execute starter.py

cd 4a augmentation
python starter.py

Hyperparameters such as learning rate, batch size, etc. can be tuned if needed in starter.py 
The resulting loss plot, segmentation overlayed on first test image and the classwise iou of validation set (in csv format) will get generated and saved in the same directory. 

3. Class Imbalance
To run this experiment, go inside the 4b class imbalance directory and execute starter.py

cd 4b class imbalance 
python starter.py

Hyperparameters such as learning rate, batch size, etc. can be tuned if needed in starter.py 
The resulting loss plot, segmentation overlayed on first test image and the classwise iou of validation set (in csv format) will get generated and saved in the same directory. 

5a - There are two models that we have developed. ResNet and ResUNet. Please go into each of the directories to test the performance of each of these models

cd 5aResNet/
python3 main.py

cd -
cd 5aResUNet/
python3 main.py

To run each of the experiments for 5a, please refer to the README.md files within 5aResNet and 5aResUNet folders
The README.md files inside 5aResNet and 5aResUNet provide the steps to run the code for part 5(a)


5. Transfer Learning
To run this experiment, go insde the 5b directory and execute transfer_learning(1).py

cd 5b
python transfer_learning(1).py

Hyperparameters such as learning rate, batch size, etc. can be tuned if needed in starter.py 
The resulting loss plot, segmentation overlayed on first test image and the classwise iou of validation set (in csv format) will get generated and saved in the same directory. 
 
6. U-net
To run this experiment, go inside the 5c directory and execute train_unet.py

cd 5c\ unet
python train_unet.py

Hyperparameters such as learning rate, batch size, etc. can be tuned if needed in starter.py 
The resulting loss plot, segmentation overlayed on first test image and the classwise iou of validation set (in csv format) will get generated and saved in the same directory. 
