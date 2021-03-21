README for transformer model 

********************************************************************************************************************
Instructions for training

If you are not on datahub, first set up the dataset and dataloader using split_videos.ipynb

After, run every cell in train.ipynb

The logic for the training, validation, and test procedures are in train_utils.py

The model architectures are in transformer.py and transformer_mict.py

metadata is located in train.json, val_1.json, and val_2.json

vocabulary generation logic is in vocab.py, and saved vocab is in savedVocab binary file

evaluation metrics are located in caption_utils.py

********************************************************************************************************************
Instructions for dataloader of activitynet dataset (not necessary if you are already on ucsd datahub)

to generate the dataset from scratch on your own machine, do the following:

1. make sure there are "train" and "val" empty directories

2. run every cell in split_videos.ipynb, this will split all videos into frames, grabbing 100 frames per video. You can change the number of frames grabbed per video like so: 

directory = "/datasets/home/22/422/sramaswa/CSE251B/activityNetData/training"
root_destination = "train"
dim = 256
frames_to_collect = 10  <--- change this value to change the number of frames
build_data(directory, root_destination, dim, frames_to_collect)

finally, the notebook builds the vocab. However, the file should already be generated so you don't have to do this. 

3. run every cell in dataloader.ipynb, this will load the data using pytorch dataloader object.

********************************************************************************************************************
References/tutorials used

https://blog.floydhub.com/the-transformer-in-pytorch/
https://github.com/fmahoudeau/MiCT-Net-PyTorch
https://github.com/kenshohara/3D-ResNets-PyTorch
https://github.com/salesforce/densecap
https://github.com/SamLynnEvans/Transformer
http://proceedings.mlr.press/v95/chen18b/chen18b.pdf
http://activity-net.org/
http://activity-net.org/challenges/2017/captioning.html
https://cs.stanford.edu/people/ranjaykrishna/densevid/

