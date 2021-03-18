readme for dataloader of activitynet dataset

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