#Reference: https://github.com/Peratham/video2text.pytorch/blob/0d530e271ee47ca4aa707dfec8d6c71801277b62/misc/data.py, #https://github.com/vijayvee/video-captioning/blob/9dfd6608a520adbd94c97b8e8e8ade9e7c3536b8/utils.py

#Pickle files and hdf5 files can be found here: https://drive.google.com/drive/folders/1oAAxKCs0ZjrUe7xbE6j-CqWhSubx3F1b?usp=sharing

import pickle
import h5py
import torch
import numpy as np
import torch.utils.data as data
from collections import defaultdict
from operator import itemgetter
class V2TDataset(data.Dataset):
    '''
    Video to Text dataset description class, used for loading and providing data.
    Support MSVD dataset
    Need the following input when constructing it:
    1. The pkl file containing text features
    2. The h5 file containing video frame information
    '''

    def __init__(self, cap_pkl, feature_h5):
        self.id2idx = defaultdict(list)
        with open(cap_pkl, 'rb') as f:
            self.pkl_captions, self.pkl_ids = pickle.load(f)
            print("length of self.captions:",len(self.pkl_captions))
            for idx in range(0,len(self.pkl_ids)):
                vid_id = self.pkl_ids[idx] #1300
                self.id2idx[vid_id+1].append(idx) #1300->[0,1,2,3,4] vid1300
              #print("Id from pickle file = {}, assigned id in id2idx as key = {}, assigned value(index) = {}".format(vid_id,vid_id+1,idx))
            #first = self.pkl_ids[0]
            #print("Pkl caption = {}, id = {}, len(captions)={}".format(self.pkl_captions[0:10],first,len(self.pkl_captions[0])))
        self.video_feats, self.video_ids,self.captions = self.get_video_feats(feature_h5) #filename = data_path+'/ResNet101_test.hdf5'

    def get_video_feats(self,filename):  
        results = defaultdict(list)
        result_ids = []
        result_captions = []
        sum = 0
        with h5py.File(filename, "r") as f:
            video_ids = list(f.keys())
            print("H5PY file length of video_ids = ",len(video_ids))
            for i in range(len(video_ids)):
                video_id = video_ids[i] #-4wsuPCjDBc_5_15 
                #print("H5PY file video_id = ",video_id)
                video_metadata = f[video_id][:]
                if "train" in filename:
                    start = 1
                elif "val" in filename:
                    start = 1201
                elif "test" in filename:
                    start = 1301
                else:
                #print("DATALOADER ERROR: Don't know what file you gave! Can't detect start index from filename that doesn't contain train/val/test keywords")
                    start = begin+1
                sum+=len(video_metadata)
              #print("{}. Video_id = {}, # frames = {},assigned_id = {}".format(i,video_id,len(video_metadata),start+i))
              #indices = [j for j in range(0,len(video_metadata),2)] # sampling every 5th frame as per 2015 paper (subhashini)
                results[start+i] = video_metadata#[indices]
                result_ids.append(start+i)
                caption_indices = self.id2idx[start+i]#[0:5] #[0,1,2,3,4]
                #print("caption indices:{}, type={}".format(caption_indices,type(caption_indices)))
                caps = list(itemgetter(*caption_indices)(self.pkl_captions))
                #caps.sort(key=lambda s: len(s),reverse=True)
                result_captions.append(caps[0:5])
        print("length of result_captions",len(result_captions))
        print("number of captions per video",len(result_captions[0])) 
        print("length of results after picking video metadata:",len(results))
        #print("length of metadata being put into results[pos]:",len(video_metadata[indices]))
        print("Average number of frames = ",sum/len(video_ids))
        return results,result_ids,result_captions

    def __getitem__(self, index):
        '''
        Return a sample pair (including video frame features)
        Find the corresponding video according to the caption
        So we need the videos are sorted by ascending order (id)
        '''
        caption = self.captions[index]
        #length = self.lengths[index]
        video_id = self.video_ids[index]
        video_feat = self.video_feats[video_id]#torch.from_numpy(self.video_feats[video_id])
        return video_feat, caption, video_id

    def __len__(self):
        return len(self.captions)

def collate_fn(data,num_frames=60,num_words=20):
    '''
    Combining multi-training samples to a mini-batch
    '''
    # Sort the videos according to video length
    data.sort(key=lambda x: x[-1], reverse=True)

    videos, captions, video_ids = zip(*data)

    # print("length of videos[0]{},videos[1]{},videos[2]{}".format(len(videos[0]),len(videos[1]),len(videos[2])))
    # print("length of captions",len(captions))
    # print("video_ids",video_ids)

    # Combing video together (make 2D Tensor video frames into one 3D Tensor)
    videos_stack = [None]*len(videos)
    #for i in range(len(videos)):
    for i, vids in enumerate(videos): 
      #print("index{} and videoID{}".format(i,video_ids[i]))
        videos_clipped = torch.zeros(num_frames,2048)
        lim = min(num_frames,len(vids))#len(videos[i]))
        videos_clipped[:lim,:] = torch.FloatTensor(vids[:lim,:]) #videos[i][:lim,:]
        videos_stack[i] = videos_clipped
    #print("video stack length is:{}, shape ={} ".format(len(videos_stack),len(videos_stack[0])))
    videos_stack = torch.stack(videos_stack, 0)
    #print("video stack shape is: ", videos_stack.shape)

    # Combine the captions together (make 1D Tensor words into on 2D Tensor)
    lengths = [len(cap[0]) for cap in captions]
    max_length = max(lengths)
    targets = torch.zeros(len(captions), max_length).long()
    cap_mask = []
    for i, cap in enumerate(captions):
        # print("type of cap{}, type of cap[0]{}, length cap{}, length of cap[0] {}".format(type(cap),type(cap[0]),len(cap),len(cap[0])))
        # print("caption ",cap[0])
        end = lengths[i]
        #cap.sort(key=lambda s: len(s),reverse=True)
        cap = cap[0] #picking the first caption out of 5 available captions per video, for training
        # print("cap in captions:",cap)
        targets[i, :end] = torch.FloatTensor(cap[:end])
        cap_mask.append([1.0]*end + [0.0]*(max_length-end))
    #captions = torch.stack(captions, 0)
    # print("shape of targets in collate_fn:",targets.shape)
    return videos_stack, targets, lengths, video_ids, torch.FloatTensor(cap_mask)

def get_data_loader(cap_pkl, feature_h5, batch_size=10, shuffle=False, num_workers=1, pin_memory=True,drop_last=True):
    v2t = V2TDataset(cap_pkl, feature_h5)
    data_loader = torch.utils.data.DataLoader(dataset=v2t,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn,
                                              pin_memory=pin_memory,
                                              drop_last = drop_last)
    return data_loader
