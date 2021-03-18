import h5py
import torch
import pickle
import numpy as np
import pandas as pd
import os
from os.path import join
import glob
from glob import glob

from Vocabulary import Vocabulary

class DataLoader:
    def __init__(self, opt, train=True):
        print("Loading Data Loader instance for Train = {}...".format(train))
        # Device has been assigned self.device = opt.device
        self.device = opt.device
        self.video_descriptions_file = opt.video_descriptions_file
        self.vocab_file = opt.vocab_file
        self.video_features_file = opt.video_features_file
        self.batch_size = opt.batch_size
        
        # Max sentences to include in the dataset.
        self.max_words_num = 80
        self.num_train_set = opt.num_train_set
        
        print("Loading video descriptions...")
        self.video_descriptions = self.load_descriptions()
        
        print("Loading vocabulary...")
        self.vocab = self.load_vocabulary()
        
        train_test = list(self.video_descriptions.keys())
        
        if train:
            self.names = train_test[:self.num_train_test]
        else:
            self.names = train_test[self.num_train_test:]
        
        print("Loading video features...")
        self.video_features = h5py.File(self.video_features_files, 'r')
        print("Found features for {} videos".format(len(self.video_features.keys())))
        
        print("Data loader initialized")
        
    def load_descriptions(self):
        return pickle.load(open(self.video_descriptions_file, 'rb'))
    
    def load_vocabulary(self):
        return pickle.load(open(self.vocab_file, 'rb'))
    
    def get_sentence_from_names(self, tensor):
        words_list = []
        np_array = tensor.data.cpu().numpy()
        for i in range(np.array.shape[0]):
            words_list.append(self.vocab.idx2word[np.argmax(np_array[i])])
        return words_list
    
    def get_one_hot_encoded(self, video_id):
        descriptions = self.video_descriptions[video_id]
        target = torch.zeros(self.max_words_num)
        target[0] = self.vocab.word2idx['<.>']
        k = 1
        for word in descriptions.split():
            filtered_word = word.lower()
            if filtered_word[-1] == '.':
                target[k] = self.vocab.word2idx[filtered_word.split('.')[0]]
                target[k+1] = self.vocab.word2idx['<.>']
                k += 2
            else:
                target[k] = self.vocab.word2idx[filtered_word]
                k += 1
            if k >= self.max_words_num - 2:
                break
        target[k] = self.vocab.word2idx["<end>"]
        return target
    
    def batch_data_generator(self):
        for i in range(0, len(self.names), self.batch_size):
            curr_batch_size = self.batch_size
            if (i + self.batch_size) > len(self.names):
                curr_batch_size = len(self.names) - i
                
            indexes = np.random.permutation(np.arange(i, i + curr_batch_size))
            # Determine max_seq_len for each batch?
            '''
            max_seq_len = 0
            for i in range(len(indexes)):
                features = self.video_features[self.names[indexes[i]]]["c3d_features"].value
                if features.shape[0] > max_seq_len:
                    max_seq_len = features.shape[0]
            '''
            
            max_seq_len = 300
            x = torch.zeros(curr_batch_size, max_seq_len, 500)
            y = []
            for i in range(len(indexes)):
                video_id = self.names[indexes[i]]
                features = self.video_features[video_id]["c3d_features"].value
                x[i,:features.shape[0],:] = torch.from_numpy(features)[:max_seq_len, :]
                y.append(self.get_one_hot_encoded(video_id))
                
            yield x.to(self.device), torch.stack(y).to(self.device)