import torch

import pickle
import numpy as np
import pandas as pd
import os
from os.path import join
import glob
from glob import glob
from tqdm import tqdm

from Vocabulary import Vocabulary

import random

class DataLoader:
    def __init__(self, opt, train=True):
        print("Loading Data Loader instance for Train = {}...".format(train))
        
        # reproducible experiments
        np.random.seed(0)
        random.seed(0)

        self.device = opt.device
        self.video_descriptions_csv = opt.video_descriptions_csv
        self.video_descriptions_file = opt.video_descriptions_file
        self.vocab_file = opt.vocab_file
        self.video_features_file = opt.video_features_file
        self.batch_size = opt.batch_size
        
        self.vid_feat_size = opt.vid_feat_size
        # max sentences to include in the dataset. '0' to include all sentences
        self.max_sentence_num = 10
        self.max_words_num = 27
        
        self.num_train_set = opt.num_train_set
        
        print("Loading video descriptions...")
        self.video_descriptions = self.load_descriptions()
        
        print("Loading vocabulary...")
        self.vocab = self.load_vocabulary()
        
        train_test = list(self.video_descriptions.keys())
        random.shuffle(train_test)

        if train:
            self.names = train_test[:self.num_train_set]
        else:
            self.names = train_test[self.num_train_set:]
        
        print("Loading video features...")
        self.video_features = np.load(self.video_features_file)
        
        print("Data Loader initialized")
    
    def load_descriptions(self):
         return pickle.load(open(self.video_descriptions_file, 'rb'))
    
    def load_vocabulary(self):
        return pickle.load(open(self.vocab_file, 'rb'))
        
    def create_descriptions_from_csv(self):
        desc = pd.read_csv(self.video_descriptions_csv)
        desc = desc[(desc['Language'] == 'English')]
        desc = desc[['VideoID', 'Start', 'End', 'Description']]
        desc_dict = {}

        for row in desc.iterrows():
            key = str(row[1][0]) + '_' + str(row[1][1]) + '_' + str(row[1][2])
            if not os.path.exists("../video-summarization/data/" + key):
                continue
                
            if key in desc_dict:
                if self.max_sentence_num != 0 and len(desc_dict[key]) < self.max_sentence_num:
                    desc_dict[key].append(str(row[1][3]))
            else:
                desc_dict[key] = [str(row[1][3])]
            
        return desc_dict
    
    def create_full_vocab(self):
        # load coco vocabulary
        # vocab = pickle.load(open(coco_vocab_dir, 'rb'))
        vocab = Vocabulary()
        vocab.add_word('<pad>')
        vocab.add_word('<start>')
        vocab.add_word('<end>')
        vocab.add_word('<unk>')
        
        for key in self.video_descriptions:
            sentences = ' '.join(self.video_descriptions[key])
            
            for word in sentences.split(' '):
                filtered_word = word.lower().split('.')[0]
                if filtered_word not in vocab.word2idx:
                    vocab.add_word(filtered_word)
                    
        return vocab 

    def get_sentence_from_tensor(self, tensor):
        words_list = []
        np_array = tensor.data.cpu().numpy()
        for i in range(np_array.shape[0]):
            words_list.append(self.vocab.idx2word[np.argmax(np_array[i])])
        return words_list

    
    def get_words_from_index(self, tensor):
        words_list = []
        for idx in tensor.data.cpu().numpy():
            words_list.append(self.vocab.idx2word[idx])
        return words_list
    
    def get_one_hot_encoded_all(self, video_id):
        target = []
            
        for sentence in self.video_descriptions[video_id]:
            x = torch.zeros(len(self.vocab), dtype=torch.float32)
            for word in sentence.split(' '):
                filtered_word = word.lower().split('.')[0]
                x[self.vocab.word2idx[filtered_word]] = 1
                target.append(x)
            
        return torch.stack(target)
    
    def get_one_hot_encoded(self, video_id):
        descriptions = self.video_descriptions[video_id]
        index = np.random.randint(low=len(descriptions))
        
        target = torch.zeros(self.max_words_num, len(self.vocab))
        target[0, self.vocab.word2idx["<start>"]] = 1
        k = 1
        for word in descriptions[index].split(' '):    
            filtered_word = word.lower().split('.')[0]
            target[k, self.vocab.word2idx[filtered_word]] = 1
            k = k + 1
            if k == self.max_words_num-1:
                break
        target[k, self.vocab.word2idx["<end>"]] = 1    
        return target

    def encoded_sent(self, video_id):
        descriptions = self.video_descriptions[video_id]
        index = np.random.randint(low=len(descriptions))
        
        target = [self.vocab.word2idx["<end>"]] * self.max_words_num # pre-pad with <end>
        target[0] = self.vocab.word2idx["<start>"]
        for i, word in enumerate(descriptions[index].split(' ')):   
            filtered_word = word.lower().split('.')[0] 
            if i == self.max_words_num-2: # <start> ... <end>
                break
            target[i + 1] = self.vocab.word2idx[filtered_word]

        return torch.tensor(target, dtype = torch.long)
            
    def batch_data_generator(self):
        
        for i in tqdm(range(0, len(self.names), self.batch_size)):
            curr_batch_size = self.batch_size
            if (i + self.batch_size) > len(self.names):
                curr_batch_size = len(self.names) - i
                
            indexes = np.random.permutation(np.arange(i, i + curr_batch_size))
            max_seq_len = 0
            for i in range(len(indexes)):
                features = self.video_features[self.names[indexes[i]]]
                if features.shape[0] > max_seq_len:
                    max_seq_len = features.shape[0]
            
            x = torch.zeros(curr_batch_size, max_seq_len, self.vid_feat_size)
            y = []
            vid_names = []
            for i in range(len(indexes)):
                video_id = self.names[indexes[i]]
                features = self.video_features[video_id]
                x[i,:features.shape[0],:] = torch.from_numpy(features)
                # y.append(self.get_one_hot_encoded(video_id))
                y.append(self.encoded_sent(video_id))
                vid_names.append(video_id)
                
            yield x.to(self.device), torch.stack(y).to(self.device), vid_names