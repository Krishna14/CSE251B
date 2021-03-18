#Reference: PA4 starter code provided in CSE 251B Winter 2021
import os, pickle, json, csv, copy

# A simple wrapper class for Vocabulary. No changes are required in this file
class Vocabulary(object):
    """Simple vocabulary wrapper."""
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            print("word:{},id:{}".format(word,self.idx))
            self.idx += 1

    def __call__(self, word):
        if not word.lower() in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word.lower()]

    def __len__(self):
        return len(self.word2idx)

def load_vocab(data_path,labels):
    labels = data_path+'/'+labels
    if os.path.isfile(data_path+'/savedVocab'):
        with open(data_path+'/savedVocab', 'rb') as savedVocab:
            vocab = pickle.load(savedVocab)
            print("Using the saved vocab.")
    else:
        vocab = build_vocab(labels)
        with open(data_path+'/savedVocab', 'wb') as savedVocab:
            pickle.dump(vocab, savedVocab)
            print("Saved the vocab.")
    return vocab

def build_vocab(labels):
    vocab = Vocabulary()
    vocab.add_word('<pad>') #will get index 0
    vocab.add_word('<bos>') #index 1
    vocab.add_word('<eos>') #index 2
    vocab.add_word('<unk>') #index 3
    
    with open(labels,'r') as f:
        for line in f:
            word = line.strip()
            vocab.add_word(word)
        
    return vocab
