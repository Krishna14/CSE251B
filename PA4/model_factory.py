################################################################################
# CSE 253: Programming Assignment 4
# Code snippet by Ajit Kumar, Savyasachi
# Fall 2020
################################################################################

# Build and return the model here based on the configuration.
import torch.nn as nn
import torch
from vocab import *
from torch.nn.utils.rnn import pack_padded_sequence
import torchvision.models as models


class encoder(nn.Module):
    """
        Defines the encoder for the image captioning task
    """
    # TODO: Check the number of classes for the output of the linear layer
    def __init__(self, experiment_name, embedding_size):
        """
            Initialize the experiment name
        """
        super(encoder, self).__init__()
        self.experiment_name = experiment_name
        res50_model = models.resnet50(pretrained=True)
        #get all the layers of resnet50
        layers = list(res50_model.children())
        # Removing the last layer 
        layers = layers[:-1]
        self.resnet50_model = nn.Sequential(*layers)
        #replacing the last layer with linear layer
        self.linear = nn.Linear(res50_model.fc.in_features, embedding_size)
        # Decide on the feature sizes
        self.batchNorm = nn.BatchNorm1d(embedding_size, momentum=0.01)
    
    def forward(self, x):
        """
           forward pass computation
        """
        #print('shape of input to forward',x.size())
        with torch.no_grad():
            x1 = self.resnet50_model(x)
        #print('shape of output from resnet',x1.size())
        x1 = x1.reshape(x1.size(0), -1)
        x1 = self.linear(x1)
        x1 = self.batchNorm(x1)
        return x1

class decoder(nn.Module):
    """
        Decoder implementation
    """
    def __init__(self, embed_size, hidden_size, vocab_size, experiment_name, num_layers=1):
        
        super(decoder, self).__init__()
        
        self.embedding_layer = nn.Embedding(vocab_size, embed_size)
        
        if(experiment_name != "vanilla_rnn"):
            self.sequence_model = nn.LSTM(input_size = embed_size,hidden_size = hidden_size, num_layers = num_layers, batch_first = True)
        else:
            # TODO: Tri please fill this function
            self.sequence_model = nn.RNN()

        self.linear = nn.Linear(hidden_size, vocab_size)
        #self.max_length = max_length
        
    
    def forward(self, features, captions, lengths):
        #captions = captions[:, :-1]
        embed = self.embedding_layer(captions)
        embed = torch.cat((features.unsqueeze(1), embed), dim = 1)
        packed = pack_padded_sequence(embed, lengths, batch_first=True)
        lstm_outputs, _ = self.sequence_model(packed)
        #print(lstm_outputs.shape, lstm_outputs[0].shape)
        outputs = self.linear(lstm_outputs[0])
        return outputs
    
    def generate_captions_deterministic(self, features,max_count = 20,states=None):
        # takes the features from encoder and generates captions
        caption = []
        features = features.unsqueeze(1)
        for i in range(max_count): # caption of maximum length 56 seen in training set
            lstm_outputs, states = self.sequence_model(features,states)
            lstm_outputs = lstm_outputs.squeeze(1)
            out = self.linear(lstm_outputs)
            #print('shape of linear output ',out.shape())
            max_output = out.max(1)[1]
            #print('max output = ,max_output')
            caption.append(max_output)
            features = self.embedding_layer(max_output).unsqueeze(1)
        #print('Caption of one batch shape is', caption.size())    
        caption = torch.stack(caption, 1) 
        #print('Caption of one batch shape is', caption.size())
        caption = caption.cpu().numpy()
        return caption
       
    
def get_model(config_data, vocab):
    hidden_size = config_data['model']['hidden_size']
    embedding_size = config_data['model']['embedding_size']
    model_type = config_data['model']['model_type']
    experiment_name = config_data['experiment_name']
    vocab_threshold = config_data['dataset']['vocabulary_threshold']
    trainingAnnotationFile = config_data['dataset']['training_annotation_file_path']
    
    # Load the vocab
    savedVocab = load_vocab(trainingAnnotationFile, vocab_threshold)
    vocab_size = len(savedVocab)
    # Check for experiment_name as "baseline_deterministic"
    if(experiment_name == "baseline_deterministic"):
        # Here, we have the Convolutional Neural Network Encoder
        CNN_encoder = encoder(experiment_name, embedding_size)
        LSTM_decoder = decoder(embedding_size, hidden_size, vocab_size, experiment_name)
        return CNN_encoder, LSTM_decoder
    elif (experiment_name == "baseline_stochastic"):
        raise NotImplementedError("{} Not Implemented".format(experiment_name))
    elif (experiment_name == "vanilla_rnn"):
        raise NotImplementedError("{} Not Implemented".format(experiment_name))
    elif (experiment_name == "final_experiment"):
        raise NotImplementedError("{} Not Implemented".format(experiment_name))
    else:
        raise NotImplementedError("{} wrong experiment name".format(experiment_name))
    
    #return 
