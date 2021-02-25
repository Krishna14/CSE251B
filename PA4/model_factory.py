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
    def __init__(self, embed_size, hidden_size, vocab_size, experiment_name, max_caption_count, num_layers=1):
        super(decoder, self).__init__()
        
        self.embedding_layer = nn.Embedding(vocab_size, embed_size)
        
        if(experiment_name != "vanilla_rnn"):
            self.sequence_model = nn.LSTM(input_size = embed_size,hidden_size = hidden_size, num_layers = num_layers, batch_first = True)
        else:
            # TODO: Tri please fill this function
            self.sequence_model = nn.RNN(input_size = embed_size,hidden_size = hidden_size, num_layers = num_layers, batch_first = True)

        self.linear = nn.Linear(hidden_size, vocab_size)
        self.max_length = max_caption_count
    
    def forward(self, features, captions, lengths):
        #captions = captions[:, :-1]
        embed = self.embedding_layer(captions)
        embed = torch.cat((features.unsqueeze(1), embed), dim = 1)
        packed = pack_padded_sequence(embed, lengths, batch_first=True)
        lstm_outputs, _ = self.sequence_model(packed)
        outputs = self.linear(lstm_outputs[0])
        
        return outputs
    
    def generate_captions(self, features, states=None):
        """
            Given image features, generate captions based on the history
        """
        # Unsqueeze the features and convert it to generate the appropriate captions here
        caption = []
        features = features.unsqueeze(1)
        for i in range(self.max_length):
            # We pass the features through the sequence model
            lstm_outputs, states = self.sequence_model(features, states)
            lstm_outputs = lstm_outputs.squeeze(1)
            output = self.linear(lstm_outputs)
            #print("Shape of output = {}".format(output.shape))
            # Pick the value that's performing the best (This is only true for the deterministic case)
            # In the stochastic case, we will have a softmax and argmax after that
            last_pick = output.max(1)[1]
            caption.append(last_pick)
            features = self.embedding_layer(last_pick).unsqueeze(1)
        caption = torch.stack(caption, 1)
        caption = caption.cpu().numpy()
        return caption
    
def get_model(config_data, vocab):
    hidden_size = config_data['model']['hidden_size']
    embedding_size = config_data['model']['embedding_size']
    model_type = config_data['model']['model_type']
    
    # We need to modify this
    experiment_name = config_data['experiment_name']
    
    # 
    vocab_threshold = config_data['dataset']['vocabulary_threshold']
    trainingAnnotationFile = config_data['dataset']['training_annotation_file_path']
    
    # The type of the experiment and the maximum lengths have been provided
    experiment_type = "stochastic" if (config_data['generation']['deterministic'] == "false") else "deterministic"
    
    # Required for the captions
    max_length_captions = config_data['generation']['max_length']
    
    # Here, we compute the name of the experiment
    experiment_name = experiment_name + '_' + str(experiment_type)
    print("Experiment name = {}".format(experiment_name))
    
    # Load the vocab
    savedVocab = load_vocab(trainingAnnotationFile, vocab_threshold)
    vocab_size = len(savedVocab)
    
    # Check for experiment_name as "baseline_deterministic"
    if(experiment_name == "baseline_deterministic"):
        # Here, we have the Convolutional Neural Network Encoder
        CNN_encoder = encoder(experiment_name, embedding_size)
        LSTM_decoder = decoder(embedding_size, hidden_size, vocab_size, experiment_name, max_length_captions)
        return CNN_encoder, LSTM_decoder
    elif (experiment_name == "baseline_stochastic"):
        raise NotImplementedError("{} Not Implemented".format(experiment_name))
    elif (experiment_name == "vanilla_rnn"):
        raise NotImplementedError("{} Not Implemented".format(experiment_name))
    elif (experiment_name == "final_experiment"):
        raise NotImplementedError("{} Not Implemented".format(experiment_name))
    else:
        raise NotImplementedError("{} wrong experiment name".format(experiment_name))
