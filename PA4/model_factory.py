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
import numpy


class encoder(nn.Module):
    """
        Defines the encoder for the image captioning task
    """
    # TODO: Check the number of classes for the output of the linear layer
    def __init__(self, experiment_name, hidden_size):
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
        self.linear = nn.Linear(res50_model.fc.in_features, hidden_size)
        # Decide on the feature sizes
        self.batchNorm = nn.BatchNorm1d(hidden_size, momentum=0.01) 
        #We can change momentum and see what happens, if there is time. 
    
    def forward(self, x):
        """
           forward pass computation
        """
        #print('shape of input to forward',x.size())
        with torch.no_grad():
            x1 = self.resnet50_model(x)
        #print('shape of output from resnet',x1.size())
        x1 = x1.reshape(x1.size(0), -1)
        #Trainable 
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
        
        self.hidden_size = hidden_size
        self.experiment_name = experiment_name
        
        if(experiment_name != "vanilla_rnn"):
            self.sequence_model = nn.LSTM(input_size = embed_size,hidden_size = hidden_size, num_layers = num_layers, batch_first = True)
        else:
            # TODO: Tri please fill this function
            self.sequence_model = nn.RNN(input_size = embed_size,hidden_size = hidden_size, num_layers = num_layers, batch_first = True)

        self.softmax = nn.Softmax(dim=1)
        self.linear = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, features, captions, lengths): 
        lstm_hidden = features.unsqueeze(0) #torch.Size([1,64,512])
        lstm_input = self.embedding_layer(captions) #torch.Size([ caption_length,64, 300])
        
        if(self.experiment_name != "vanilla_rnn"):
            states = (lstm_hidden,lstm_hidden)
        else:
            states = lstm_hidden
            
        lstm_outputs, states = self.sequence_model(lstm_input,states) 
        outputs = self.linear(lstm_outputs)[:,:-1]
        outs = self.softmax(outputs)
        outputs = torch.reshape(outputs, (outputs.shape[0]*outputs.shape[1],outputs.shape[2]))
        max_outputs = outs.max(2)[1]
        return outputs,max_outputs[:,:-1] #outputs_lengths
    
    def generate_captions_deterministic(self, features,max_count,states=None):
        #print(features.shape)
        lstm_hidden = features.unsqueeze(0)
        generated_caption = []
        if(self.experiment_name != "vanilla_rnn"):
            states = (lstm_hidden,lstm_hidden)
        else:
            states = lstm_hidden
        
        captions = torch.ones(features.size(0)).long().cuda() #<start>
        generated_caption.append(captions) #<start>
        captions = captions.unsqueeze(0)
        captions = captions.reshape(captions.size(1), -1)
        captions = self.embedding_layer(captions)
        for i in range(max_count-1): # caption of maximum length 56 seen in training set
            lstm_outputs, states = self.sequence_model(captions,states)
            out = self.linear(lstm_outputs)
            max_output = out.max(2)[1] #Take the maximum output at each step.
            generated_caption.append(max_output.squeeze(1)) 
            captions = max_output#.unsqueeze(1)
            captions = self.embedding_layer(captions)#.unsqueeze(0)
        generated_caption = torch.stack(generated_caption, 1) 
        generated_caption = generated_caption.cpu().numpy()#numpy.asarray(generated_caption)
        return generated_caption
    
    def generate_captions_stochastic(self, features, temperature, max_count, states=None):
        # takes the features from encoder and generates captions
        lstm_hidden = features.unsqueeze(0)
        generated_caption = []
        if(self.experiment_name != "vanilla_rnn"):
            states = (lstm_hidden,lstm_hidden)
        else:
            states = lstm_hidden
        captions = torch.ones(features.size(0)).long().cuda() #<start>
        generated_caption.append(captions) #<start>
        captions = captions.unsqueeze(0)
        captions = captions.reshape(captions.size(1), -1)
        captions = self.embedding_layer(captions)
        for i in range(max_count-1): # caption of maximum length 56 seen in training set
            lstm_outputs, states = self.sequence_model(captions,states)
            out = self.linear(lstm_outputs) / temperature
            out = nn.Softmax(dim=2)(out) 
            output = torch.multinomial(out.squeeze(1), 1)
            generated_caption.append(output.squeeze(1)) 
            captions = output
            captions = self.embedding_layer(captions)#.unsqueeze(0)
        generated_caption = torch.stack(generated_caption, 1) 
        generated_caption = generated_caption.cpu().numpy()#numpy.asarray(generated_caption)
        return generated_caption
       
    
def get_model(config_data, vocab):
    hidden_size = config_data['model']['hidden_size']
    embedding_size = config_data['model']['embedding_size']
    model_type = config_data['model']['model_type']
    experiment_name = config_data['experiment_name']
    vocab_threshold = config_data['dataset']['vocabulary_threshold']
    trainingAnnotationFile = config_data['dataset']['training_annotation_file_path']
    
    vocab_size = len(vocab)
    # Check for experiment_name as "baseline_deterministic"
    if(experiment_name == "baseline_deterministic"):
        # Here, we have the Convolutional Neural Network Encoder
        CNN_encoder = encoder(experiment_name, hidden_size)
        LSTM_decoder = decoder(embedding_size, hidden_size, vocab_size, experiment_name)
        return CNN_encoder, LSTM_decoder
    elif (experiment_name == "baseline_stochastic"):
        # Here, we have the Convolutional Neural Network Encoder
        CNN_encoder = encoder(experiment_name, embedding_size)
        LSTM_decoder = decoder(embedding_size, hidden_size, vocab_size, experiment_name)
        return CNN_encoder, LSTM_decoder
    elif (experiment_name == "vanilla_rnn"):
        CNN_encoder = encoder(experiment_name, hidden_size)
        rnn_decoder = decoder(embedding_size, hidden_size, vocab_size, experiment_name)
        return CNN_encoder, rnn_decoder
#     elif (experiment_name == "final_experiment"):
#         raise NotImplementedError("{} Not Implemented".format(experiment_name))
    else:
        CNN_encoder = encoder(experiment_name, hidden_size)
        LSTM_decoder = decoder(embedding_size, hidden_size, vocab_size, experiment_name)
        return CNN_encoder, LSTM_decoder
    
   