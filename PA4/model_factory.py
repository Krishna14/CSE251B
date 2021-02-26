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
        
        if(experiment_name != "vanilla_rnn"):
            self.sequence_model = nn.LSTM(input_size = embed_size,hidden_size = hidden_size, num_layers = num_layers, batch_first = True)
        else:
            # TODO: Tri please fill this function
            self.sequence_model = nn.RNN()

        self.softmax = nn.Softmax(dim=1)
        self.linear = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, features, captions, lengths): 
        lstm_hidden = features.unsqueeze(0) #torch.Size([1,64,512])
        #print("lstm hidden:",lstm_hidden.shape)
        lstm_input = self.embedding_layer(captions) #torch.Size([64, caption_length, 300])
        #print("LSTM input before padding: ",lstm_input.shape)
        #DOUBT: Ask TA whether we have to send (h_0,c_0) separately looping. BLEU score is high. ???
        packed_input = pack_padded_sequence(lstm_input, lengths, batch_first=True)# to pack the embed captions according to decreasing length  
        #print("LSTM input after padding: {} and shape :{} ".format(packed_input.data,packed_input.data.shape))
        lstm_outputs, (h_n, c_n) = self.sequence_model(packed_input,(lstm_hidden,lstm_hidden)) 
        #print("lstm outputs packed shape = ",lstm_outputs.data.shape)
        #print(h_n.shape,c_n.shape)#torch.Size([1, 64, 512]) torch.Size([1, 64, 512])
    
        outputs = self.linear(lstm_outputs[0])
        #print("LSTM outputs shape = ",outputs.shape)
        outs = self.softmax(outputs)
        max_outputs = outs.max(1)[1]
        #outputs_lengths = [len(cap) for cap in outputs]
        return outputs[:-1],max_outputs[:-1] #outputs_lengths
    
    def generate_captions_deterministic(self, features,max_count,states=None):
        #print(features.shape)
        lstm_hidden = features.unsqueeze(0)
        generated_caption = []
        #print("lstm hidden:",lstm_hidden.shape)
        states = (lstm_hidden,lstm_hidden)
        captions = torch.ones(features.size(0)).long().cuda() #<start>
        generated_caption.append(captions) #<start>
        captions = captions.unsqueeze(0)
        captions = captions.reshape(captions.size(1), -1)
        #print("init captions shape before embed",captions.shape)
        captions = self.embedding_layer(captions)
        #print("init captions shape after embed",captions.shape)
       # features = features.unsqueeze(1)
        for i in range(max_count-1): # caption of maximum length 56 seen in training set
            lengths = [len(cap) for cap in captions]
            captions = pack_padded_sequence(captions, lengths, batch_first=True)# to pack the embed captions according to decreasing length  
            #print("i = {}, captions.batch = {}, states.shape = {}".format(i,captions.batch_sizes,states[0].shape))
            lstm_outputs, states = self.sequence_model(captions,states)
            #lstm_outputs = lstm_outputs.squeeze(0)
            #print('lstm output after squeeze',lstm_outputs.shape)
            output = self.linear(lstm_outputs[0])
            out = self.softmax(output)
            #print('shape of softmax output ',out.shape) 
            max_output = out.max(1)[1] #Take the maximum output at each step.
            #print('max output shape = ',max_output.shape)
            generated_caption.append(max_output) 
            captions = max_output.unsqueeze(1)
            #print('captions shape from max_output = ',captions.shape)
            #captions = captions.reshape(captions.size(1), -1)
            #print("captions shape before embed",captions.shape)
            captions = self.embedding_layer(captions)#.unsqueeze(0)
            #print("captions shape before exiting ",captions.shape)
            #print("i = {}, captions.batch = {}".format(i,captions.batch_sizes))
        generated_caption = torch.stack(generated_caption, 1) 
        #print('Caption of one batch shape is', generated_caption.size())
        generated_caption = generated_caption.cpu().numpy()
        return generated_caption
    
    def generate_captions_stochastic(self, features, temperature, max_count, states=None):
        # takes the features from encoder and generates captions
        caption = []
        features = features.unsqueeze(1)
        for i in range(max_count): # caption of maximum length 56 seen in training set
            lstm_outputs, states = self.sequence_model(features,states)
            lstm_outputs = lstm_outputs.squeeze(1)
            out = self.linear(lstm_outputs) / temperature
            out = self.softmax(out) #weighted softmax of the outputs
            sample = torch.rand(out.shape[0], 1).cuda()
            sums = torch.zeros(out.shape).cuda()
            sums[:, 0:1] = out[:, 0:1]
            for i in range(1, out.shape[1]):
                sums[:, i:i+1] = sums[:, i-1:i] + out[:, i:i+1]
            tmp = torch.arange(out.shape[1], 0, -1).cuda()
            tmp2 = torch.where(sample > sums, torch.tensor(1).cuda(), torch.tensor(-1).cuda())
            output = torch.argmin(tmp * tmp2, 1)
            caption.append(output)
            features = self.embedding_layer(output).unsqueeze(1)
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
        raise NotImplementedError("{} Not Implemented".format(experiment_name))
#     elif (experiment_name == "final_experiment"):
#         raise NotImplementedError("{} Not Implemented".format(experiment_name))
    else:
        raise NotImplementedError("{} wrong experiment name".format(experiment_name))
    
    #return 
