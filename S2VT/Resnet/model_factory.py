#Reference: PA4 starter code provided in CSE 251B Winter 2021, Tensorflow implementatation: https://github.com/vijayvee/video-captioning/blob/9dfd6608a520adbd94c97b8e8e8ade9e7c3536b8/training_vidcap.py

import torch.nn as nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence
import torchvision.models as models
from torch.autograd import Variable
import torch.nn.functional as F

class encoder(nn.Module):
    """
        Defines the encoder 
    """
    def __init__(self, hidden_size, n_layers=1):
        super(encoder, self).__init__()
        self.hidden_size = hidden_size # hidden_size: dimension of the hidden state
        self.input_size = 512 ## input size: number of underlying factors 
        
        self.enc_linear = nn.Linear(2048,512)# hidden_size) # 32x80x2048 -> 32x80x512
        self.batchNorm = nn.BatchNorm1d(512, momentum=0.01)
        self.lstm1 = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=n_layers)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, video_seq,captions,max_frame):
        """
           forward pass computation
        """
        video_seq = self.dropout(video_seq)
        video_seq = self.enc_linear(video_seq) #32x80x512
        video_seq = video_seq.permute(0,2,1) #32x512x80
        video_seq = self.batchNorm(video_seq) 
        video_seq = video_seq.permute(0,2,1) #32x60x512 
        batch_size = video_seq.shape[0]
        #print("shape of video_seq for encoder input:", video_seq.shape) #torch.Size([32, 60, 512])
        
        h_n = torch.zeros(1,video_seq.shape[0],self.hidden_size).cuda() #1x32x1024
        s_n = torch.zeros(1,video_seq.shape[0],self.hidden_size).cuda() #1x32x1024
        hidden_enc = []

        video_seq = video_seq.permute(1,0,2) #-> 80x32x512
        for t in range(max_frame):
            _, final_state = self.lstm1(video_seq[t,:,:].unsqueeze(0), (h_n, s_n)) #1x32x512
            h_n = final_state[0]
            s_n = final_state[1]
            #print("h_n output shape = ",h_n.shape) #torch.Size([1, 32, 1024])
            hidden_enc.append(h_n.squeeze(0)) 
        
        padding_input = torch.zeros(captions.shape[1], batch_size, video_seq.shape[2]).cuda() #removing batch_first, caplenx32x512
        #video_seq = video_seq.permute(1,0,2)
        #print("Video seq shape before padding = ",video_seq.shape) #torch.Size([60, 32, 512])
        #lstm1_input = torch.cat((video_seq, padding_lstm1), 0) #removing batch_first,(80+caplen)x32x512
        caplen = padding_input.shape[0]
        # print("Shape of lstm1_input = ",lstm1_input.shape) #torch.Size([84, 32, 512])
        # print("Shape of lstm1_input per time step = ",lstm1_input[t,:,:].unsqueeze(0).shape) #1x32x512
        
        total_time_steps = max_frame + caplen
        #print("Total time steps in encoder = ",total_time_steps)
        for t in range(0,caplen):
            #print("Encoding at time step ",t)
            _, (h_n,s_n) = self.lstm1(padding_input[t,:,:].unsqueeze(0),(h_n,s_n)) #32x102/103/104x512
            hidden_enc.append(h_n.squeeze(0)) 
        
        hidden_enc = torch.stack(hidden_enc, 1) 
        #print("shape of final hidden layer from encoder",hidden_enc.shape) #torch.Size([32, 82, 1024])
        return hidden_enc 

        #return hidden1 from encoder after each frame . This should be sent right then to decoder. 
        # This is a stacked LSTM architecture As per Subhashini

class decoder(nn.Module):
    """
        Decoder implementation
    """
    def __init__(self, hidden_size, vocab_size, max_frame,embedding_size, n_layers=1):
         
        super(decoder, self).__init__() 
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.max_frame = max_frame
        self.T = max_frame
       
        self.embedding = nn.Embedding(vocab_size, self.embedding_size) 
        self.lstm2 = nn.LSTM(self.embedding_size + self.hidden_size, self.hidden_size, n_layers, batch_first = True)
        self.linear = nn.Linear(self.hidden_size, vocab_size)
        self.softmax = nn.Softmax(dim=2)
        self.dropout = nn.Dropout(p=0.5)

        # need to take the hidden 1x32x512 and pad it with <pad> for video frames 
        # this is the input to LSTM. So input will be      
    def forward(self, feature, captions): 
        d_n = torch.zeros(1, feature.size(0), self.hidden_size).cuda() #1xbatchx1024
        c_n = torch.zeros(1, feature.size(0), self.hidden_size).cuda() #1xbatchx1024
        lstm_outputs = []
        embedded = self.embedding(captions) #32x22/23/24x512
        pad_token = torch.zeros(embedded.shape[0],1,embedded.shape[2]).cuda()
        for t in range(self.max_frame): 
            #print("Decoding frame at time step ",t)
            lstm_input = torch.cat((pad_token,feature[:,t,:].unsqueeze(1)),dim=2) #stacking pad + hidden from encoder vertically
            #print("Decoding frame, LSTM input shape = ",lstm_input.shape)
            _,final_state = self.lstm2(lstm_input,(d_n,c_n))
            d_n = final_state[0]
            c_n = final_state[1]

        caplen = embedded.shape[1] 
        total_time_steps = self.max_frame + caplen
        #print("Total time steps in decoder = ",total_time_steps)
        t = t + 1
        for i in range(0,caplen):      
            #print("Decoding caption at time step ",t)
            lstm_input = torch.cat((embedded[:,i,:].unsqueeze(1),feature[:,t,:].unsqueeze(1)),dim=2) #stacking current caption + hidden from encoder vertically
            #print("Decoding caption, LSTM input shape = ",lstm_input.shape)
            output,final_state = self.lstm2(lstm_input,(d_n,c_n))
            d_n = final_state[0]
            c_n = final_state[1]
            #out = self.dropout(output)
            out = self.linear(output) 
            #print("Shape of decoder out ",out.shape) #32x1xhidden #[32, 1, 12597]
            lstm_outputs.append(out.squeeze(1)) #32x12597
            t = t + 1

        lstm_outputs = torch.stack(lstm_outputs,1) #32xcaplenx12597
        lstm_outputs = lstm_outputs[:,:-1,:] #32xcaplen-1x12597 #ignoring <end>'s decoded output
        #print("Output shape from decoder before reshape = ",lstm_outputs.shape)
        lstm_outputs = torch.reshape(lstm_outputs, (lstm_outputs.shape[0]*lstm_outputs.shape[1],lstm_outputs.shape[2]))
        #print("Output shape from decoder before return = ",lstm_outputs.shape) 
        return lstm_outputs
    
    def generate_captions_deterministic(self, feature,max_count,batch_size,states=None):
        d_n = torch.zeros(1, feature.size(0), self.hidden_size).cuda() #1xbatchx1024
        c_n = torch.zeros(1, feature.size(0), self.hidden_size).cuda() #1xbatchx1024
        generated_captions = []
        start_token = torch.ones(batch_size).long().cuda() #32
        generated_captions.append(start_token)
        embedded = self.embedding(start_token.unsqueeze(1)) #32x1x512
        pad_token = torch.zeros(embedded.shape[0],1,embedded.shape[2]).cuda()
        for t in range(self.max_frame):
            #print("Generating caption, ignoring frame at time step ",t)
            lstm_input = torch.cat((pad_token,feature[:,t,:].unsqueeze(1)),dim=2) #stacking pad + hidden from encoder vertically
            #print("Decoding frame, LSTM input shape = ",lstm_input.shape)
            _,final_state = self.lstm2(lstm_input,(d_n,c_n))
            d_n = final_state[0]
            c_n = final_state[1]

        t = t + 1
        for i in range(0,max_count-1): #<start> + (max_count - 1) tokens = max_count tokens
            #print("Generating caption, one word at time step ",t)
            lstm_input = torch.cat((embedded,feature[:,t,:].unsqueeze(1)),dim=2) #stacking current caption + hidden from encoder vertically
            #print("Decoding caption, LSTM input shape = ",lstm_input.shape)
            output,final_state = self.lstm2(lstm_input,(d_n,c_n))
            d_n = final_state[0]
            c_n = final_state[1]
            out = self.linear(output) 
            #print("Shape of decoder out ",out.shape) 
            output_token = out.max(2)[1] #32x1
            generated_captions.append(output_token.squeeze(1)) #32
            embedded = self.embedding(output_token)#32x1x512
            t = t + 1
        generated_captions = torch.stack(generated_captions,1) #32xmax_countx512
        generated_captions = generated_captions.cpu().numpy()
        return generated_captions

def get_model(vocab,max_frame):
    hidden_size = 1024
    embedding_size = 512
    maxword_caption=20
    
    vocab_size = len(vocab)
    #print("vocab size is:",vocab_size)
    LSTM_encoder = encoder(hidden_size)
    LSTM_decoder = decoder( hidden_size, vocab_size,max_frame,embedding_size)
    return LSTM_encoder, LSTM_decoder
    