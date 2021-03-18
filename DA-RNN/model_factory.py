#Reference: PA4 starter code provided in CSE 251B Winter 2021, https://github.com/Zhenye-Na/DA-RNN/blob/master/src/da_rnn.ipynb, http://chandlerzuo.github.io/blog/2017/11/darnn

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

        self.T = 60 #max_frame
        self.encoder_attn = nn.Linear(
            in_features=2 * self.hidden_size + self.T, #2*1024+60 
            out_features=1
        ) #2108x1
        
#         self.encoder_attn = nn.Sequential(
#             nn.Linear(2 * self.hidden_size + self.T, self.hidden_size),
#             nn.Tanh(),
#             nn.Linear(self.hidden_size, 1)
#         ) -> Giving CUDA out of memory error within an epoch

    def forward(self, video_seq,captions):
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

        # input_data: batch_size * T * input_size
        for t in range(self.T):
            torch.cuda.empty_cache()
            #print("Encoding with attention at time step ",t)
            # batch_size * input_size * (2 * hidden_size + T - 1)
            x = torch.cat((h_n.repeat(self.input_size, 1, 1).permute(1, 0, 2),s_n.repeat(self.input_size, 1, 1).permute(1, 0, 2),
                           video_seq.permute(0, 2, 1)), dim=2)  #torch.Size([32, 512, 2108])
#             #print("x.shape before passing to attention layer = ",x.shape)
#             #print("shape of x.view going to attention layer = ",x.view(-1, self.hidden_size * 2 + self.T).shape) #torch.Size([16384, 2108])
            x = self.encoder_attn(x.view(-1, self.hidden_size * 2 + self.T))

#           # get weights by softmax
            alpha = F.softmax(x.view(-1, self.input_size),dim=1)

#             # get new input for LSTM
            x_tilde = torch.mul(alpha, video_seq[:, t, :]) 
#           #print("Shape of x_tilde.unsqueeze(0), input to lstm1 = ",x_tilde.unsqueeze(0).shape) #(seq_len, batch, input_size)
            _, final_state = self.lstm1(x_tilde.unsqueeze(0), (h_n, s_n)) 
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
        
        total_time_steps = self.T + caplen
        #print("Total time steps in encoder = ",total_time_steps)
        for t in range(0,caplen):
            #print("Encoding without attention at time step ",t)
            _, (h_n,s_n) = self.lstm1(padding_input[t,:,:].unsqueeze(0),(h_n,s_n)) #32x102/103/104x512
            hidden_enc.append(h_n.squeeze(0)) 
        
        hidden_enc = torch.stack(hidden_enc, 1) 
        #print("shape of final hidden layer from encoder",hidden_enc.shape) #torch.Size([32, 82, 1024])
        return hidden_enc 

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
        
        self.attn_layer = nn.Sequential(
                nn.Linear(2 * self.hidden_size + self.hidden_size, self.hidden_size),
                nn.Tanh(), #replacing with ReLU didn't improve
                nn.Linear(self.hidden_size, 1)
            )

    def forward(self, feature, captions): 
        d_n = torch.zeros(1, feature.size(0), self.hidden_size).cuda() #1xbatchx1024
        c_n = torch.zeros(1, feature.size(0), self.hidden_size).cuda() #1xbatchx1024
        lstm_outputs = []
        embedded = self.embedding(captions) #32x22/23/24x512
        pad_token = torch.zeros(embedded.shape[0],2,embedded.shape[2]).cuda() #32x1x512
        #print("Feature.shape = ",feature.shape)
        #print("d_n.shape = ",d_n.shape)
        X_encoded = feature[:,:self.T,:]
        #print("shape of attention encoded x = ",X_encoded.shape)
        for t in range(self.T): 
            #print("Decoding frame at time step ",t)
            x = torch.cat((d_n.repeat(self.T, 1, 1).permute(1, 0, 2),
                   c_n.repeat(self.T, 1, 1).permute(1, 0, 2),
                   X_encoded), dim=2)
            
            beta = F.softmax(self.attn_layer(x.view(-1, 2 * self.hidden_size + self.hidden_size)).view(-1, self.T),dim=1)
            context = torch.bmm(beta.unsqueeze(1), X_encoded)
            #print("Context shape = ",context.shape)
            y_tilde = torch.cat((context, X_encoded[:, t,:].unsqueeze(1)), dim=1)
            #print("y_tilde shape = ",y_tilde.shape)
            lstm_input = torch.cat((pad_token,y_tilde),dim=2) #stacking pad + hidden from encoder vertically
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
        pad_token = torch.zeros(embedded.shape[0],2,embedded.shape[2]).cuda() #32x2x512
        X_encoded = feature[:,:self.T,:]
        #print("shape of attention encoded x = ",X_encoded.shape)
        for t in range(self.T): 
            #print("Decoding frame at time step ",t)
            x = torch.cat((d_n.repeat(self.T, 1, 1).permute(1, 0, 2),
                   c_n.repeat(self.T, 1, 1).permute(1, 0, 2),
                   X_encoded), dim=2)
            
            beta = F.softmax(self.attn_layer(x.view(-1, 2 * self.hidden_size + self.hidden_size)).view(-1, self.T),dim=1)
            context = torch.bmm(beta.unsqueeze(1), X_encoded)
            #print("Context shape = ",context.shape)
            y_tilde = torch.cat((context, X_encoded[:, t,:].unsqueeze(1)), dim=1)
            #print("y_tilde shape = ",y_tilde.shape)
            lstm_input = torch.cat((pad_token,y_tilde),dim=2) #stacking pad + hidden from encoder vertically
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
    hidden_size = 512
    embedding_size = 512
    maxword_caption=20
    
    vocab_size = len(vocab)
    #print("vocab size is:",vocab_size)
    LSTM_encoder = encoder(hidden_size)
    LSTM_decoder = decoder( hidden_size, vocab_size,max_frame,embedding_size)
    return LSTM_encoder, LSTM_decoder
    