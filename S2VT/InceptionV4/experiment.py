#Reference: PA4 Starter code of CSE 251B Winter 2021
import matplotlib.pyplot as plt
import numpy as np
import torch
from datetime import datetime
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import torch.nn as nn
import torch.optim as optim
import json
from dataloader import *
from vocab import *
from model_factory import *
from caption_utils import *

data_path = '/home/a2raju/Project' 
# Class to encapsulate a neural experiment.
# The boilerplate code to setup the experiment, log stats, checkpoints and plotting
class Experiment(object):
    def __init__(self, name):
        self.batch_size = 32
        self.__train_loader = get_data_loader( data_path+'/train_captions.pkl', data_path+'/MSVD_InceptionV4.hdf5',0,1200,batch_size=self.batch_size,shuffle=True)
        self.__val_loader = get_data_loader( data_path+'/val_captions.pkl', data_path+'/MSVD_InceptionV4.hdf5',1200,100,batch_size=self.batch_size,shuffle=True)
        self.__test_loader = get_data_loader( data_path+'/test_captions.pkl', data_path+'/MSVD_InceptionV4.hdf5',1300,670,batch_size=self.batch_size,shuffle=False)

        self.__experiment_dir = data_path

        self.__vocab = load_vocab(data_path,'vocabulary.txt')
        self.__name = name
        # Setup Experiment
        self.__epochs = 100
        self.__early_stop_threshold = 5
        self.__max_caption_count = 20
        self.__learning_rate = 1e-4
        self.__max_frame = 60
        #self.__test_caption_path = config_data['dataset']['test_annotation_file_path']
       
        self.__train_caption_path = 'sents_train_lc_nopunc.txt'
        self.__current_epoch = 0
        self.__training_losses = []
        self.__val_losses = []
        self.__best_model = None  # Save your best model in this field and use this in test method.
        #
        self.__best_encoder_model = None  # Save the best encoder model here
        self.__best_decoder_model = None  # Save the best decoder model here

        # Init Model
        self.__encoder_model, self.__decoder_model = get_model(self.__vocab,self.__max_frame)
        # TODO: Set these Criterion and Optimizers Correctly
        self.__criterion = nn.CrossEntropyLoss()
        parameters = list(self.__decoder_model.parameters()) + list(self.__encoder_model.parameters())+list(self.__encoder_model.batchNorm.parameters()) 
        self.__optimizer = optim.Adam(parameters, lr=self.__learning_rate)
   
        self.__MODEL_NAME = self.__name + '_' + str(self.__learning_rate) + '_' + str(self.__epochs) + 'final'
        self.__use_gpu = False
        self.__init_model()

        # Load Experiment Data if available
        #self.__load_experiment()

    # Loads the experiment data if exists to resume training from last saved checkpoint.
    def __load_experiment(self):
        os.makedirs(ROOT_STATS_DIR, exist_ok=True)

        if os.path.exists(self.__experiment_dir):
            self.__training_losses = read_file_in_dir(self.__experiment_dir, 'training_losses.txt')
            self.__val_losses = read_file_in_dir(self.__experiment_dir, 'val_losses.txt')
            self.__current_epoch = len(self.__training_losses)

            state_dict = torch.load(os.path.join(self.__experiment_dir, 'latest_model.pt'))
            self.__encoder_model.load_state_dict(state_dict['encoder_model'])
            self.__decoder_model.load_state_dict(state_dict['decoder_model'])
            self.__optimizer.load_state_dict(state_dict['optimizer'])

        else:
            os.makedirs(self.__experiment_dir)

    def __init_weights(self,m):
        for param in m.parameters():
            if len(param.shape) >= 2:
                nn.init.kaiming_normal_(param.data)

    def __init_model(self):
        if torch.cuda.is_available():
            print("Using GPU")
            self.__use_gpu = True
            self.__encoder_model = self.__encoder_model.cuda().float()
            self.__decoder_model = self.__decoder_model.cuda().float()
            self.__criterion = self.__criterion.cuda()
            self.__init_weights(self.__encoder_model)
            self.__init_weights(self.__decoder_model)

    # Main method to run your experiment. Should be self-explanatory.
    def run(self):
        start_epoch = self.__current_epoch
        min_val_loss = float('inf')
        for epoch in range(start_epoch, self.__epochs):  # loop over the dataset multiple times
            print("EPOCH = ",epoch)
            start_time = datetime.now()
            self.__current_epoch = epoch
            train_loss = self.__train()
            val_loss = self.__val()
            # Saving the best model here
            if(val_loss < min_val_loss):
                print ("Saving best model after %d epochs." % (epoch))
                min_val_loss = val_loss
                self.__best_encoder_model = self.__encoder_model
                self.__best_decoder_model = self.__decoder_model
                self.__save_model('best_model'+self.__MODEL_NAME+'.pt')
            self.__record_stats(train_loss, val_loss)
            self.__log_epoch_stats(start_time)
            self.__save_model()
            #Early stop to prevent model from overfitting
            if epoch>=self.__early_stop_threshold:
                stop = 0
                for i in range(0,self.__early_stop_threshold):
                     if self.__val_losses[epoch-i] > self.__val_losses[epoch-i-1]:
                            stop = stop + 1
                if stop == self.__early_stop_threshold :
                    print ("EarlyStop after %d epochs." % (epoch))
                    break
        print("Experiment done!")

    # TODO: Perform one training iteration on the whole dataset and return loss value
    def __train(self):
        #print("Inside TRAIN")
        self.__encoder_model.train()
        self.__decoder_model.train()
        train_loss_batch = []

        for i, (videos, captions, lengths, video_ids,cap_mask) in enumerate(self.__train_loader):
            self.__optimizer.zero_grad()
            self.__max_caption_count = max(lengths) #just for visualization
            if self.__use_gpu:
                inputs = videos.cuda()
                train_labels = captions.cuda()#train_labels[0][:-1].cuda() #remove last <end> from inputs to decoder
                targets = captions[:,1:].long().cuda()#targets = targets[0][1:].long().cuda()#remove 1st <start> 
                # padding = Variable(torch.zeros(targets.shape[0], self.__max_frame)).long().cuda()
                # targets = torch.cat((padding, targets), 1)
                cap_mask = cap_mask[:, 1:].cuda()    
            else:
                inputs, train_labels, targets,cap_mask = videos, captions, captions[:,1:].long(),cap_mask[:,1:]

            #print("TRAIN: For iter {}, maxcaplen = {}".format(i,train_labels.shape[1]))
            features = self.__encoder_model(inputs,train_labels,self.__max_frame)
            #print("output shape from encoder:", features.shape) #[32, 60, 1024]
            
            pred_caption = self.__decoder_model.generate_captions_deterministic(features,self.__max_caption_count,self.batch_size) #for caption
            #print('{} in iter {}, Pred:{}, truth: {}'.format(video_ids[0],i,pred_caption[0],targets[0]))
            
            sentences = generate_text_caption(pred_caption,self.__vocab,self.__max_caption_count)
            truth = generate_text_caption(targets.cpu().numpy(),self.__vocab,self.__max_caption_count)

            if i%10 == 0:
                for num in range(0,5):
                    sentence = sentences[num]
                    print('TRAIN: EPOCH {}: Video_id = {}, iteration # {}, image # = {}, prediction: {}, truth: {}'.format(self.__current_epoch,video_ids[num],i,num,sentence,truth[num]))
            
            outputs = self.__decoder_model(features, train_labels)
            # outputs = outputs[:,self.__max_frame:,:]
            #print("shape of preds:",outputs.shape) caplenxvocabsize
            #print("train outputs in iter {}, outputs {}".format(i,outputs[0:3,:]))
            targets = torch.reshape(targets, (targets.shape[0]*targets.shape[1],1)).squeeze(1)
            #print("shape of targets:",targets.shape) #caplen
            #print("train target in iter {}, targets {}".format(i,targets[0:3]))
            logit_loss = self.__criterion(outputs, targets) 
            masked_loss = logit_loss*cap_mask
            loss = torch.sum(masked_loss)/torch.sum(cap_mask)
#             loss = self.__criterion(outputs, targets) 
#             loss = torch.unsqueeze(loss,0)
#             loss = loss.mean()
            train_loss_batch.append(loss.item())
            loss.backward()
            self.__optimizer.step()
        return np.mean(np.array(train_loss_batch))

    # TODO: Perform one Pass on the validation set and return loss value. You may also update your best model here.
    def __val(self):
        self.__encoder_model.eval()
        self.__decoder_model.eval()
        val_loss_batch = []
        with torch.no_grad():
            for i, (videos, captions, lengths, video_ids,cap_mask) in enumerate(self.__val_loader):
                self.__max_caption_count = max(lengths)
                if self.__use_gpu:
                    inputs = videos.cuda()
                    val_labels = captions.cuda()
                    #targets = targets[0].cuda()
                    targets = captions[:,1:].long().cuda()
                    #padding = Variable(torch.zeros(targets.shape[0], self.__max_frame)).long().cuda()
                    #targets = torch.cat((padding, targets), 1)
                    cap_mask = cap_mask[:, 1:].cuda()  
                else:
                    inputs, val_labels, targets,cap_mask = videos, captions, captions[:,1:].long(), cap_mask[:, 1:]

                #print("VAL: For iter {}, maxcaplen = {}".format(i,val_labels.shape[1]))
                features = self.__encoder_model(inputs,val_labels,self.__max_frame)
                #print("val target in iter {}, targets {}".format(i,targets[0,:]))
                pred_caption = self.__decoder_model.generate_captions_deterministic(features,self.__max_caption_count,self.batch_size) #for caption
                sentences = generate_text_caption(pred_caption,self.__vocab,self.__max_caption_count)
                truth = generate_text_caption(targets.cpu().numpy(),self.__vocab,self.__max_caption_count)

                if i%10 == 0:
                    for num in range(0,5):
                        sentence = sentences[num]
                        print('VAL: EPOCH {}: Video_id = {}, iteration # {}, image # = {}, prediction: {}, truth: {}'.format(self.__current_epoch,video_ids[num],i,num,sentence,truth[num]))
                
                targets = torch.reshape(targets, (targets.shape[0]*targets.shape[1],1)).squeeze(1)  
                outputs = self.__decoder_model(features, val_labels)
                #outputs = outputs[:,self.__max_frame:,:]
#                 loss = self.__criterion(outputs, targets)
#                 loss = torch.unsqueeze(loss,0)
#                 loss = loss.mean()
                logit_loss = self.__criterion(outputs, targets) 
                masked_loss = logit_loss*cap_mask
                loss = torch.sum(masked_loss)/torch.sum(cap_mask)
                val_loss_batch.append(loss.item())
            return np.mean(np.array(val_loss_batch))

    def test(self):
        print('Running test on best_model'+self.__MODEL_NAME)
        state_dict = torch.load(os.path.join(self.__experiment_dir, 'best_model'+self.__MODEL_NAME+'.pt'))
        self.__encoder_model.load_state_dict(state_dict['encoder_model'])
        self.__decoder_model.load_state_dict(state_dict['decoder_model'])
        self.__optimizer.load_state_dict(state_dict['optimizer'])
        
        self.__encoder_model.eval()
        self.__decoder_model.eval()
        test_loss = 0
        bleu1_score = 0
        bleu4_score = 0
        test_loss_batch = []
        predicted_sentences = []
        meteor_predicted_sentences = []
        true_sentences = []
        meteor_true_sentences = [] 
        total = 0
        with torch.no_grad():
            for iter, (images, captions, lengths, video_ids,cap_mask) in enumerate(self.__test_loader):
                #print("Inside iter = ",iter)
                self.__max_caption_count = max(lengths)
                if self.__use_gpu:
                    inputs = images.cuda()
                    test_labels = captions.cuda()
                    targets = captions[:,1:].long().cuda()
                    # padding = Variable(torch.zeros(targets.shape[0], self.__max_frame)).long().cuda()
                    # targets = torch.cat((padding, targets), 1)
                    cap_mask = cap_mask[:, 1:].cuda()
                else:
                    inputs, test_labels, targets,cap_mask = images, captions, captions[:,1:].long(),cap_mask[:, 1:]

                features = self.__encoder_model(inputs,test_labels,self.__max_frame)
 

                #caption generation part
                pred_caption = self.__decoder_model.generate_captions_deterministic(features,self.__max_caption_count,self.batch_size) #for caption
                sentences = generate_text_caption(pred_caption,self.__vocab,self.__max_caption_count)
                predicted_sentences.extend(sentences)
                meteor_sentences = generate_text_sentence(pred_caption,self.__vocab,self.__max_caption_count)
                meteor_predicted_sentences.extend(meteor_sentences)

                total+=len(captions)
                truth = generate_text_caption(targets.cpu().numpy(),self.__vocab,self.__max_caption_count)
                meteor_truth = generate_text_sentence(targets.cpu().numpy(),self.__vocab,self.__max_caption_count)
                true_sentences.extend(truth)
                meteor_true_sentences.extend(meteor_truth)
                #if iter%10 == 0:
                for num in range(0,len(sentences)):
                    sentence = sentences[num]
                    meteor_sentence = meteor_sentences[num]
                    #truth_ref = id2truth[video_ids[num]][0]
                    print('TEST: Video_id = {}, sentence for image # {} in iteration # {} is: {}, truth is: {}'.format(video_ids[num],num,iter,sentence,truth[num]))

                targets = torch.reshape(targets, (targets.shape[0]*targets.shape[1],1)).squeeze(1)  
                outputs = self.__decoder_model(features, test_labels)
                #outputs = outputs[:,self.__max_frame:,:]
#                 loss = self.__criterion(outputs, targets)
#                 loss = torch.unsqueeze(loss,0)
#                 loss = loss.mean()
                logit_loss = self.__criterion(outputs, targets) 
                masked_loss = logit_loss*cap_mask
                loss = torch.sum(masked_loss)/torch.sum(cap_mask)
                test_loss_batch.append(loss.item())     

        test_loss = np.mean(np.array(test_loss_batch))
        #true_sentences = true_sentences[:total] #dropping off last incomplete batch
        #meteor_true_sentences = meteor_true_sentences[:total]
        print("Length of true sentences = {}, length of predicted sentences = {}".format(len(true_sentences),len(predicted_sentences)))
        #write_to_file_in_dir(self.__experiment_dir, 'true_sentences.txt', true_sentences)
        self.write_to_file('predicted_sentences.txt', predicted_sentences)
        meteor_score = meteor(meteor_true_sentences,meteor_predicted_sentences)
        result_str = "Test Performance: Loss: {}, Meteor: {}".format(test_loss,meteor_score)
        self.__log(result_str)
        return test_loss, meteor_score

    def __save_model(self,name='latest_model.pt'):
        root_model_path = os.path.join(self.__experiment_dir, name)
        encoder_model_dict = self.__encoder_model.state_dict()
        decoder_model_dict = self.__decoder_model.state_dict()
        state_dict = {'encoder_model': encoder_model_dict, 'decoder_model': decoder_model_dict, 'optimizer': self.__optimizer.state_dict()}
        torch.save(state_dict, root_model_path)

    def __record_stats(self, train_loss, val_loss):
        self.__training_losses.append(train_loss)
        self.__val_losses.append(val_loss)

        self.plot_stats()

        # write_to_file_in_dir(self.__experiment_dir, 'training_losses.txt', self.__training_losses)
        # write_to_file_in_dir(self.__experiment_dir, 'val_losses.txt', self.__val_losses)

    def __log(self, log_str, file_name=None):
        print(log_str)
        # log_to_file_in_dir(self.__experiment_dir, 'all.log', log_str)
        # if file_name is not None:
        #     log_to_file_in_dir(self.__experiment_dir, file_name, log_str)

    def __log_epoch_stats(self, start_time):
        time_elapsed = datetime.now() - start_time
        time_to_completion = time_elapsed * (self.__epochs - self.__current_epoch - 1)
        train_loss = self.__training_losses[self.__current_epoch]
        val_loss = self.__val_losses[self.__current_epoch]
        summary_str = "Epoch: {}, Train Loss: {}, Val Loss: {}, Took {}, ETA: {}\n"
#         summary_str = summary_str.format(self.__current_epoch + 1, train_loss, str(time_elapsed),
#                                          str(time_to_completion))
        summary_str = summary_str.format(self.__current_epoch + 1, train_loss, val_loss, str(time_elapsed),
                                         str(time_to_completion))
        self.__log(summary_str, 'epoch.log')

    def plot_stats(self):
        e = len(self.__training_losses)
        x_axis = np.arange(1, e + 1, 1)
        plt.figure()
        plt.plot(x_axis, self.__training_losses, label="Training Loss")
        plt.plot(x_axis, self.__val_losses, label="Validation Loss")
        plt.xlabel("Epochs")
        plt.legend(loc='best')
        plt.title(self.__name + " Stats Plot")
        plt.savefig(os.path.join(self.__experiment_dir,self.__MODEL_NAME+"stat_plot.png"))
        plt.show()
        
    def write_to_file(self,file_name, data):
        with open(data_path+'/'+file_name, "w") as outfile:
            json.dump(data, outfile)
