################################################################################
# CSE 253: Programming Assignment 4
# Code snippet by Ajit Kumar, Savyasachi
# Fall 2020
################################################################################

import matplotlib.pyplot as plt
import numpy as np
import torch
from datetime import datetime

import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence

from caption_utils import *
from constants import ROOT_STATS_DIR
from dataset_factory import get_datasets
from file_utils import *
from model_factory import get_model

import nltk
#nltk.download('punkt')

# Class to encapsulate a neural experiment.
# The boilerplate code to setup the experiment, log stats, checkpoints and plotting have been provided to you.
# You only need to implement the main training logic of your experiment and implement train, val and test methods.
# You are free to modify or restructure the code as per your convenience.
class Experiment(object):
    def __init__(self, name):
        config_data = read_file_in_dir('./', name + '.json')
        if config_data is None:
            raise Exception("Configuration file doesn't exist: ", name)

        self.__name = config_data['experiment_name']
        self.__experiment_dir = os.path.join(ROOT_STATS_DIR, self.__name)

        # Load Datasets
        self.__coco_test, self.__vocab, self.__train_loader, self.__val_loader, self.__test_loader = get_datasets(
            config_data)

        # Setup Experiment
        self.__generation_config = config_data['generation']
        self.__epochs = config_data['experiment']['num_epochs']
        self.__learning_rate = config_data['experiment']['learning_rate']
        self.__early_stop_threshold = config_data['experiment']['early_stop_threshold']
        self.__test_caption_path = config_data['dataset']['test_annotation_file_path']
        self.__max_caption_count = config_data['generation']['max_length']
        self.__current_epoch = 0
        self.__training_losses = []
        self.__val_losses = []
        self.__best_model = None  # Save your best model in this field and use this in test method.
        #
        self.__best_encoder_model = None  # Save the best encoder model here
        self.__best_decoder_model = None  # Save the best decoder model here

        # Init Model
        self.__encoder_model, self.__decoder_model = get_model(config_data, self.__vocab)
        # TODO: Set these Criterion and Optimizers Correctly
        self.__criterion = nn.CrossEntropyLoss()
        parameters = list(self.__decoder_model.parameters()) + list(self.__encoder_model.parameters()) + list(self.__encoder_model.batchNorm.parameters())
        self.__optimizer = optim.Adam(parameters, lr=self.__learning_rate)
        self.__MODEL_NAME = self.__name + '_' + str(self.__learning_rate) + '_' + str(self.__epochs) + '_' + \
        str(config_data['model']['embedding_size']) + '_' + str(config_data['model']['hidden_size'])

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

    def __init_model(self):
        if torch.cuda.is_available():
            print("Using GPU")
            self.__use_gpu = True
            self.__encoder_model = self.__encoder_model.cuda().float()
            self.__decoder_model = self.__decoder_model.cuda().float()
            self.__criterion = self.__criterion.cuda()

    # Main method to run your experiment. Should be self-explanatory.
    def run(self):
        start_epoch = self.__current_epoch
        min_val_loss = float('inf')
        continuous_increase = 0
        for epoch in range(start_epoch, self.__epochs):  # loop over the dataset multiple times
            start_time = datetime.now()
            self.__current_epoch = epoch
            # Here, we compute the training loss
            train_loss = self.__train()
            val_loss = self.__val()
            # Saving the best model here
            if(val_loss < min_val_loss):
                min_val_loss = val_loss
                print("Saving the best model after {} epochs".format(epoch))
                continuous_increase = 0
                self.__best_encoder_model = self.__encoder_model
                self.__best_decoder_model = self.__decoder_model
                self.__save_model('best_model'+self.__MODEL_NAME+'.pt')
#             else:
#                 continuous_increase += 1
#                 if(continuous_increase == self.__early_stop_threshold):
#                     print("Early stopping: Epoch = {}".format(epoch))
#                     break
            self.__record_stats(train_loss, val_loss)
            self.__log_epoch_stats(start_time)
            self.__save_model()
            
            if epoch >= self.__early_stop_threshold:
                stop = 0
                for i in range(0,self.__early_stop_threshold):
                    if self.__val_losses[epoch-i] > self.__val_losses[epoch-i-1]:
                        stop = stop + 1
                if stop == self.__early_stop_threshold :
                    print ("EarlyStop after %d epochs." % (epoch))
                    break
        print ("Experiment done!")

    # TODO: Perform one training iteration on the whole dataset and return loss value
    def __train(self):
        self.__encoder_model.train()
        self.__decoder_model.train()
        train_loss_batch = []
        for i, (images, captions, img_ids, lengths) in enumerate(self.__train_loader):
            self.__optimizer.zero_grad()
            targets = pack_padded_sequence(captions, lengths, batch_first=True)
            if self.__use_gpu:
                inputs = images.cuda()
                train_labels = captions.cuda()
                targets = targets[0].cuda()
            else:
                inputs, train_labels, targets = images, captions, targets[0]

            features = self.__encoder_model(inputs)
            # Caption generation part
            pred_captions = self.__decoder_model.generate_captions(features)
            sentences = generate_text_caption(pred_captions, self.__vocab, self.__max_caption_count)
            
            if (i % 100 == 0):
                plt.imshow(images[0].permute(1, 2, 0))
                plt.axis('off')
                plt.show()
                for num in range(0, 1):
                    sentence = sentences[num]
                    print('Sentence for image # {} in iteration # {} is {}'.format(num, i, sentence))

            outputs = self.__decoder_model(features, train_labels, lengths)
            loss = self.__criterion(outputs, targets)
            loss = torch.unsqueeze(loss,0)
            loss = loss.mean()
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
            for i, (images, captions, img_ids, lengths) in enumerate(self.__val_loader):
                targets = pack_padded_sequence(captions, lengths, batch_first=True)
                if self.__use_gpu:
                    inputs = images.cuda()
                    val_labels = captions.cuda()
                    targets = targets[0].cuda()
                else:
                    inputs, val_labels, targets = images, captions, targets[0]

                features = self.__encoder_model(inputs)
                outputs = self.__decoder_model(features, val_labels, lengths)
                loss = self.__criterion(outputs, targets)
                loss = torch.unsqueeze(loss,0)
                loss = loss.mean()
                val_loss_batch.append(loss.item())
            return np.mean(np.array(val_loss_batch))

    #  TODO: Implement your test function here. Generate sample captions and evaluate loss and
    #  bleu scores using the best model. Use utility functions provided to you in caption_utils.
    #  Note than you'll need image_ids and COCO object in this case to fetch all captions to generate bleu scores.
    def test(self):
        state_dict = torch.load(os.path.join(self.__experiment_dir, 'best_model'+self.__MODEL_NAME+'.pt'))
        self.__encoder_model.load_state_dict(state_dict['encoder_model'])
        self.__decoder_model.load_state_dict(state_dict['decoder_model'])
        self.__optimizer.load_state_dict(state_dict['optimizer'])
        self.__encoder_model.eval()
        self.__decoder_model.eval()
        test_loss = 0
        bleu1_score = 0
        bleu4_score = 0
        test_loss_batch, bleu1_score_batch, bleu4_score_batch = [], [], []
        coco = COCO(self.__test_caption_path)
        predicted_sentences, true_sentences = [], []
        with torch.no_grad():
            for itera, (images, captions, img_ids, lengths) in enumerate(self.__test_loader):
                targets = pack_padded_sequence(captions, lengths, batch_first=True)
                if self.__use_gpu:
                    inputs = images.cuda()
                    test_labels = captions.cuda()
                    targets = targets[0].cuda()
                else:
                    inputs, test_labels, targets = images, captions, targets[0]
                
                features = self.__encoder_model(inputs)
                
                # Caption generation part
                pred_caption = self.__decoder_model.generate_captions(features)
                predicted_sentences.extend(generate_text_caption(pred_caption, self.__vocab, self.__max_caption_count))
                # Caption length at iteration "x" is:
                true_sentences.extend(get_true_captions(img_ids, coco))
                
                if itera == 0:
                    for num in range(0, 1):
                        plt.imshow(images[num].permute(1, 2, 0))
                        plt.axis('off')
                        plt.show()
                        sentence = predicted_sentences[num]
                        truth = true_sentences[num]
                        print("Sentence for image # {} in iteration # {} is {}, actual: {}".\
                              format(num, itera, sentence, truth))
                outputs = self.__decoder_model(features, test_labels, lengths)
                loss = self.__criterion(outputs, targets)
                loss = torch.unsqueeze(loss, 0)
                loss = loss.mean()
                test_loss_batch.append(loss.item())
                    
        test_loss = np.mean(np.array(test_loss_batch))
        print("Length of true sentences = {}, length of predicted sentences = {}".format(len(true_sentences),len(predicted_sentences)))
        
        write_to_file_in_dir(self.__experiment_dir, 'true_sentences.txt', true_sentences)
        write_to_file_in_dir(self.__experiment_dir, 'predicted_sentences.txt', predicted_sentences)
        bleu1_score = bleu1(true_sentences,predicted_sentences)
        bleu4_score = bleu4(true_sentences,predicted_sentences)
        result_str = "Test Performance: Loss: {}, Bleu1: {}, Bleu4: {}".format(test_loss,bleu1_score,bleu4_score)
        self.__log(result_str)
        
        return test_loss, bleu1_score, bleu4_score

    def __save_model(self, name='latest_model.pt'):
        root_model_path = os.path.join(self.__experiment_dir, name)
        encoder_model_dict = self.__encoder_model.state_dict()
        decoder_model_dict = self.__decoder_model.state_dict()
        state_dict = {'encoder_model': encoder_model_dict, 'decoder_model': decoder_model_dict, 'optimizer': self.__optimizer.state_dict()}
        torch.save(state_dict, root_model_path)

    def __record_stats(self, train_loss, val_loss):
        self.__training_losses.append(train_loss)
        self.__val_losses.append(val_loss)

        self.plot_stats()

        write_to_file_in_dir(self.__experiment_dir, 'training_losses.txt', self.__training_losses)
        write_to_file_in_dir(self.__experiment_dir, 'val_losses.txt', self.__val_losses)

    def __log(self, log_str, file_name=None):
        print(log_str)
        log_to_file_in_dir(self.__experiment_dir, 'all.log', log_str)
        if file_name is not None:
            log_to_file_in_dir(self.__experiment_dir, file_name, log_str)

    def __log_epoch_stats(self, start_time):
        time_elapsed = datetime.now() - start_time
        time_to_completion = time_elapsed * (self.__epochs - self.__current_epoch - 1)
        train_loss = self.__training_losses[self.__current_epoch]
        val_loss = self.__val_losses[self.__current_epoch]
        summary_str = "Epoch: {}, Train Loss: {}, Val Loss: {}, Took {}, ETA: {}\n"
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
        plt.savefig(os.path.join(self.__experiment_dir, "stat_plot.png"))
        plt.show()
