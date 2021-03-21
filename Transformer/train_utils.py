import torch
import vocab
import nltk
from caption_utils import *
import torch.nn.functional as F
import numpy as np
import time
from torch.autograd import Variable


def train(model, modelname, loader, val_loader, optim, activity_vocab):
    start = time.time()
    temp = start

    modelname = modelname # please change this 
    early_stop_patience = 5
    early_stop_counter = 0
    total_loss = 0
    epochs = 30
    print_every = 200
    train_losses = []
    val_losses = []
    meteors = []
    bleu1s = []
    bleu4s = []
    best_meteor = 0
    for epoch in range(epochs):
        val_loss, meteor_avg, bleu1_avg, bleu4_avg = get_val_loss(model, val_loader, activity_vocab)
        val_losses.append(val_loss)
        meteors.append(meteor_avg)
        bleu1s.append(bleu1_avg)
        bleu4s.append(bleu4_avg)
        print("avg val loss: {}".format(val_losses))
        print("avg meteor: {}".format(meteors))
        print("avg bleu1: {}".format(bleu1s))
        print("avg bleu4: {}".format(bleu4s))
        if meteor_avg > best_meteor:
            best_meteor = meteor_avg
            print("best meteor achieved ({:.2f}), saving model {}".format(best_meteor, modelname))
            torch.save(model.state_dict(), modelname)
        else:
            early_stop_counter += 1
            if early_stop_counter >= early_stop_patience:
                print("early stopping...")
                break
        model.train()
        epoch_loss = 0 
        for i, a in enumerate(loader):
            src = a[0].squeeze().cuda()
            target_input = a[1][:,:-1].cuda()
            target = a[1][:,1:].cuda()
            nopeak_mask = np.triu(np.ones((1, target.shape[1], target.shape[1])),k=1)
            nopeak_mask = Variable(torch.from_numpy(nopeak_mask) == 0).cuda()
            preds = model(src, target_input, None, nopeak_mask)

            optim.zero_grad()

            loss = F.cross_entropy(preds.view(-1, preds.size(-1)), target.squeeze(), ignore_index=0)
            loss.backward()
            optim.step()
            total_loss += loss.data
            epoch_loss += loss.data
            if (i + 1) % print_every == 0:
                loss_avg = total_loss / print_every
                print("time = %dm, epoch %d, iter = %d, loss = %.3f,\
                %ds per %d iters" % ((time.time() - start) // 60,
                epoch + 1, i + 1, loss_avg, time.time() - temp,
                print_every))
                total_loss = 0
                temp = time.time()
            if i == len(loader) - 1:
                print("-" * 20)
                caption = ""
                for wordidx in a[1][0].numpy():
                    caption += (" " + activity_vocab.idx2word[wordidx])
                print("caption: {}".format(caption))
                pred_caption = ""
                for wordidx in preds.max(2)[1][0].cpu().numpy():
                    pred_caption += (" " + activity_vocab.idx2word[wordidx])
                print("prediction: {}".format(pred_caption))
        epoch_loss /= len(loader)
        train_losses.append(epoch_loss.item())
        print("avg epoch loss: {}".format(train_losses))
        print("-" * 20)
    np.save("{}_stats".format(modelname), [train_losses, val_losses, meteors, bleu1s, bleu4s])


def get_val_loss(model, loader, activity_vocab):
    val_loss = 0
    model.eval()
    meteor_avg = 0
    bleu1_avg = 0
    bleu4_avg = 0
    with torch.no_grad():
        for a in loader:
            src = a[0].squeeze().cuda()
            target_input = a[1][:,:-1].cuda()
            target = a[1][:,1:].cuda()
            mask_src = torch.ones((a[0].shape[1])).cuda()
            mask_target = torch.ones((a[1].shape[1] - 1)).cuda()
            preds = model(src, target_input, None, None)
            
            caption = ""
            for wordidx in a[1][0].numpy()[1:-1]:
                caption += (" " + activity_vocab.idx2word[wordidx])
            pred_caption = ""
            for wordidx in preds.max(2)[1][0].cpu().numpy()[:-1]:
                pred_caption += (" " + activity_vocab.idx2word[wordidx])
#             print("caption: {}".format(caption))
#             print("pred caption: {}".format(pred_caption))
#             print(round(nltk.translate.meteor_score.single_meteor_score(caption, pred_caption),4))
#             print(bleu1([[caption]], [pred_caption]))
#             print(bleu4([[caption]], [pred_caption]))
            meteor_avg += round(nltk.translate.meteor_score.single_meteor_score(caption, pred_caption),4)
            bleu1_avg += bleu1([[caption]], [pred_caption])
            bleu4_avg += bleu4([[caption]], [pred_caption])
            loss = F.cross_entropy(preds.view(-1, preds.size(-1)), target.squeeze(), ignore_index=0)
            val_loss += loss.data
    loss_avg = val_loss / len(loader)
    meteor_avg = meteor_avg / len(loader) * 100
    bleu1_avg /= len(loader)
    bleu4_avg /= len(loader)
    return loss_avg.item(), meteor_avg, bleu1_avg, bleu4_avg

def generate_captions(model, src, max_length = 20):
    start = torch.tensor([[0]]).cuda()
    curr_input = start
    pred_caption = ""
    curr_length = 1
    while curr_length < max_length:
        nopeak_mask = np.triu(np.ones((1, curr_length, curr_length)), k=1)
        nopeak_mask = Variable(torch.from_numpy(nopeak_mask) == 0).cuda()
        pred = model(src, curr_input, None, nopeak_mask)
        pred_idx = pred[0][-1].max(0).indices.item()
        pred_word = activity_vocab.idx2word[pred_idx]
#         print("pred word: {}".format(pred_word))
        if pred_word == "<end>":
            break
        curr_input = torch.cat((curr_input, torch.tensor([[pred_idx]]).cuda()), dim=1) # concat the last word
        curr_length += 1
        pred_caption += (" " + pred_word)
    return pred_caption[1:]