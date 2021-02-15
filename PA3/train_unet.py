from torchvision import utils
from unet import *
from dataloader import *
from utils import *
import torchvision
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import time
import matplotlib.pyplot as plt


# TODO: Some missing values are represented by '__'. You need to fill these up.

train_dataset = IddDataset(csv_file='train.csv')
val_dataset = IddDataset(csv_file='val.csv')
test_dataset = IddDataset(csv_file='test.csv')

BATCH_SIZE = 6
NUM_WORKERS = 8
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False)


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.xavier_uniform_(m.weight.data)
        torch.nn.init.zeros_(m.bias.data)        

epochs = 100        
criterion = dice_coefficient_loss() # Choose an appropriate loss function from https://pytorch.org/docs/stable/_modules/torch/nn/modules/loss.html
unet_model = unet(n_class=n_class)
unet_model.apply(init_weights)

optimizer = optim.Adam(unet_model.parameters(), lr=3e-3)

use_gpu = torch.cuda.is_available()
if use_gpu:
    unet_model = unet_model.cuda()

        
def train():
    print("starting training")
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    for epoch in range(epochs):
        ts = time.time()
        train_loss_batch = []
        for iter, (X, tar, Y) in enumerate(train_loader):
            optimizer.zero_grad()

            if use_gpu:
                inputs = X.cuda()# Move your inputs onto the gpu
                labels = Y.cuda()# Move your labels onto the gpu
                targets = tar.cuda()
            else:
                inputs, targets, labels = X, tar, Y# Unpack variables into inputs and labels

            outputs = unet_model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss_batch.append(loss.item())

            if iter % 120 == 0:
                print("epoch{}, iter{}, loss: {}".format(epoch, iter, loss.item()))
        
        print("Finish epoch {}, time elapsed {}".format(epoch, time.time() - ts))

        train_losses.append(sum(train_loss_batch)/len(train_loss_batch))
        val_loss, val_iou = val(epoch)
        val_losses.append(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print("best val loss achieved, saving model")
            torch.save(unet_model, 'unet')
        if epoch >= 3:
            stop = 0
            for i in range(0,3):
                if val_losses[epoch-i] > val_losses[epoch-i-1]:
                    stop = stop + 1
            if stop == 3:
                print ("EarlyStop after %d epochs." % (epoch))
                return train_losses, val_losses
        unet_model.train()
        print("-" * 20)
    return train_losses, val_losses


def val(epoch):
    unet_model.eval() # Don't forget to put in eval mode !
    #Complete this function - Calculate loss, accuracy and IoU for every epoch
    # Make sure to include a softmax after the output from your model
    val_loss = []
    val_iou = []
    val_acc = []
    with torch.no_grad():
        for itera, (X, tar, Y) in enumerate(val_loader):
            if use_gpu:
                inputs = X.cuda()# Move your inputs onto the gpu
                val_labels = Y.cuda()# Move your labels onto the gpu
                val_targets = tar.cuda()
            else:
                inputs, val_targets, val_labels = X, tar, Y#.long()# Unpack variables into inputs and labels
            outputs = unet_model(inputs)
            loss = criterion(outputs, val_targets)
            loss = torch.unsqueeze(loss,0)
            loss = loss.mean()
            val_loss.append(loss.item())
            predictions = F.softmax(outputs,1)
            predictions = torch.argmax(predictions,dim=1)
            iou_row,avg_iou = iou(predictions,val_labels)
            val_iou.append(avg_iou)
            val_acc.append(pixel_acc(predictions,val_labels))
        avg_loss = np.mean(np.array(val_loss))
        avg_iou = np.mean(np.array(val_iou))
        avg_acc = np.mean(np.array(val_acc))
        print("Validation epoch {}: avg_dice_loss = {}, avg_iou = {}, avg_acc = {}".format(epoch,avg_loss,avg_iou,avg_acc))
        return avg_loss, avg_iou
    
def test():
    unet_model = torch.load('unet-Copy1')
    unet_model.eval()
    #Complete this function - Calculate accuracy and IoU 
    # Make sure to include a softmax after the output from your model
    val_iou = []
    val_acc = []
    val_ious_cls = []
    with torch.no_grad():
        for itera, (X, tar, Y) in enumerate(val_loader):
            if use_gpu:
                inputs = X.cuda()# Move your inputs onto the gpu
                test_labels = Y.cuda()# Move your labels onto the gpu
            else:
                inputs, test_labels = X,Y#.long()# Unpack variables into inputs and labels
            outputs = unet_model(inputs)
            predictions = torch.nn.functional.softmax(outputs,1)
            # create one-hot encoding
            predictions = torch.argmax(predictions,dim=1)
            iou_row,avg_iou = iou(predictions,test_labels)
            val_ious_cls.append(iou_row)
            val_iou.append(avg_iou)
            val_acc.append(pixel_acc(predictions,test_labels))
        avg_iou = np.mean(np.asarray(val_iou))
        avg_acc = np.mean(np.asarray(val_acc))
        avg_ious_cls = np.nanmean(np.asarray(val_ious_cls),axis=0) #iou for the class when it's union=0 will be nan
        print("Final test from best model : avg_iou = {}, avg_acc = {}".format(avg_iou,avg_acc))
        print(" Class wise ious getting saved in unet_IOU_Classwise.csv file")
        

        d = []
        labels_len = len(labels)
        for idx in range(0,labels_len-1):
             d.append((labels[idx].name, avg_ious_cls[labels[idx].level3Id]))
        df = pd.DataFrame(d, columns=('Label name', 'IoU'))
        df.to_csv('unet_IOU_Classwise.csv', sep='\t')

        test_loader = DataLoader(dataset=test_dataset, batch_size= 1, num_workers=8, shuffle=False)
        for itera, (X, tar, Y) in enumerate(test_loader):
            if use_gpu:
                inputs = X.cuda()# Move your inputs onto the gpu
                test_labels = Y.cuda()# Move your labels onto the gpu
            else:
                inputs, test_labels = X, Y#.long() # Unpack variables into inputs and labels
            outputs = unet_model(inputs)
            predictions = torch.nn.functional.softmax(outputs,1)
            predictions = torch.argmax(predictions,dim=1)
            break
        predictions = predictions.cpu().numpy()
        inputImage = inputs[0].permute(1, 2, 0).cpu().numpy()
        plt.imshow(inputImage, cmap='gray')
        plt.show()
        rows, cols = predictions.shape[1], predictions.shape[2]
        #print(labels)
        new_predictions = np.zeros((predictions.shape[1], predictions.shape[2], 3))
        for row in range(rows):
            for col in range(cols):
                idx = int(predictions[0][row][col])
                new_predictions[row][col][:] = np.asarray(labels[idx].color)/255       

        plt.imshow(inputImage)
        plt.imshow(new_predictions, alpha=0.5)
        plt.axis('off')
        fig_name = "Overlayed_unet.jpg"  
        plt.savefig(fig_name, dpi=300)
        plt.show()
            

if __name__ == "__main__":
    val(0)  # show the accuracy before training
    train()