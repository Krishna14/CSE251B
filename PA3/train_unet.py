from torchvision import utils
from basic_fcn import *
from dataloader import *
from utils import *
import torchvision
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import time


# TODO: Some missing values are represented by '__'. You need to fill these up.

train_dataset = IddDataset(csv_file='train.csv')
val_dataset = IddDataset(csv_file='val.csv')
test_dataset = IddDataset(csv_file='test.csv')


train_loader = DataLoader(dataset=train_dataset, batch_size=3, num_workers=4, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=3, num_workers=4, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=3, num_workers=4, shuffle=False)


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.xavier_uniform_(m.weight.data)
        torch.nn.init.zeros_(m.bias.data)        

epochs = 100        
criterion = nn.CrossEntropyLoss() # Choose an appropriate loss function from https://pytorch.org/docs/stable/_modules/torch/nn/modules/loss.html
fcn_model = FCN(n_class=n_class)
fcn_model.apply(init_weights)

optimizer = optim.Adam(fcn_model.parameters(), lr=5e-3)

use_gpu = torch.cuda.is_available()
if use_gpu:
    fcn_model = fcn_model.cuda()

        
def train():
    print("starting training")
    for epoch in range(epochs):
        ts = time.time()
        for iter, (X, tar, Y) in enumerate(train_loader):
            optimizer.zero_grad()

            if use_gpu:
                inputs = X.cuda()# Move your inputs onto the gpu
                labels = Y.cuda()# Move your labels onto the gpu
            else:
                inputs, labels = X, Y# Unpack variables into inputs and labels

            outputs = fcn_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if iter % 500 == 0:
                print("epoch{}, iter{}, loss: {}".format(epoch, iter, loss.item()))
        
        print("Finish epoch {}, time elapsed {}".format(epoch, time.time() - ts))
        torch.save(fcn_model, 'best_model')

        val(epoch)
        fcn_model.train()
    


def val(epoch):
    fcn_model.eval() # Don't forget to put in eval mode !
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
            else:
                inputs, val_labels = X, Y#.long()# Unpack variables into inputs and labels
            outputs = fcn_model(inputs)
            loss = criterion(outputs, val_labels)
            loss = torch.unsqueeze(loss,0)
            loss = loss.mean()
            val_loss.append(loss.item())
#             if itera % 100 == 0:
#                 print("VALIDATION: iter{}, loss: {}".format(itera, loss.item()))
            predictions = F.softmax(outputs,1)
            predictions = torch.argmax(predictions,dim=1)
            iou_row,avg_iou = iou(predictions,val_labels)
            val_iou.append(avg_iou)
            val_acc.append(pixel_acc(predictions,val_labels))
#         val_loss = val_loss[:-1]
#         val_iou = val_iou[:-1]
#         val_acc = val_acc[:-1]
        avg_loss = np.mean(np.asarray(val_loss))
        avg_iou = np.mean(np.asarray(val_iou))
        avg_acc = np.mean(np.asarray(val_acc))
        print("Validation epoch {}: avg_iou = {}, avg_acc = {}".format(epoch,avg_iou,avg_acc))
        return avg_loss, inputs   
    
def test():
	fcn_model.eval()
    #Complete this function - Calculate accuracy and IoU 
    # Make sure to include a softmax after the output from your model
    
if __name__ == "__main__":
    val(0)  # show the accuracy before training
    train()