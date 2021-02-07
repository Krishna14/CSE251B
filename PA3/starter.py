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


train_loader = DataLoader(dataset=train_dataset, batch_size= __, num_workers= __, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size= __, num_workers= __, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size= __, num_workers= __, shuffle=False)


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.xavier_uniform(m.weight.data)
        torch.nn.init.xavier_uniform(m.bias.data)        

epochs = 100        
criterion = # Choose an appropriate loss function from https://pytorch.org/docs/stable/_modules/torch/nn/modules/loss.html
fcn_model = FCN(n_class=n_class)
fcn_model.apply(init_weights)

optimizer = optim.Adam(fcn_model.parameters(), lr=__)

use_gpu = torch.cuda.is_available()
if use_gpu:
    fcn_model = fcn_model.cuda()

        
def train():
    for epoch in range(epochs):
        ts = time.time()
        for iter, (X, tar, Y) in enumerate(train_loader):
            optimizer.zero_grad()

            if use_gpu:
                inputs = # Move your inputs onto the gpu
                labels = # Move your labels onto the gpu
            else:
                inputs, labels = # Unpack variables into inputs and labels

            outputs = fcn_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if iter % 10 == 0:
                print("epoch{}, iter{}, loss: {}".format(epoch, iter, loss.item()))
        
        print("Finish epoch {}, time elapsed {}".format(epoch, time.time() - ts))
        torch.save(fcn_model, 'best_model')

        val(epoch)
        fcn_model.train()
    


def val(epoch):
    fcn_model.eval() # Don't forget to put in eval mode !
    #Complete this function - Calculate loss, accuracy and IoU for every epoch
    # Make sure to include a softmax after the output from your model
    
def test():
	fcn_model.eval()
    #Complete this function - Calculate accuracy and IoU 
    # Make sure to include a softmax after the output from your model
    
if __name__ == "__main__":
    val(0)  # show the accuracy before training
    train()