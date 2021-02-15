from torchvision import utils
from resNet import *
from utils import *
from dataloader import *
import torchvision
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import time
import torch.nn as nn
import pandas as pd 
from collections import defaultdict

# First read the dataset
train_dataset = IddDataset(csv_file='../train.csv')
val_dataset = IddDataset(csv_file='../val.csv')
test_dataset = IddDataset(csv_file='../test.csv')

train_loader = DataLoader(dataset=train_dataset, batch_size= 32, num_workers=4, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size= 32, num_workers=4, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size= 32, num_workers=4, shuffle=False)

def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.xavier_uniform_(m.weight.data)
        torch.nn.init.zeros_(m.bias.data)

# dice_coefficient_loss() is used to compute the Dice Coefficient loss function
class dice_coefficient_loss(nn.Module):
    def __init__(self):
        """
            Initialize the constructor for this class
        """
        super(dice_coefficient_loss, self).__init__()
    
    def forward(self, pred, target):
        axes = tuple(range(2, len(pred.shape)))
        numerator = 2 * torch.sum(pred * target, axes)
        denominator = torch.sum((pred * pred) + (target * target), axes)
        return 1 - torch.mean((numerator + 1e-6) / (denominator + 1e-6))
    
epochs = 100
n_class = 27
criterion = dice_coefficient_loss()
resNet_model = resNet(n_class=n_class)
print("No of classes = ",n_class)
learning_rate = 0.005
optimizer = optim.Adam(resNet_model.parameters(), lr=learning_rate)
resNet_model.apply(init_weights)

use_gpu = torch.cuda.is_available()
if use_gpu:
    print("Using GPU")
    resNet_model = resNet_model.cuda()

def train(learning_rate):
    """
        train() - This function is used to complete training including early stopping
    """
    print("Reached train function")
    continuous_epochs, best_loss = 0, float('inf')
    val_loss = []
    train_loss = []
    for epoch in range(epochs):
        train_loss_batch = []
        ts = time.time()
        print("Epoch: {}".format(epoch))
        for itera, (X, tar, Y) in enumerate(train_loader):
            optimizer.zero_grad()
            if use_gpu:
                inputs = X.cuda()
                train_labels = Y.cuda()
                train_targets = tar.cuda()
            else:
                inputs, train_labels, train_targets = X, Y, tar

            outputs = resNet_model(inputs)
            loss = criterion(outputs, train_targets)
            loss = torch.unsqueeze(loss, 0)
            loss = loss.mean()
            train_loss_batch.append(loss.item())
            loss.backward()
            optimizer.step()
            
            if itera % 100 == 0:
                print("TRAINING: epoch{}, iter{}, loss: {}".format(epoch, itera, loss.item()))
        torch.cuda.empty_cache()
        print("Finish epoch {}, time elapsed {}".format(epoch, time.time() - ts))
        train_loss.append(np.mean(np.array(train_loss_batch)))
        curr_val_loss = val(epoch)
        val_loss.append(curr_val_loss)
        if curr_val_loss < best_loss:
            best_loss = curr_val_loss
            model_name = 'resNet_model_best_model_' + str(learning_rate) + '_' + str(epochs) + "_modifiedEarlyStop"
            torch.save(resNet_model, model_name)
            continuous_epochs = 0
        else:
            continuous_epochs += 1
            if(continuous_epochs == 5):
                print("Earlystop after {} epochs".format(epoch))
                return train_loss, val_loss
        torch.save(resNet_model, 'resNet_last_saved_model_modifiedEarlyStop')
        resNet_model.train()
    return train_loss, val_loss

def val(epoch):
    """
        Runs inference of the best model on the validation set
    """
    resNet_model.eval()
    val_loss = []
    val_iou = []
    val_acc = []
    with torch.no_grad():
        for itera, (X, tar, Y) in enumerate(val_loader):
            if use_gpu:
                inputs = X.cuda()
                val_labels = Y.cuda()
                val_targets = tar.cuda()
            else:
                inputs, val_labels = X, Y
            
            outputs = resNet_model(inputs)
            loss = criterion(outputs, val_targets)
            loss = torch.unsqueeze(loss, 0)
            loss = loss.mean()
            # Appending this value to the validation loss
            val_loss.append(loss.item())
            if itera % 100 == 0:
                print("VALIDATION: iter{}, loss: {}".format(itera, loss.item()))
            
            predictions = F.softmax(outputs, 1)
            predictions = torch.argmax(predictions, dim=1)
            iou_row, avg_iou, pix_acc = iou(predictions, val_labels, True)
            val_iou.append(avg_iou)
            val_acc.append(pix_acc)
        
        avg_loss = np.mean(np.asarray(val_loss))
        avg_iou = np.mean(np.asarray(val_iou))
        avg_acc = np.mean(np.asarray(val_acc))
        print("Validation epoch {}: avg_iou = {}, avg_acc = {}".format(epoch,avg_iou,avg_acc))
    
    return avg_loss

def test(learning_rate, epochs):
    """
        Runs inference after loading the best ResNet model using PyTorch
    """
    model_name = 'resNet_model_best_model_' + str(learning_rate) + '_' + str(epochs) + "_modifiedEarlyStop"
    resNet_model = torch.load(model_name)
    resNet_model.eval()
    
    # Make sure to include a softmax after the output from your model
    test_iou = []
    test_acc = []
    test_ious_cls = []
    with torch.no_grad():
        for itera, (X, tar, Y) in enumerate(val_loader):
            if use_gpu:
                inputs = X.cuda()# Move your inputs onto the gpu
                test_labels = Y.cuda()# Move your labels onto the gpu
            else:
                inputs, test_labels = X, Y
            outputs = resNet_model(inputs)
            predictions = F.softmax(outputs, 1)
            # Create one-hot encoding
            predictions = torch.argmax(predictions, dim = 1)
            iou_row, avg_iou, pix_acc = iou(predictions, test_labels)
            # iou_row shouldn't be None
            if iou_row is not None:
                test_ious_cls.append(iou_row)
            # IoU and accuracy values have been computed here
            test_iou.append(avg_iou)
            test_acc.append(pix_acc)
        
        # Here, I have computed both the mean and accuracy values
        avg_iou = np.mean(np.asarray(test_iou))
        avg_acc = np.mean(np.asarray(pix_acc))
        if iou_row is not None:
            avg_ious_cls = np.nanmean(np.asarray(test_ious_cls), axis=0)
        print("Final test from best model : avg_iou = {}, avg_acc = {}".format(avg_iou,avg_acc))
        print(" Class wise ious getting saved in resNet_IOU_Classwise.csv file")
        
        if iou_row is not None:
            d = []
            labels_len = len(labels)
            for idx in range(0, labels_len - 1):
                d.append((labels[idx].name, avg_ious_cls[labels[idx].level3Id]))
                df = pd.DataFrame(d, columns=('Label name', 'IoU'))
                df.to_csv('resNet_IOU_Classwise_modifiedEarlyStop.csv', sep='\t')
        
            test_loader = DataLoader(dataset=test_dataset, batch_size= 32, num_workers=4, shuffle=False)
            for itera, (X, tar, Y) in enumerate(test_loader):
                if use_gpu:
                    inputs = X.cuda()# Move your inputs onto the gpu
                    test_labels = tar.cuda()# Move your labels onto the gpu
                else:
                    inputs, test_labels = X, tar#.long() # Unpack variables into inputs and labels
                outputs = resNet_model(inputs)
                predictions = F.softmax(outputs, 1)
                predictions = torch.argmax(predictions,dim=1)
                break
            predictions = predictions.cpu().numpy()
            inputImage = inputs[0].permute(1, 2, 0).cpu().numpy()
            plt.imshow(inputImage, cmap='gray')
            plt.show()
            rows, cols = predictions.shape[1], predictions.shape[2]

            # Print the labels array here
            new_predictions = np.zeros((predictions.shape[1], predictions.shape[2], 3))
            # Rearranging the image here
            for row in range(rows):
                for col in range(cols):
                    idx = int(predictions[0][row][col])
                    new_predictions[row][col][:] = np.asarray(labels[idx].color)/255       

            # Print and display overlaying grayscale and semantically segmented image
            plt.imshow(inputImage, cmap='gray')
            plt.imshow(new_predictions,alpha=0.5)
            plt.axis('off')
            fig_name = "Overlayed_resNet_modifiedEarlyStop" + str(learning_rate) + '_' + str(epochs) + ".jpg"  
            plt.savefig(fig_name)
            plt.show()
            
def plot_loss_curves(train_loss,val_loss, learning_rate=10**-3):
    title = "Loss curves at learning rate = " + str(learning_rate)
    fig_name = "Loss_resNet_modifiedEarlyStopping.jpg"
    x = [i for i in range(len(train_loss))]
    plt.plot(x, train_loss,label="Train Loss")
    plt.plot(x, val_loss,label="Validation Loss")
    plt.legend()
    plt.xlabel("# of epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.title(title)
    plt.savefig(fig_name)
    plt.show()

# main() is the function that's called here
#def main():
if __name__ == '__main__':
    # Here, we have the main function calling all the other functions
    train_loss, val_loss = train(0.005)
    best_val_loss = float('inf')
    curr_val_loss = np.min(np.asarray(val_loss))
    if(curr_val_loss < best_val_loss):
        best_val_loss = curr_val_loss
        best_learning_rate = lr
    print("Best validation loss = {}".format(best_val_loss))
    plot_loss_curves(train_loss,val_loss, lr)
    test(0.005, 100)