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

train_dataset = IddDataset(csv_file='../train.csv')
val_dataset = IddDataset(csv_file='../val.csv')
test_dataset = IddDataset(csv_file='../test.csv',resize=False)


# train_loader, val_loader and test_loader are three different sets of data
train_loader = DataLoader(dataset=train_dataset, batch_size= 6, num_workers=8, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size= 6, num_workers=8, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size= 6, num_workers=8, shuffle=False)


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.xavier_uniform_(m.weight.data)
        #print("Dimension of weight = {}".format(m.weight.shape))
        #print("Dimension of bias = {}".format(m.bias.shape))
        #torch.nn.init.xavier_uniform(m.bias.data)
        torch.nn.init.zeros_(m.bias.data)     

epochs = 100        
criterion = nn.CrossEntropyLoss()# Choose an appropriate loss function from https://pytorch.org/docs/stable/_modules/torch/nn/modules/loss.html
fcn_model = FCN(n_class=n_class)
fcn_model.apply(init_weights)

optimizer = optim.Adam(fcn_model.parameters(), lr=5e-3)
MODEL_NAME = "best_model_baseline_"

use_gpu = torch.cuda.is_available()
if use_gpu:
    print("Using GPU")
    fcn_model = fcn_model.cuda()

        
def train():
    print("Reached train function")
    best_loss = float('inf')
    val_loss = []
    train_loss = []
    for epoch in range(epochs):
        train_loss_batch = []
        ts = time.time()
        print("Epoch: {}".format(epoch))
        for itera, (X, tar, Y) in enumerate(train_loader):
            #print("Printing the contents of X, tar and Y")
            #print("X, tar, Y are {}, {} and {}".format(type(X), type(tar), type(Y)))
            optimizer.zero_grad()
            if use_gpu:
                inputs = X.cuda()# Move your inputs onto the gpu
                #labels = tar.long()
                train_labels = Y.cuda()# Move your labels onto the gpu
            else:
                inputs, train_labels = X, Y#.long() # Unpack variables into inputs and labels

            outputs = fcn_model(inputs)
            loss = criterion(outputs, train_labels)
            loss = torch.unsqueeze(loss,0)
            loss = loss.mean()
            train_loss_batch.append(loss.item())
            loss.backward()
            optimizer.step()
            
            if itera % 100 == 0:
                print("TRAINING: epoch{}, iter{}, loss: {}".format(epoch, itera, loss.item())) 
        
        print("Finish epoch {}, time elapsed {}".format(epoch, time.time() - ts))
        train_loss.append(np.mean(np.array(train_loss_batch)))
        curr_val_loss, val_inputs = val(epoch)
        val_loss.append(curr_val_loss)
        if curr_val_loss<best_loss:
            print ("Saving best model after %d epochs." % (epoch))
            best_loss = curr_val_loss
            torch.save(fcn_model, MODEL_NAME)
        if epoch>=5:
            stop = 0
            for i in range(0,5):
                if val_loss[epoch-i] > val_loss[epoch-i-1]:
                    stop = stop + 1
            if stop == 5 :
                print ("EarlyStop after %d epochs." % (epoch))
                return train_loss, val_loss, val_inputs
        torch.save(fcn_model, 'last_saved_model')
        fcn_model.train()
    return train_loss, val_loss, val_inputs

    


def val(epoch):
    fcn_model.eval() # Don't forget to put in eval mode !
    val_loss = []
    val_iou = []
    val_acc = []
    with torch.no_grad():
        for itera, (X, tar, Y) in enumerate(val_loader):
            #print("Printing the contents of X, tar and Y")
            #print("X, tar, Y are {}, {} and {}".format(type(X), type(tar), type(Y)))
            if use_gpu:
                inputs = X.cuda()# Move your inputs onto the gpu
                #labels= tar.long()
                val_labels = Y.cuda()# Move your labels onto the gpu
            else:
                inputs, val_labels = X, Y#.long()# Unpack variables into inputs and labels
            outputs = fcn_model(inputs)
            loss = criterion(outputs, val_labels)
            loss = torch.unsqueeze(loss,0)
            loss = loss.mean()
            val_loss.append(loss.item())
            if itera % 100 == 0:
                print("VALIDATION: iter{}, loss: {}".format(itera, loss.item()))
            predictions = F.softmax(outputs,1)
            predictions = torch.argmax(predictions,dim=1)
            #print("Preds shape = ",predictions.shape) #[2, 256, 256]
            #print("Shape of Y = ",Y.shape) #[2, 256, 256]
            iou_row,avg_iou = iou(predictions,val_labels)
            val_iou.append(avg_iou)
            #print("Val acc = ",pixel_acc(predictions,Y))
            val_acc.append(pixel_acc(predictions,val_labels))
        avg_loss = np.mean(np.asarray(val_loss))
        avg_iou = np.mean(np.asarray(val_iou))
        avg_acc = np.mean(np.asarray(val_acc))
        print("Validation epoch {}: avg_iou = {}, avg_acc = {}".format(epoch,avg_iou,avg_acc))
        return avg_loss, inputs    
    
def test():
	fcn_model = torch.load(MODEL_NAME)
    fcn_model.eval()
    val_iou = []
    val_acc = []
    val_ious_cls = []
    with torch.no_grad():
        for itera, (X, tar, Y) in enumerate(val_loader):
            #print("Printing the contents of X, tar and Y")
            #print("X, tar, Y are {}, {} and {}".format(type(X), type(tar), type(Y)))
            if use_gpu:
                inputs = X.cuda()# Move your inputs onto the gpu
                #labels= tar.long()
                test_labels = Y.cuda()# Move your labels onto the gpu
            else:
                inputs, test_labels = X,Y#.long()# Unpack variables into inputs and labels
            outputs = fcn_model(inputs)
            predictions = F.softmax(outputs,1)
            # create one-hot encoding
            predictions = torch.argmax(predictions,dim=1)
            #print("Preds shape = ",predictions.shape) #[2, 256, 256]
            #print("Shape of Y = ",Y.shape) #[2, 256, 256]
            iou_row,avg_iou = iou(predictions,test_labels)
            val_ious_cls.append(iou_row)
            val_iou.append(avg_iou)
            val_acc.append(pixel_acc(predictions,test_labels))
        avg_iou = np.mean(np.asarray(val_iou))
        avg_acc = np.mean(np.asarray(val_acc))
        avg_ious_cls = np.nanmean(np.asarray(val_ious_cls),axis=0) #iou for the class when it's union=0 will be nan
        print("Final test from best model : avg_iou = {}, avg_acc = {}".format(avg_iou,avg_acc))
        print(" Class wise ious getting saved in Baseline_IOU_Classwise.csv file")
        
        
        d = []
        labels_len = len(labels)
        for idx in range(0,labels_len-1):
             d.append((labels[idx].name, avg_ious_cls[labels[idx].level3Id]))
        df = pd.DataFrame(d, columns=('Label name', 'IoU'))
        df.to_csv(MODEL_NAME+"IOU_Classwise.csv", sep='\t')

        test_loader = DataLoader(dataset=test_dataset, batch_size= 1, num_workers=8, shuffle=False)
        for itera, (X, tar, Y) in enumerate(test_loader):
            if use_gpu:
                inputs = X.cuda()# Move your inputs onto the gpu
                test_labels = Y.cuda()# Move your labels onto the gpu
            else:
                inputs, test_labels = X, Y#.long() # Unpack variables into inputs and labels
            outputs = fcn_model(inputs)
            predictions = torch.nn.functional.softmax(outputs,1)
            predictions = torch.argmax(predictions,dim=1)
            break
        predictions = predictions.cpu().numpy()
        inputImage = inputs[0].permute(1, 2, 0).cpu().numpy()
        plt.imshow(inputImage)#, cmap='gray')
        plt.show()
        rows, cols = predictions.shape[1], predictions.shape[2]
        #print(labels)
        new_predictions = np.zeros((predictions.shape[1], predictions.shape[2], 3))
        #unique_labels = set()
        for row in range(rows):
            for col in range(cols):
                idx = int(predictions[0][row][col])
                #unique_labels.add(labels[idx].name)
                new_predictions[row][col][:] = np.asarray(labels[idx].color)/255       

        #print("Detected labels in the image: ",unique_labels)
        plt.imshow(inputImage)#, cmap='gray')
        plt.imshow(new_predictions,alpha=0.5)#, cmap='jet', alpha=0.5)
        fig_name = MODEL_NAME+"Overlayed.jpg"  
        plt.savefig(fig_name)
        plt.show()

    
if __name__ == "__main__":
    val(0)  # show the accuracy before training
    train_loss, val_loss, val_inputs = train()
    plot_loss_curves(train_loss,val_loss,MODEL_NAME)