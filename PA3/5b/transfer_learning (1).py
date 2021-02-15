from torchvision import utils
from dataloader import *
from utils import *
import torchvision
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import time
from matplotlib import pyplot as plt
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision.models.segmentation.fcn import FCNHead

class dice_coefficient_loss(torch.nn.Module):
    def init(self):
        super(dice_coefficient_loss, self).init()
        
    def forward(self,pred, target):
        # skip the batch and class axis for calculating Dice score (bxcx......)
        axes = tuple(range(2, len(pred.shape))) 
        #we can approximate |A∩B| as the element-wise multiplication between the prediction and target, and then sum the resulting matrix.
        #common activations between our prediction and target
        numerator = 2 * torch.sum(pred * target,axes) 
        #quantity of activations in pred & target separately
        denominator = torch.sum((pred*pred) + (target*target),axes)
        #formulate a loss function which can be minimized, we'll simply use 1−Dice, same effect as normalizing loss
        return 1 - torch.mean((numerator + 1e-6) / (denominator + 1e-6)) 
    
def plot_loss_curves(train_loss,val_loss,MODEL_NAME):
    title = "Loss "
    fig_name = MODEL_NAME+"_loss.jpg"
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
    
train_dataset = IddDataset(csv_file='train.csv',transforms_=False)
val_dataset = IddDataset(csv_file='val.csv')
test_dataset = IddDataset(csv_file='test.csv')

train_loader = DataLoader(dataset=train_dataset, batch_size= 4, num_workers=8, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size= 4, num_workers=8, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size= 4, num_workers=8, shuffle=False)


print(len(train_dataset),train_loader.__len__())
print(len(val_dataset))
print(len(test_dataset))
    


n_class = 27
deeplab_v3 = torch.hub.load('pytorch/vision:v0.6.0', 'deeplabv3_resnet101', pretrained=True)

for param in deeplab_v3.parameters():
    param.requires_grad = False

deeplab_v3.classifier = DeepLabHead(2048, n_class)
# deeplab_v3.aux_classifier = FCNHead(1024, n_class)
# deeplab_v3.classifier.requires_grad =
# deeplab_v3.classifier[4] = nn.Conv2d(
#     in_channels=256,
#     out_channels=n_class,
#     kernel_size=1,
#     stride=1
# )

for param in deeplab_v3.parameters():
    print(param.requires_grad)
# Set the model in training mode
# deeplab_v3.train()
# deeplab_v3.eval()
print(deeplab_v3)

epochs = 50       
criterion = dice_coefficient_loss()
# deep_lab = deeplab_v3(n_class=n_class)

optimizer = optim.Adam(deeplab_v3.parameters(), lr=5e-3)
MODEL_NAME = "best_model_5b_"

use_gpu = torch.cuda.is_available()
if use_gpu:
    print("Using GPU")
    deeplab_v3 = deeplab_v3.cuda()

        
def train():
    deeplab_v3.train()
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
                train_targets = tar.cuda()
            else:
                inputs, train_labels,train_targets = X, Y,tar # Unpack variables into inputs and labels

            outputs = deeplab_v3(inputs)['out']
            #loss = criterion(outputs, train_labels)
            predictions = F.softmax(outputs,1)
            loss = criterion(predictions,train_targets)
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
            torch.save(deeplab_v3, MODEL_NAME)
        if epoch>=5:
            stop = 0
            for i in range(0,5):
                if val_loss[epoch-i] > val_loss[epoch-i-1]:
                    stop = stop + 1
            if stop == 5 :
                print ("EarlyStop after %d epochs." % (epoch))
                return train_loss, val_loss, val_inputs
        torch.save(deeplab_v3, 'last_saved_model')
        deeplab_v3.train()
    return train_loss, val_loss, val_inputs


def val(epoch):
    deeplab_v3.eval() # Don't forget to put in eval mode !
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
                val_targets = tar.cuda()
            else:
                inputs, val_labels,val_targets = X, Y,tar#.long()# Unpack variables into inputs and labels
            outputs = deeplab_v3(inputs)['out']
            #loss = criterion(outputs, val_labels)
            predictions = F.softmax(outputs,1)
#             print(outputs.shape)
#             print(predictions.shape)
#             print(inputs.shape)
#             print(outputs.shape)
#             print(val_targets.shape)
            loss = criterion(predictions,val_targets)
            loss = torch.unsqueeze(loss,0)
            loss = loss.mean()
            val_loss.append(loss.item())
            if itera % 100 == 0:
                print("VALIDATION: iter{}, loss: {}".format(itera, loss.item()))
            #predictions = F.softmax(outputs,1)
            predictions = torch.argmax(predictions,dim=1)
            #print("Preds shape = ",predictions.shape) #[2, 256, 256]
            #print("Shape of Y = ",Y.shape) #[2, 256, 256]
            iou_row,avg_iou = iou(predictions,val_labels)
            val_iou.append(avg_iou)
            val_acc.append(pixel_acc(predictions,val_labels))
            #print("Val acc = ",pixel_acc(predictions,Y))
        avg_loss = np.mean(np.asarray(val_loss))
        avg_iou = np.mean(np.asarray(val_iou))
        avg_acc = np.mean(np.asarray(val_acc))
        print("Validation epoch {}: avg_iou = {}, avg_acc = {}".format(epoch,avg_iou,avg_acc))
        return avg_loss, inputs    
    
def test():
    deeplab_v3 = torch.load(MODEL_NAME)
    deeplab_v3.eval()
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
            outputs = deeplab_v3(inputs)['out']
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
            outputs = deeplab_v3(inputs)
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
        plt.axis('off')
        fig_name = MODEL_NAME+"Overlayed.jpg"  
        plt.savefig(fig_name)
        plt.show()

    
if __name__ == "__main__":
    val(0)  # show the accuracy before training
    train_loss, val_loss, val_inputs = train()
#     test()
    plot_loss_curves(train_loss,val_loss,MODEL_NAME)