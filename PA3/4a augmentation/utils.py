import numpy as np
import torch 

def iou(pred, target):
    torch.backends.cudnn.enabled = True
    torch.set_printoptions(precision=8)
    ious = []
    int_sum = torch.zeros(pred.shape[0]).long().cuda()
    uni_sum = torch.zeros(pred.shape[0]).long().cuda()
    n_class = 27
    for cls in range(0,n_class-1): #do not include unlabeled class (26)
        tens = torch.Tensor(np.full(pred.shape, cls)).long().cuda()
        a = (pred == tens)
        b = (target == tens)
        intersection = torch.sum( a & b , dim=(1,2))
        #print("Intersection for cls {} = {}".format(cls,intersection))
        int_sum = int_sum + intersection
        union = torch.sum( a | b, dim=(1,2))
        #print("Union for cls {} = {}".format(cls,union))
        #iou = torch.Tensor.float(torch.div(intersection.type(torch.LongTensor), union.type(torch.LongTensor)).cuda())
        iou = torch.Tensor.float(torch.Tensor.float(intersection)/torch.Tensor.float(union))
        iou = iou[union!=0]
        iou = torch.mean(iou)    
        uni_sum = uni_sum + union
        #print("Appending to ious: ",iou)
        ious.append(iou)
    int_sum = torch.Tensor.float(int_sum)
    uni_sum = torch.Tensor.float(uni_sum)
    avg_iou = torch.mean(torch.Tensor.float(int_sum/uni_sum))
    #print("Returning back from utils.py iou")
    return ious,float(avg_iou)
    
def pixel_acc(pred, target):
    torch.backends.cudnn.enabled = True
    ious = []
    total = torch.zeros(pred.shape[0]).long().cuda()
    int_sum = torch.zeros(pred.shape[0]).long().cuda()
    n_class = 27
    for cls in range(0,n_class-1): #do not include unlabeled class (26)
        tens = torch.Tensor(np.full(pred.shape, cls)).long().cuda()
        a = (pred == tens)
        b = (target == tens)
        intersection = torch.sum( a & b , dim=(1,2))
        int_sum = int_sum + intersection
        total = total + torch.sum(b, dim=(1,2))
    int_sum = torch.Tensor.float(int_sum)
    total = torch.Tensor.float(total)
    pixel_acc = torch.mean(torch.Tensor.float(int_sum/total))
    return float(pixel_acc)

def plot_loss_curves(train_loss,val_loss):
    title = "Loss "
    fig_name = "Loss_baseline.jpg"
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
