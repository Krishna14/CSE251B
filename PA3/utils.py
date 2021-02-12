import numpy as np
import torch 

class dice_coefficient_loss(torch.nn.Module):
    def init(self):
        super(dice_coefficient_loss, self).init()
        
    def forward(self, pred, target):
        # skip the batch and class axis for calculating Dice score (bxcx......)
        axes = tuple(range(2, len(pred.shape))) 
        #we can approximate |A∩B| as the element-wise multiplication between the prediction and target, and then sum the resulting matrix.
        #common activations between our prediction and target
        numerator = 2 * torch.sum(pred * target,axes) 
        #quantity of activations in pred & target separately
        denominator = torch.sum((pred*pred) + (target*target),axes)
        #formulate a loss function which can be minimized, we'll simply use 1−Dice, same effect as normalizing loss
        return 1 - torch.mean((numerator + 1e-6) / (denominator + 1e-6)) 
    
# def iou(pred,target):
#     pred = pred.cpu().numpy()
#     target = target.cpu().numpy()
#     ious = [None]*len(pred) 
#     avg_iou = []
#     n_class = 27
#     for i in range(0,len(pred)):
#         temp_row = []
#         for cls in range(n_class): # Go over all classes
#             intersection=np.float32(np.sum((pred[i][:]==target[i][:])*(target[i][:]==cls)))# per class Intersection
#             union=np.sum(target[i][:]==cls)+np.sum(pred[i][:]==cls)-intersection # per class Union
#             if union == 0:
#                 #temp_row.append(np.float32('nan'))  # if there is no ground truth, do not include in evaluation
#                 temp_row.append(0) 
#             else:
#                 temp_row.append(intersection/union)# Append the calculated IoU to the list ious
#         ious[i] = temp_row
#     ious = np.array(ious)
#     ious = np.mean(ious, axis=0)
#     ious_disc = ious[:-1] #discarding class=26
#     avg_iou = np.mean(ious_disc) 
#     return ious,avg_iou

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

# def iou(pred, target):
#     torch.backends.cudnn.enabled = True
#     ious = []
#     n_class = 27
#     for cls in range(0,n_class-1):
#         # Complete this function
#         tens = torch.Tensor(np.full(pred.shape, cls)).long().cuda()
#         a = (pred == tens)
#         b = (target == tens)
#         intersection = torch.sum( a & b , dim=(1,2))
#         union = torch.sum( a | b, dim=(1,2))
#         iou = torch.div(intersection, union.type(torch.LongTensor).cuda())
#         iou[iou != iou] = 0
#         iou_mean = torch.mean(torch.Tensor.float(iou))
#         ious.append(iou_mean)
#         print("Appended {} to ious[{}]".format(iou_mean,cls))
#         #ious.append(float(torch.sum(iou)/pred.shape[0]))
#         #ious_disc = ious[:-1] #discarding class=26
# #     avg_iou = np.mean(ious_disc) 
#     ious = np.array(ious)
#     avg_iou = float(np.sum(ious)/len(ious))
#     print("Average iou before returning = ",avg_iou) #-> Messed up
#     return ious,avg_iou
    
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


# def pixel_acc(pred, target):
#     pred = pred.cpu().numpy()
#     target = target.cpu().numpy()
#     accs = [0]*len(pred)
#     n_class = 27
#     for i in range(0,len(pred)):
#         temp_row = []
#         for cls in range(n_class):
#             #shape of prediction in each image per class
#             intersection=np.float32(np.sum((pred[i][:]==target[i][:])*(target[i][:]==cls)))
#             denominator = np.sum(target[i][:]==cls)
#             if denominator == 0:
#                 #temp_row.append(np.float32('nan'))  # if there is no ground truth, do not include in evaluation
#                 temp_row.append(0) 
#             else:
#                 temp_row.append(intersection/denominator)# Append the calculated IoU to the list ious
#         accs[i] = temp_row
#     accs = np.array(accs)
#     accs = accs[:-1]
#     if(len(accs) >= 1):
#         avg_acc = np.mean(accs)
#     else:
#         #print("Average accuracy length = {}".format(len(accs)))
#         avg_acc = 0
#     return avg_acc