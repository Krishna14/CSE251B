import numpy as np
import torch 
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
    ious = []
    total = torch.zeros(pred.shape[0]).long().cuda()
    int_sum = torch.zeros(pred.shape[0]).long().cuda()
    n_class = 27
    for cls in range(0,n_class):
        # Complete this function
        tens = torch.Tensor(np.full(pred.shape, cls)).long().cuda()
        a = (pred == tens)
        b = (target == tens)
        intersection = torch.sum( a & b , dim=(1,2))
        int_sum = int_sum + intersection
        union = torch.sum( a | b, dim=(1,2))
        iou = torch.div(intersection, union.type(torch.LongTensor).cuda())
        iou[iou != iou] = 0
        ious.append(float(torch.sum(iou)/pred.shape[0]))
        #ious_disc = ious[:-1] #discarding class=26
#     avg_iou = np.mean(ious_disc) 
    ious = np.array(ious)
    return ious, np.mean(ious)
    
def pixel_acc(pred, target):
    torch.backends.cudnn.enabled = True
    ious = []
    total = torch.zeros(pred.shape[0]).long().cuda()
    int_sum = torch.zeros(pred.shape[0]).long().cuda()
    n_class = 27
    for cls in range(0,n_class):
        # Complete this function
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