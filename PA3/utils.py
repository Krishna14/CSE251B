import numpy as np
def iou(pred,target):
    pred = pred.cpu().numpy()
    target = target.cpu().numpy()
    ious = [None]*len(pred) 
    avg_iou = []
    n_class = 27
    for i in range(len(pred)):
        temp_row = []
        for cls in range(n_class): # Go over all classes
            intersection=np.float32(np.sum((pred[i][:]==target[i][:])*(target[i][:]==cls)))# per class Intersection
            union=np.sum(target[i][:]==cls)+np.sum(pred[i][:]==cls)-intersection # per class Union
            if union == 0:
                #temp_row.append(np.float32('nan'))  # if there is no ground truth, do not include in evaluation
                temp_row.append(0) 
            else:
                temp_row.append(intersection/union)# Append the calculated IoU to the list ious
        ious[i] = temp_row
    ious = np.array(ious)
    ious = np.mean(ious, axis=0)
    ious_disc = ious[:-1] #discarding class=26
    avg_iou = np.mean(ious_disc) 
    return ious,avg_iou

def pixel_acc(pred, target):
    pred = pred.cpu().numpy()
    target = target.cpu().numpy()
    accs = [None]*len(pred)
    n_class = 27
    for i in range(len(pred)):
        temp_row = []
        for cls in range(n_class):
            #shape of prediction in each image per class
            intersection=np.float32(np.sum((pred[i][:]==target[i][:])*(target[i][:]==cls)))
            denominator = np.sum(target[i][:]==cls)
            if denominator == 0:
                #temp_row.append(np.float32('nan'))  # if there is no ground truth, do not include in evaluation
                temp_row.append(0) 
            else:
                temp_row.append(intersection/denominator)# Append the calculated IoU to the list ious
        
        accs[i] = temp_row
    accs = np.array(accs)
    accs = accs[:-1]
    avg_acc = np.mean(accs)   
    return avg_acc