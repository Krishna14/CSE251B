def iou(pred,target):
    ious = []
    avg_iou = []
    for i in range(len(pred)):
        for cls in range(n_class): # Go over all classes
            intersection=float(sum((pred[i][:]==target[i][:])*(target[i][:]==cls)))# per class Intersection
            union=sum(target[i][:]==cls)+sum(pred[i][:]==cls)-intersection # per class Union
            if union == 0:
                ious[i].append(float('nan'))  # if there is no ground truth, do not include in evaluation
            else:
                ious[i].append(intersection/union)# Append the calculated IoU to the list ious
        avg_iou[i] = np.mean(np.array(ious[i][0:26])) #discarding class=26
    return ious

def pixel_acc(pred, target):
    accs = []
    for i in range(len(pred)):
        h, w = target[i].shape
        for cls in range(n_class):
            #shape of prediction in each image per class
            if cls!=26:
                accs[i].append(float(sum(pred[i][cls] == target[i][cls])/(h*w)))
        avg_acc = np.mean(np.array(accs[i][:]))   
    return avg_acc
