def iou(pred, target):
    """
        Computes the intersection over union metric
        pred - Prediction
        target - Target sample
    """
    ious = []
    for i in range(len(pred)):
        for cls in range(n_class):
            # Complete this function
            #Gautham: Completed assuming pred and target are tuples with each image in them being for a particular class
            intersection = numpy.logical_and(pred[i][cls], target[i][cls])# intersection calculation
            union = numpy.logical_or(pred[i][cls], target[i][cls])#Union calculation
            if union == 0:
                ious[i].append(float('nan'))  # if there is no ground truth, do not include in evaluation
            else:
                ious[i].append(float(intersection/union))# Append the calculated IoU to the list ious
        avg_iou.append(np.mean(np.array(ious[i][0:26])))
    return ious


def pixel_acc(pred, target):
    """
        Computes the pixel accuracy for a given prediction
        Target
    """
    #Complete this function
    accs = []
    for i in range(len(pred)):
        h, w = target[i][0].shape #shape of prediction in each image per class
        for cls in range(n_class):
            if cls!=26:
                accs[i].append(float(sum(pred[i][cls] == target[i][cls])/(h*w)))
        avg_acc = np.mean(np.array(accs[i][:]))   
    return avg_acc
