def iou(pred, target):
    """
        Computes the intersection over union metric
        pred - Prediction
        target - Target sample
    """
    ious = []
    for cls in range(n_class):
        # Complete this function
        #Gautham: Completed assuming pred and target are tuples with each image in them being for a particular class
        intersection = numpy.logical_and(pred[cls], target[cls])# intersection calculation
        union = numpy.logical_or(pred[cls], target[cls])#Union calculation
        if union == 0:
            ious.append(float('nan'))  # if there is no ground truth, do not include in evaluation
        else:
            ious.append(float(intersection/union))# Append the calculated IoU to the list ious
    avg_iou = np.mean(np.array(ious[0:26]))
    return ious


def pixel_acc(pred, target):
    """
        Computes the pixel accuracy for a given prediction
        Target
    """
    #Complete this function
    accs = []
    h, w = label.target[0]
    for cls in range(n_class):
        if cls!=26:
            accs.append(float(sum(pred[cls] == target[cls])/(h*w)))
    avg_acc = np.mean(np.array(accs))   
    return avg_acc
