def iou(pred, target):
    """
        Computes the intersection over union metric
        pred - Prediction
        target - Target sample
    """
    ious = []
    for cls in range(n_class):
        # Complete this function
        intersection = target.intersection(pred)# intersection calculation
        union = target.union(pred)#Union calculation
        if union == 0:
            ious.append(float('nan'))  # if there is no ground truth, do not include in evaluation
        else:
            # Append the calculated IoU to the list ious
            ious.append(intersection/union)
    return ious


def pixel_acc(pred, target):
    """
        Computes the pixel accuracy for a given prediction
        Target
    """
    #Complete this function
    return sum(pred == target)/len(target)