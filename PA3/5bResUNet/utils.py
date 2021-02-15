import numpy as np
import torch

def iou(pred, target, compute_pix_acc=True):
    """
        Compute both IoU and pixel accuracy for the given parameters
    """
    torch.backends.cudnn.enabled = True
    ious = []
    total = torch.zeros(pred.shape[0]).long().cuda()
    union = torch.zeros(pred.shape[0]).long().cuda()
    int_sum = torch.zeros(pred.shape[0]).long().cuda()
    uni_sum = torch.zeros(pred.shape[0]).long().cuda()
    n_class = 27
    ious = []
    
    # Here, we don't include all the classes for the computation of the image.
    for cls in range(n_class-1):
        # Create a temporary variable
        tensor = torch.Tensor(np.full(pred.shape, cls)).long().cuda()
        # Compare these values
        a = (pred == tensor)
        b = (target == tensor)
        # Intersection - Both layers are providing same class label
        intersection = torch.sum(a & b, dim=(1, 2))
        # Union - Either of layers providing the same class label
        union = torch.sum(a | b, dim=(1, 2))
        # Computing sum values
        uni_sum = uni_sum + union
        int_sum = int_sum + intersection
        # Computing the total number of values in the pixel
        total = total + torch.sum(b, dim=(1, 2))
        
        iou = torch.Tensor.float(torch.Tensor.float(intersection)/torch.Tensor.float(union))
        # To avoid the presence of "nan"
        iou = iou[union != 0]
        iou = torch.mean(iou)
        # Inserting iou to the list of ious.
        ious.append(iou)
        
    int_sum = torch.Tensor.float(int_sum)
    uni_sum = torch.Tensor.float(uni_sum)
    total = torch.Tensor.float(total)
    avg_iou = torch.mean(torch.Tensor.float(int_sum/uni_sum))
    pix_acc = torch.mean(torch.Tensor.float(int_sum/total))
            
    return ious, float(avg_iou), float(pix_acc)