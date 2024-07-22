
import torch
import random
import torch.nn.functional as F


# accept alpha channel and epsilon value, return trimap and modified alpha channel
def make_trimap(alpha, epsilon=0.01, dilation_size=None, apply_closing=False, ignore_mask=None, device='cuda'):
    # Adjust alpha values based on epsilon to avoid precision issues
    import time
    alpha = alpha.to(device)
    alpha = torch.where(alpha < epsilon, torch.zeros_like(alpha), alpha)
    alpha = torch.where(alpha > 1 - epsilon, torch.ones_like(alpha), alpha)

    # Generate initial trimap based on alpha values
    trimap = ((alpha > 0) & (alpha < 1)).float()

    import time
    start = time.time()
    # Optionally apply closing operation before dilation
    if apply_closing:
        original_trimap = trimap.clone()
        kernel_size = random.randint(0, 5)
        trimap = close_trimap(trimap, kernel_size, ignore_mask)
    start = time.time()
    # Apply dilation to the trimap
    dilation_kernel = dilation_size
    trimap = dilate_trimap(trimap, dilation_kernel)
    # Combine original and processed trimaps if closing was applied
    if apply_closing:
        trimap = trimap + original_trimap

    # Classify trimap into background, uncertain, and foreground regions
    trimap_final = classify_trimap(trimap, alpha)

    # Return the final one-hot encoded trimap and modified alpha channel
    return one_hot_encode_trimap(trimap_final), alpha


def close_trimap(trimap, kernel_size, ignore_mask):
    """Apply morphological closing to the trimap."""
    trimap = 1. - trimap
    if ignore_mask is not None:
        trimap[ignore_mask] = 0
    trimap = F.max_pool2d(trimap, kernel_size=kernel_size*2+1, stride=1, padding=kernel_size)
    trimap = 1. - trimap
    return trimap


def dilate_trimap(trimap, kernel_size):
    """Dilate the trimap using a specified kernel size."""
    return F.max_pool2d(trimap, kernel_size=kernel_size*2+1, stride=1, padding=kernel_size)


def classify_trimap(trimap, alpha):
    """Classify regions of the trimap into background, uncertain, and foreground."""
    trimap_classes = torch.where(trimap > 0.5, torch.ones_like(alpha), 2 * (alpha > 0.5)).long()
    return trimap_classes


def one_hot_encode_trimap(trimap):
    """Convert the trimap to a one-hot encoded format."""
    return F.one_hot(trimap.squeeze(1), num_classes=3).permute(0, 3, 1, 2).float()