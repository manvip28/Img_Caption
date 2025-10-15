import torch

def collate_fn(batch, pad_idx=0):
    """
    batch: list of tuples (image, caption) or None
    pad_idx: index used for <PAD> token
    """
    # Filter out None items
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None, None  # all items were invalid

    images = []
    captions = []

    for img, cap in batch:
        images.append(img)
        captions.append(cap)

    # Stack images: [batch, channels, H, W]
    images = torch.stack(images, 0)

    # Pad captions to the max length in the batch
    lengths = [len(cap) for cap in captions]
    max_len = max(lengths)

    padded_captions = torch.full((len(captions), max_len), pad_idx, dtype=torch.long)
    for i, cap in enumerate(captions):
        end = lengths[i]
        padded_captions[i, :end] = cap

    return images, padded_captions
