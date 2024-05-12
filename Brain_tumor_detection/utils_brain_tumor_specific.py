import torch
from torchvision.transforms import v2 as T


def get_transform(train=True):
  """generates a set of image transformations based on whether it's for training or not.
  For training, it includes a random horizontal flip with a probability of 0.5. Both for training and evaluation,
  it converts the image to a float tensor and then to a pure tensor. It returns the composed transformations."""
  transforms = []
  if train:
    transforms.append(T.RandomHorizontalFlip(0.5))

  transforms.append(T.ToDtype(torch.float, scale=True))
  transforms.append(T.ToPureTensor())
  return T.Compose(transforms)


def collate_fn(batch):
  """It takes a batch of samples, typically tuples of images and their corresponding targets,
  and returns a transposed tuple of batches, which separates images and targets.
  This function is used during data loading to format batches for training or evaluation."""
  return tuple(zip(*batch))