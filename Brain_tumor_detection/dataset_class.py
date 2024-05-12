import os
import torch
from torchvision import transforms as tv_transforms
from torchvision.transforms.v2 import functional as F
from torchvision.io import read_image
from PIL import Image
from torchvision import tv_tensors


class brain_image_Dataset(torch.utils.data.Dataset):
    """The brain_image_Dataset class manages brain image data for object detection tasks.
        It initializes with the root directory, transformation pipeline, and process type.
        Loading images and labels, it transforms images and parses label files for bounding box data.
        Bounding box coordinates are converted to pixel values, and data is organized into a dictionary.
        The __getitem__ method retrieves a single data sample, and __len__ returns the total number of samples. """

    def __init__(self, root, transforms, process="train"):
        self.root = root
        self.transforms = transforms
        self.process = process  # train or test

        # load image all images' names
        self.images = list(sorted(os.listdir(os.path.join(root, "images", process))))
        self.labels = list(sorted(os.listdir(os.path.join(root, "labels", process))))

    def __getitem__(self, idx):
        target = {}
        target["image_id"] = idx

        # load images and bounding-boxes path
        img_path = os.path.join(self.root, "images", self.process, self.images[idx])
        label_path = os.path.join(self.root, "labels", self.process, self.labels[idx])

        # read image and wrap image into torchvision tv_tensors
        img = Image.open(img_path)
        img = tv_tensors.Image(img) / 255.0

        # transform img if self.transforms is not None:
        if self.transforms is not None:
            img = self.transforms(img)

        # Get the image size
        width, height = F.get_size(img)

        # read the .txt file with the labels and bounding boxes
        with open(label_path) as file:
            lines = file.readlines()

        # Extract labels and fractional bounding boxes: works on one line or multiple lines
        labels = [int(line.strip().split()[0])+1 for line in lines]
        bboxes_frac = [line.strip().split()[1:5] for line in lines]
        bboxes_frac = [[float(x) for x in bbox] for bbox in bboxes_frac]

        # Convert fractional bounding boxes to pixel coordinates
        bboxes_pixels = []
        areas = []
        for bbox_frac in bboxes_frac:
            x_center = bbox_frac[0] * width
            y_center = bbox_frac[1] * height
            bbox_width = bbox_frac[2] * width
            bbox_height = bbox_frac[3] * height
            bboxes_pixels.append([
                x_center - bbox_width / 2,  # x_min
                y_center - bbox_height / 2,  # y_min
                x_center + bbox_width / 2,  # x_max
                y_center + bbox_height / 2,  # y_max
            ])
            areas.append(bbox_width * bbox_height)

        target["labels"] = torch.tensor(labels, dtype=torch.int64)
        target["area"] = torch.tensor(areas)
        target["boxes"] = tv_tensors.BoundingBoxes(bboxes_pixels, format="XYXY", canvas_size=F.get_size(img))
        target["iscrowd"] = torch.zeros((len(labels),), dtype=torch.int64)

        return img, target

    def __len__(self):
        return len(self.images)


