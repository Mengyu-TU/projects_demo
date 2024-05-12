# README Brain Tumor Detection with Faster R-CNN and YOLO

## **The folder contains the following files**:
1. `faster_R_CNN_brain_tumor_notebook.ipynb`
   - This contains the **training log of the faster R-CNN model**
   - This is the entry point to train and test on the faster R-CNN model.

2. `yolo8_brain_tumor.ipynb`
   - This contains the **training log of the YOLO model**
   - This is the entry point to train and test on the YOLO model.

3. `dataset_class.py`
   - Custom dataset class for the brain tumor dataset_class

4. `train_n_eval.py`
   - Contains the training function and the evaluation function

5. `utils_brain_tumor_specific.py`
   - Contains the utility functions specific to the brain tumor dataset

6. `coco_eval.py`
   - coco-evaluation functions

7. `coco_utils.py`
   - utility functions for coco evaluation

8. `transforms.py`
   - tranforms function used in coco_utils

9. `utils.py`
   - contains the metric logger class for logging metrics.

## **Structure of dataset**: 
- Three subfolders, all follow the similar structure as laid out in `axial_t1wce_2_class`
- .txt labels remarks: 
  - 0: negative/benign tumor; 
  - 1: positive/malicious tumor
- brain_tumor_tidy
  - axial_t1wce_2_class
     - images
        - test
        - train
     - labels
        - test
        - train
     - axial_t1wce_2_class.yaml

  - coronal_t1wce_2_class
     - ...
  - sagittal_t1wce_2_class
    - ...

## Training log can be found in:
1. `faster_R_CNN_brain_tumor_notebook.ipynb`
2. `yolo8_brain_tumor.ipynb`


## **How to Run a faster R-CNN model on the brain tumor dataset**:
1. Put the .py files in the working directory.
2. Edit in the `faster_R_CNN_brain_tumor_notebook.ipynb` how to load the dataset. Currently set to load from Google Drive.
3. Run `faster_R_CNN_brain_tumor_notebook.ipynb`


## **How to Run a YOLO model on the brain tumor dataset**:
1. Edit the path in .yaml file in brain_tumor_tidy subfolder
2. Edit in the `yolo8_brain_tumor.ipynb` how to load the dataset. Currently set to load from Google Drive.
2. Run `yolo8_brain_tumor.ipynb`
