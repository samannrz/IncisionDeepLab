import os.path

import cv2
import numpy as np
from statistics import mean

path_ref = '/data/projects/IncisionDeepLab/input/inference_data/'
path_ref = '/data/projects/IncisionDeepLab/input/inference_data/'
path_inference = '/data/DATA/Incision_predictions/Batch8-9/GG'
# path_inference = '/data/DATA/incision/temp/1'


def calculate_iou(mask1, mask2):
    epsilon = 1e-15
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    iou = np.sum(intersection) / (np.sum(union) + epsilon)
    if np.sum(intersection) == 0 and np.sum(union) == 0:
        iou = 1
    return iou


def calculate_confusion_matrix(mask_GT, mask_pred):
    epsilon = 1e-15

    intersection = np.logical_and(mask_GT, mask_pred)
    TP = np.sum(intersection)
    FN = np.sum(mask_GT) - TP
    FP = np.sum(mask_pred) - TP
    TN = np.sum(not mask_pred) - FN
    TN = np.sum(np.logical_or(not mask_pred, not mask_GT))

    return TP, FP, TN, FN


ious_treat = []
ious_check = []
mask_list = []
inf_masks = os.listdir(os.path.join(path_inference, 'mask', 'Treat'))
for mask in inf_masks:
    mask_ref = cv2.imread(os.path.join(path_ref, 'mask', 'Treat', mask))
    mask_inf = cv2.imread(os.path.join(path_inference, 'mask', 'Treat', mask))
    mask_list.append(mask)
    ious_treat.append(calculate_iou(mask_ref, mask_inf))
for mask in inf_masks:
    mask_ref = cv2.imread(os.path.join(path_ref, 'mask', 'Check', mask))
    mask_inf = cv2.imread(os.path.join(path_inference, 'mask', 'Check', mask))
    ious_check.append(calculate_iou(mask_ref, mask_inf))
print(len(mask_list))
print(mask_list)
print(ious_treat)
print(ious_check)
print('Treat IOU = ', mean(ious_treat))
print('Check IOU = ', mean(ious_check))
