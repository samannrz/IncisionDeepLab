import os.path

import cv2
import numpy as np
from statistics import mean

path_ref = '/data/projects/IncisionDeepLab/input/inference_data/'
path_ref = '/data/DATA/incision/4/'
path_inference = '/data/DATA/Incision_predictions/Batch8-9/consensus/'


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
    mask_GT = mask_GT != 0
    mask_pred = mask_pred != 0

    intersection = np.logical_and(mask_GT, mask_pred)
    TP = np.sum(intersection)
    im = intersection.astype(int)
    FN = np.sum(mask_GT) - TP
    FP = np.sum(mask_pred) - TP
    TN = np.sum(~ mask_pred) - FN
    return TP, FP, TN, FN


epsilon = 1e-15

ious_treat = []
ious_check = []
mask_list = []
sensitivity_treat = []
specificity_treat = []
sensitivity_check = []
specificity_check = []
f_treat = []
f_check = []
inf_masks = os.listdir(os.path.join(path_inference, 'mask', 'Treat'))
for mask in inf_masks:
    mask_ref = cv2.imread(os.path.join(path_ref, 'mask', 'Treat', mask))
    mask_inf = cv2.imread(os.path.join(path_inference, 'mask', 'Treat', mask))
    mask_list.append(mask)
    ious_treat.append(calculate_iou(mask_ref, mask_inf))
    TP, FP, TN, FN = calculate_confusion_matrix(mask_ref, mask_inf)
    sensitivity_treat.append(TP / ((TP + FN) + epsilon))
    specificity_treat.append(TN / ((TN + FP) + epsilon))
    specificity_treat.append(TN / ((TN + FP) + epsilon))
    f_treat.append(((2 * TP)+epsilon) / ((2 * TP + FP + FN)+epsilon))

for mask in inf_masks:
    mask_ref = cv2.imread(os.path.join(path_ref, 'mask', 'Check', mask))
    mask_inf = cv2.imread(os.path.join(path_inference, 'mask', 'Check', mask))
    ious_check.append(calculate_iou(mask_ref, mask_inf))
    TP, FP, TN, FN = calculate_confusion_matrix(mask_ref, mask_inf)
    sensitivity_check.append(TP / ((TP + FN) + epsilon))
    specificity_check.append(TN / ((TN + FP) + epsilon))
    f_check.append(2 * TP / (2 * TP + FP + FN))

print(len(mask_list))
print(mask_list)
print(ious_treat)
print(ious_check)
print('Treat IOU = ', mean(ious_treat))
print('Check IOU = ', mean(ious_check))
print('Treat sensitivity = ', mean(sensitivity_treat))
print('Treat specificity = ', mean(specificity_treat))
print('Check sensitivity = ', mean(sensitivity_check))
print('Check specificity = ', mean(specificity_check))
print('Treat F-Score = ', mean(f_treat))
print('Check F-Score = ', mean(f_check))


