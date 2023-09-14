import os.path

import cv2
import numpy as np
from statistics import mean

path_ref = '/data/projects/IncisionDeepLab/input/inference_data/'
path_ref = '/data/DATA/incision/4/'
path_inference = '/data/DATA/Incision_predictions/Batch8-9/all_data/'


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
treat_row = []
check_row = []
bg_row = []

inf_masks = os.listdir(os.path.join(path_inference, 'mask', 'Treat'))
for mask in inf_masks:
    mask_ref_treat = cv2.imread(os.path.join(path_ref, 'mask', 'Treat', mask))
    mask_inf_treat = cv2.imread(os.path.join(path_inference, 'mask', 'Treat', mask))
    mask_list.append(mask)
    ious_treat.append(calculate_iou(mask_ref_treat, mask_inf_treat))
    TP_treat, FP_treat, TN_treat, FN_treat = calculate_confusion_matrix(mask_ref_treat, mask_inf_treat)
    sensitivity_treat.append(TP_treat / ((TP_treat + FN_treat) + epsilon))
    specificity_treat.append(TN_treat / ((TN_treat + FP_treat) + epsilon))
    specificity_treat.append(TN_treat / ((TN_treat + FP_treat) + epsilon))
    f_treat.append(((2 * TP_treat) + epsilon) / ((2 * TP_treat + FP_treat + FN_treat) + epsilon))

    mask_ref_check = cv2.imread(os.path.join(path_ref, 'mask', 'Check', mask))
    mask_inf_check = cv2.imread(os.path.join(path_inference, 'mask', 'Check', mask))
    ious_check.append(calculate_iou(mask_ref_check, mask_inf_check))
    TP_check, FP_check, TN_check, FN_check = calculate_confusion_matrix(mask_ref_check, mask_inf_check)
    sensitivity_check.append(TP_check / ((TP_check + FN_check) + epsilon))
    specificity_check.append(TN_check / ((TN_check + FP_check) + epsilon))
    f_check.append(2 * TP_check / (2 * TP_check + FP_check + FN_check))

    mask_ref_treat = mask_ref_treat != 0
    mask_ref_check = mask_ref_check != 0

    if np.sum(mask_ref_treat) != 0:
        treat_row.append([TP_treat,
                          np.sum(np.logical_and(mask_ref_treat, mask_inf_check)),
                          np.sum(np.logical_and(mask_ref_treat, ~np.logical_or(mask_inf_check, mask_inf_treat)))] / (
                             np.sum(mask_ref_treat)))
    if np.sum(mask_ref_check) != 0:
        check_row.append([np.sum(np.logical_and(mask_ref_check, mask_inf_treat)),
                          TP_check,
                          np.sum(np.logical_and(mask_ref_check, ~np.logical_or(mask_inf_check, mask_inf_treat)))] / (
                             np.sum(mask_ref_check)))

    mask_ref_bg = np.logical_and(~ mask_ref_check, ~ mask_ref_treat)
    mask_inf_bg = np.logical_and(~ mask_inf_check, ~ mask_inf_treat)
    bg_row.append([np.sum(np.logical_and(mask_ref_bg, mask_inf_treat)),
                   np.sum(np.logical_and(mask_ref_bg, mask_inf_check)),
                   np.sum(np.logical_and(mask_ref_bg, mask_inf_bg))] / np.sum(mask_ref_bg))
    # print((bg_row))
    # print(treat_row)
    # print(check_row)

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
conf_matrix = [list(np.mean(treat_row, 0)), list(np.mean(check_row, 0)), list(np.mean(bg_row, 0))]
print('conf_matrix_treat : ', np.mean(treat_row, 0))
print('conf_matrix_check : ', np.mean(check_row, 0))
print('conf_matrix_backg : ', np.mean(bg_row, 0))

import numpy as np
import matplotlib.pyplot as plt

plt.imshow(conf_matrix,cmap='Blues')
for i in range(3):
    for j in range(3):
        c = conf_matrix[j][i]
        plt.text(i, j, str(round(c,3)), va='center', ha='center')
plt.text(0, -.6, 'treat', va='center', ha='center')
plt.text(1, -.6, 'check', va='center', ha='center')
plt.text(2, -.6, 'background', va='center', ha='center')
plt.text(-.6, 0, 'treat', va='center', ha='center',rotation = 'vertical')
plt.text(-.6, 1, 'check', va='center', ha='center',rotation = 'vertical')
plt.text(-.6, 2, 'background', va='center', ha='center',rotation = 'vertical')
plt.text(1, -.8, 'Prediction', va='center', ha='center')
plt.text(-.8, 1, 'Ground Truth', va='center', ha='center', rotation = 'vertical')

plt.xticks([])
plt.yticks([])
plt.savefig('all-data.png')