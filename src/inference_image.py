import torch
import argparse
import cv2
import os

from utils import get_segment_labels, draw_segmentation_map, get_mask_by_color, overlayMasks, replace_color
from PIL import Image
from config import ALL_CLASSES, VIS_LABEL_MAP
from model import prepare_model

# Construct the argument parser.
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', help='path to input dir', default='../input/inference_data/images')
parser.add_argument(
    '--model',
    default='../outputs_consensus_Batch3-7/model.pth',
    help='path to the model checkpoint'
)
parser.add_argument('-o', '--output', help='path to output dir', default=os.path.join('..', 'outputs', 'inference_results'))

args = parser.parse_args()

out_dir = args.output
os.makedirs(out_dir, exist_ok=True)
os.makedirs(os.path.join(out_dir,'final'), exist_ok=True)
os.makedirs(os.path.join(out_dir,'mask'), exist_ok=True)
os.makedirs(os.path.join(out_dir,'mask/Treat'), exist_ok=True)
os.makedirs(os.path.join(out_dir,'mask/Check'), exist_ok=True)


# Set computation device.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')


model = prepare_model(num_classes=len(ALL_CLASSES)).to(device)
ckpt = torch.load(args.model)
model.load_state_dict(ckpt['model_state_dict'])
model.eval().to(device)

all_image_paths = os.listdir(args.input)
for i, image_path in enumerate(all_image_paths):
    # print(f"Image {i + 1}:", os.path.join(args.input, image_path))
    # if '../input/inference_data/images/FCF1_GY_20221017_040_VID001_anon_trim1.mp4_00134.png' != os.path.join(args.input, image_path):
    #     continue
    # Read the image.
    image = Image.open(os.path.join(args.input, image_path))
    fr_width, fr_height = image.size
    image = image.resize((512, 512))

    # Do forward pass and get the output dictionary.
    outputs = get_segment_labels(image, model, device)
    outputs = outputs['out']
    segmented_image = draw_segmentation_map(outputs)
    mask1 = get_mask_by_color(segmented_image, VIS_LABEL_MAP[1])
    mask2 = get_mask_by_color(segmented_image, VIS_LABEL_MAP[2])
    final_image = overlayMasks(image, mask1, mask2)
    # cv2.imshow('Segmented image', final_image)
    # cv2.waitKey(1)
    cv2.imwrite(os.path.join(out_dir,'final', image_path),
                cv2.cvtColor(cv2.resize(final_image, (fr_width, fr_height), interpolation=cv2.INTER_AREA),
                             cv2.COLOR_BGR2RGB))
    print('saved')

    mask1 = replace_color(mask1, (255, 0, 0), (255, 255, 255))
    mask1.resize((fr_width, fr_height)).save(os.path.join(out_dir , 'mask/Treat/', image_path ))
    mask2 = replace_color(mask2, (0, 255, 0), (255, 255, 255))
    mask2.resize((fr_width, fr_height)).save(os.path.join(out_dir , 'mask/Check/', image_path ))
