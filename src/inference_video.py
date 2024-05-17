import cv2
import torch
import argparse
import time
import os

from PIL import Image

from utils import get_segment_labels, draw_segmentation_map, image_overlay,get_mask_by_color, overlayMasks
from config import ALL_CLASSES, VIS_LABEL_MAP
from model import prepare_model

# Construct the argument parser.
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', help='path to input dir', default= '/data/Videos/endodata/sequ/CHU-CF/6/FCF1_GY_20230208_067_VID001_anon_trim1.mp4' )
parser.add_argument(
    '--model',
    default='../outputs_consensus_Batch3-7/model.pth',
    help='path to the model checkpoint'
)
parser.add_argument(
    '--output',
    default='../outputs/inference_results_video',
    help='path to the output dir'
)
args = parser.parse_args()

out_dir = args.output
os.makedirs(out_dir, exist_ok=True)

# Set computation device.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = prepare_model(num_classes=len(ALL_CLASSES)).to(device)
ckpt = torch.load(args.model)
model.load_state_dict(ckpt['model_state_dict'])
model.eval().to(device)
for video_name in os.listdir(args.input):
    cap = cv2.VideoCapture(os.path.join(args.input,video_name))
    if (cap.isOpened() == False):
        print('Error while trying to read video. Please check path again')

    # get the frame width and height
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    save_name = f"{args.input.split('/')[-1].split('.')[0]}"
    save_name = f"{video_name.split('/')[-1].split('.')[0]}"

    #save_name = video_name
    # define codec and create VideoWriter object
    out = cv2.VideoWriter(f"{out_dir}/{save_name}.mp4",
                          cv2.VideoWriter_fourcc(*'mp4v'), 30,
                          (frame_width, frame_height))

    frame_count = 0 # to count total frames
    total_fps = 0 # to get the final frames per second
    print(save_name)
    # read until end of video
    while(cap.isOpened()):
        # capture each frame of the video
        ret, frame = cap.read()
        print(save_name)
        fr_height, fr_width = frame.shape[:2]

        if ret:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame = cv2.resize(rgb_frame, (512, 512))
            # get the start time
            start_time = time.time()
            # Do forward pass and get the output dictionary.
            outputs = get_segment_labels(rgb_frame, model, device)
            outputs = outputs['out']
            segmented_image = draw_segmentation_map(outputs)
            mask1 = get_mask_by_color(segmented_image, VIS_LABEL_MAP[1])
            mask2 = get_mask_by_color(segmented_image, VIS_LABEL_MAP[2])
            frame = Image.fromarray(rgb_frame)

            final_image = overlayMasks(frame, mask1, mask2)



            #final_image = image_overlay(rgb_frame, segmented_image)

            # get the end time
            end_time = time.time()
            # get the current fps
            fps = 1 / (end_time - start_time)
            # add current fps to total fps
            total_fps += fps
            # increment frame count
            frame_count += 1
            # put the FPS text on the current frame
            # cv2.putText(final_image, f"{fps:.3f} FPS", (20, 35),
            #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            # press `q` to exit
            #final_image = cv2.resize(final_image, dsize=(frame_width, frame_height), interpolation=cv2.INTER_CUBIC)

            #cv2.imshow('image', cv2.cvtColor(cv2.resize(final_image, (fr_width, fr_height), interpolation=cv2.INTER_AREA),
            #cv2.COLOR_BGR2RGB))
            out.write(cv2.cvtColor(cv2.resize(final_image, (fr_width, fr_height), interpolation=cv2.INTER_AREA),
                                   cv2.COLOR_BGR2RGB))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    # release VideoCapture()
    cap.release()
    # close all frames and video windows
    cv2.destroyAllWindows()
    # calculate and print the average FPS
    avg_fps = total_fps / frame_count
    print(video_name, ' saved!')
    print(f"Average FPS: {avg_fps:.3f}\n")