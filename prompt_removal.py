import sys
import os

HOME = os.getcwd()
sys.path.append(os.path.join(HOME, 'GroundingDINO'))
sys.path.append(os.path.join(HOME, 'ProPainter'))

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from PIL import Image
import numpy as np
import cv2
import supervision as sv

import torch
import subprocess
from GroundingDINO.groundingdino.util.inference import load_model, load_image, predict, annotate
from torchvision.ops import box_convert

from utils.video import video_to_frame, display_video

import locale
locale.getpreferredencoding = lambda: "UTF-8"

import argparse

# Create the parser
parser = argparse.ArgumentParser(description="Load Prompts")
parser.add_argument('-v', '--VIDEO_NAME', type=str, required=True, help='Name of the video to inpaint')
parser.add_argument('-h_res', '--HIGH', type=bool, default=False, help='Set to True for higher resolution (more GPU memory requirement)')
parser.add_argument('-display', '--DISPLAY', type=bool, default=True, help='Set to True to display video results')
parser.add_argument('-box_t', '--BOX_THRESHOLD', type=float, default=0.35, help='DINO Bounding Box threshold')
parser.add_argument('-text_t', '--TEXT_THRESHOLD', type=float, default=0.25, help='DINO Textual Prompt threshold')
args = parser.parse_args()

PROMPT = input("Enter a Textual Prompt: ")
assert type(PROMPT) == str, 'The Prompt should be of type str'

## Variables
device = 'cuda' if torch.cuda.is_available() else 'cpu'

BOX_THRESHOLD = args.BOX_THRESHOLD
TEXT_THRESHOLD = args.TEXT_THRESHOLD
HIGH = args.HIGH
DISPLAY = args.DISPLAY

VIDEO_PATH = os.path.join(HOME, "videos", args.VIDEO_NAME)
if not os.path.isfile(VIDEO_PATH):
    parser.error('--VIDEO_NAME does not exist!')

## Load the SAM model
SAM_WEIGHTS_PATH = os.path.join(HOME, 'weights', 'sam_vit_h_4b8939.pth')
model_type = 'vit_h'
sam_model = sam_model_registry[model_type](checkpoint=SAM_WEIGHTS_PATH).to(device=device)
sam_predictor = SamPredictor(sam_model)

## Load the GroundingDINO model
DINO_CONFIG_PATH = os.path.join(HOME, "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
DINO_WEIGHTS_PATH = os.path.join(HOME, "weights", "groundingdino_swint_ogc.pth")
dino_model = load_model(DINO_CONFIG_PATH, DINO_WEIGHTS_PATH)

## Convert video to frame is necessary
video_name = args.VIDEO_NAME.replace('.mp4','')
FRAMES_OUTPUT_PATH = os.path.join(HOME, "videos", video_name+'_frames')
MASKS_OUTPUT_PATH = os.path.join(HOME, "videos", video_name+'_masks')
if not os.path.exists(FRAMES_OUTPUT_PATH):
    os.mkdir(FRAMES_OUTPUT_PATH)
    print(f'Converting {video_name} into frames..')
    n_frames = video_to_frame(VIDEO_PATH, FRAMES_OUTPUT_PATH)
    print(f'Successfully converted {video_name} into {n_frames} frames')
else:
    print('video frames already present.')

## Create Video Masks
if not (os.path.isdir(MASKS_OUTPUT_PATH)):
    os.mkdir(MASKS_OUTPUT_PATH)
    plot = False
    sorted_dir = sorted(os.listdir(FRAMES_OUTPUT_PATH))
    for filename in sorted_dir:
        if filename.endswith('.jpg'):
            #print(f'processing {filename}...')

            # Save path name
            PATH_NAME = os.path.join(FRAMES_OUTPUT_PATH, filename)

            image_source, image = load_image(PATH_NAME)

            boxes, logits, phrases = predict(
                model=dino_model,
                image=image,
                caption=PROMPT,
                box_threshold=BOX_THRESHOLD,
                text_threshold=TEXT_THRESHOLD,
                device=device
            )

            if plot:
                annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
                sv.plot_image(annotated_frame, (16, 16))

            ## initialize the sam predictor
            image_bgr = cv2.imread(PATH_NAME)
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            sam_predictor.set_image(image_rgb)

            h, w, _ = image_source.shape
            boxes = boxes * torch.Tensor([w, h, w, h])
            sam_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
            mask_bgr = np.zeros((h, w, 3), dtype=np.uint8)

            for bbox in sam_boxes:
                pred_bbox = np.array(bbox)
                # Get the segmentation mask
                masks, _, _ = sam_predictor.predict(box=pred_bbox, multimask_output=False)
                mask = masks[0]

                ## Compute the BGR mask
                gray = mask.astype(np.uint8) * 255
                mask_bgr[:, :, 0] += gray

            ## Save the mask
            mask_filename = os.path.join(MASKS_OUTPUT_PATH, filename.replace('.jpg', '.png'))
            cv2.imwrite(mask_filename, mask_bgr)
            #print(f'file {mask_filename} saved in {directory}_mask')
else:
    mask_dir = video_name+'_masks'
    print(f'The directory {mask_dir} already exists!')

## ProPainter Inference
PROPAINTER_EXE = os.path.join(HOME, 'ProPainter', 'inference_propainter.py')
if HIGH:
    subprocess.run(["python", PROPAINTER_EXE, "--video", FRAMES_OUTPUT_PATH, "--mask", MASKS_OUTPUT_PATH, "--height", str(320), "--width", str(576), "--fp16"])
else:
    subprocess.run(["python", PROPAINTER_EXE, "--video", FRAMES_OUTPUT_PATH, "--mask", MASKS_OUTPUT_PATH])

## Define the video result path
PROPAINTER_PATH_IN = os.path.join(HOME, 'ProPainter', 'results', video_name, 'masked_in.mp4')
PROPAINTER_PATH_OUT = os.path.join(HOME, 'ProPainter', 'results', video_name, 'inpaint_out.mp4')
RESULT_PATH = os.path.join(HOME, 'results')
RESULT_PATH_IN = os.path.join(RESULT_PATH, video_name+'_in.mp4')
RESULT_PATH_OUT = os.path.join(RESULT_PATH, video_name+'_out.mp4')
if not os.path.isdir(RESULT_PATH):
    os.mkdir(RESULT_PATH)
os.rename(src=PROPAINTER_PATH_IN, dst=RESULT_PATH_IN)
os.rename(src=PROPAINTER_PATH_OUT, dst=RESULT_PATH_OUT)

## Display the videos
if DISPLAY:
    display_video(RESULT_PATH_IN, RESULT_PATH_OUT)