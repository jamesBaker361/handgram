import os
import argparse
from experiment_helpers.gpu_details import print_details
from accelerate import Accelerator
from datasets import load_dataset
from PIL import Image
import mediapipe as mp
import time
import random
import numpy as np
import torch
import cv2
from torchvision import transforms



parser=argparse.ArgumentParser()

parser.add_argument("--mixed_precision",type=str,default="no")
parser.add_argument("--project_name",type=str,default="evaluation-creative")
parser.add_argument("--start",default=0,type=int)
parser.add_argument("--end",type=int,default=2)
parser.add_argument("--dataset",type=str, default="jlbaker361/processed-james")
parser.add_argument("--distortion",type=str,default="blur",help="controlnet or blur")
parser.add_argument("--epochs",type=int,default=2)
parser.add_argument("--gradient_accumulation_steps",type=int,default=2)
parser.add_argument("--batch_size",type=int,default=1)
parser.add_argument("--image_size",type=int,default=512)


#use controlnet + unet lora + meta embedding vs single perspective controlnet
#or we do image-to-image SDEDIT conditioning on wrong camera version- not sure what baseline would be
#sdedit conditioned on bones and metadata THIS vs no metadata and single perspective bones
#when we get multiple hands we can do OG hand + new handpose and we gotta do that
#finger fix- we only fuck up a few fingers and then condition on the fucked up fingers
# keypoint fix- we have a metadata embedding based on the keypoints- i.e. we have [0,1] for each keypoint or finger and embed that into the shape of the emb


def main(args):
    accelerator=Accelerator(log_with="wandb",mixed_precision=args.mixed_precision,gradient_accumulation_steps=args.gradient_accumulation_steps)
    accelerator.init_trackers(project_name=args.project_name,config=vars(args))

    preprocess = transforms.Compose(
    [
        transforms.Resize((args.image_size, args.image_size)),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
        
    ]
    )

    dataset=load_dataset(args.dataset,split="train")

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)
    #get list of names
    #alpahbaetixe
    #take the start:limit
    subject_name_set=set(dataset["subject_name"])
    subject_name_list=sorted(list(subject_name_set))[args.start:args.end]
    print("subjects are ",subject_name_list)

    finger_list=["thumb","index","middle","ring","pinky"]

    for subject in subject_name_list:
        split_dataset = dataset.filter(lambda row: row["subject_name"] in [subject]).remove_columns("subject_name").train_test_split(test_size=0.2, seed=42)

        # Access the train and test splits
        train_dataset = split_dataset["train"]
        test_dataset = split_dataset["test"]

        base_per_image=3
        def get_base_dataset(dataset): 
            #distort via blurring
            #distort via finger specific blurring/ruining?
            base_dataset={key:[] for key in [f"camera_{i}" for i in range(4)]+["fingers"]}
            for row in dataset:
                if args.distortion=="blur":
                    for b in range(base_per_image):
                        new_row=[]
                        random_fingers=random.sample(finger_list,random.randint(1,5))
                        base_dataset["fingers"].append([1 if finger in random_fingers else 0 for finger in finger_list])
                        for i in range(4):
                            pil_image=row[f"camera_{i}"]
                            # Convert PIL Image to NumPy Array
                            opencv_image = np.array(pil_image)
                            output_image = opencv_image.copy()

                            # Convert RGB to BGR (OpenCV uses BGR format by default)
                            opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_RGB2BGR)
                            h, w, _ = opencv_image.shape
                            results = hands.process(opencv_image)

                            if results.multi_hand_landmarks:
                                for hand_landmarks in results.multi_hand_landmarks:

                                    finger_regions = {
                                        "thumb": [
                                            (mp.solutions.hands.HandLandmark.THUMB_CMC, mp.solutions.hands.HandLandmark.THUMB_MCP),
                                            (mp.solutions.hands.HandLandmark.THUMB_MCP, mp.solutions.hands.HandLandmark.THUMB_IP),
                                            (mp.solutions.hands.HandLandmark.THUMB_IP, mp.solutions.hands.HandLandmark.THUMB_TIP),
                                        ],
                                        "index": [
                                            (mp.solutions.hands.HandLandmark.INDEX_FINGER_MCP, mp.solutions.hands.HandLandmark.INDEX_FINGER_PIP),
                                            (mp.solutions.hands.HandLandmark.INDEX_FINGER_PIP, mp.solutions.hands.HandLandmark.INDEX_FINGER_DIP),
                                            (mp.solutions.hands.HandLandmark.INDEX_FINGER_DIP, mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP),
                                        ],
                                        "middle": [
                                            (mp.solutions.hands.HandLandmark.MIDDLE_FINGER_MCP, mp.solutions.hands.HandLandmark.MIDDLE_FINGER_PIP),
                                            (mp.solutions.hands.HandLandmark.MIDDLE_FINGER_PIP, mp.solutions.hands.HandLandmark.MIDDLE_FINGER_DIP),
                                            (mp.solutions.hands.HandLandmark.MIDDLE_FINGER_DIP, mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP),
                                        ],
                                        "ring": [
                                            (mp.solutions.hands.HandLandmark.RING_FINGER_MCP, mp.solutions.hands.HandLandmark.RING_FINGER_PIP),
                                            (mp.solutions.hands.HandLandmark.RING_FINGER_PIP, mp.solutions.hands.HandLandmark.RING_FINGER_DIP),
                                            (mp.solutions.hands.HandLandmark.RING_FINGER_DIP, mp.solutions.hands.HandLandmark.RING_FINGER_TIP),
                                        ],
                                        "pinky": [
                                            (mp.solutions.hands.HandLandmark.PINKY_MCP, mp.solutions.hands.HandLandmark.PINKY_PIP),
                                            (mp.solutions.hands.HandLandmark.PINKY_PIP, mp.solutions.hands.HandLandmark.PINKY_DIP),
                                            (mp.solutions.hands.HandLandmark.PINKY_DIP, mp.solutions.hands.HandLandmark.PINKY_TIP),
                                        ],
                                    }

                                    random_regions=[finger_regions[finger] for finger in random_fingers]
                                    for region in random_regions:
                                        print(region)
                                        for subregion in region:
                                            start_idx, end_idx = subregion
                                            start = hand_landmarks.landmark[start_idx]
                                            end = hand_landmarks.landmark[end_idx]

                                            # Convert normalized coordinates to pixel coordinates
                                            start_point = (int(start.x * w), int(start.y * h))
                                            end_point = (int(end.x * w), int(end.y * h))

                                            # Define bounding box for the region
                                            x_min = min(start_point[0], end_point[0])
                                            x_max = max(start_point[0], end_point[0])
                                            y_min = min(start_point[1], end_point[1])
                                            y_max = max(start_point[1], end_point[1])

                                            # Calculate width and height
                                            width = x_max - x_min
                                            height = y_max - y_min

                                            n_sub=1
                                            # Divide the region into 3 smaller rectangles
                                            for i in range(n_sub):
                                                # Calculate the top-left corner of each smaller rectangle
                                                sub_x_min = x_min + (width // n_sub) * i
                                                sub_x_max = x_min + (width // n_sub) * (i + 1)
                                                sub_y_min = y_min + (height // n_sub) * i
                                                sub_y_max = y_min + (height // n_sub) * (i + 1)

                                                # Extract the sub-region (small rectangle)
                                                sub_roi = output_image[sub_y_min:sub_y_max, sub_x_min:sub_x_max]

                                                k=100

                                                # Apply Gaussian blur to the smaller rectangle
                                                if sub_roi.size > 0:
                                                    blurred_sub_roi = cv2.GaussianBlur(sub_roi, (k, k), 50)  # More aggressive blur

                                                    # Replace the processed sub-region back into the image
                                                    output_image[sub_y_min:sub_y_max, sub_x_min:sub_x_max] = blurred_sub_roi
                            new_pil_image=Image.fromarray(output_image)
                            base_dataset[f"camera_{i}"].append(new_pil_image)
            return base_dataset
        train_base_dataset=get_base_dataset(train_dataset)
        test_base_dataset=get_base_dataset(test_dataset)

        def get_batched_dataset(dataset):
            try:
                keys=[k for k in dataset.keys()]
            except:
                keys=dataset.column_names
            batched_dataset={key:[] for key in keys}
            for key in dataset:
                column=dataset[key]
                if type(column[0])==Image.Image:
                    column=[preprocess(image) for image in column]
                    
                else:
                    column=[torch.tensor(digits) for digits in column ]
                batched_dataset[key]=[torch.stack(column[i:i+args.batch_size] for i in range(0,len(column),args.batch_size))]
            return batched_dataset


    return

if __name__=='__main__':
    print_details()
    start=time.time()
    args=parser.parse_args()
    print(args)
    main(args)
    end=time.time()
    seconds=end-start
    hours=seconds/(60*60)
    print(f"successful generating:) time elapsed: {seconds} seconds = {hours} hours")
    print("all done!")