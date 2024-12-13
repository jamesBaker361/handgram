import os
import argparse
from experiment_helpers.gpu_details import print_details
from experiment_helpers.metadata_unet import MetaDataUnet
from accelerate import Accelerator
import time
from datasets import load_dataset
from diffusers import AutoencoderKL
from camera_locations import locations
from PIL import Image
import numpy as np

parser=argparse.ArgumentParser()

parser.add_argument("--mixed_precision",type=str,default="no")
parser.add_argument("--project_name",type=str,default="hands")
#parser.add_argument("--n_cameras",type=int,default=4)
parser.add_argument("--hub_dataset",type=str,default="jlbaker361/hands")
parser.add_argument("--image_size",type=int,default=256)
parser.add_argument("--task",type=str,default="image",help="one of image or video or ???")
parser.add_argument("--use_metadata",action="store_true")
parser.add_argument("--use_vae",action="store_true")
parser.add_argument("--encoder_mismatch",action="store_true",help="if using autoencoders, to mismatch them")
parser.add_argument("--n_unet",type=int,default=1,help="how many unets sequntially")
parser.add_argument("--denoise_steps",type=int,default="number of steps to use if denoising")
parser.add_argument("--pretrained_unet_path",type=str,default="",help="if not blank start with a pretrained unet checkpoint")
parser.add_argument("--pretrained_vae_path",type=str,default="",help="if non empty, use this pretrained vae path")
parser.add_argument("--shitty_dataset",type=str,default="",help="if not blank, use this for shitty AI versions of all the existing hands and use them as inputs")
#given hand, and same hand in different pose, 


def resize_images(example, image_columns, size=(128, 128)):
    # Iterate through all specified image columns
    for col in image_columns:
        if col in example and isinstance(example[col], np.ndarray):
            # Convert NumPy array to PIL Image
            img = Image.fromarray(example[col])
            # Resize the image
            img_resized = img.resize(size)
            # Convert back to NumPy array and store
            example[col] = np.array(img_resized)
    return example

def get_image_columns(data):
    return [col for col, dtype in data['train'].features.items() if dtype.__class__.__name__ == "Image"]

def main(args):
    accelerator=Accelerator(log_with="wandb",mixed_precision=args.mixed_precision)
    accelerator.init_trackers(project_name=args.project_name,config=vars(args))
    _data=load_dataset(args.hub_dataset)
    image_columns=get_image_columns(_data)
    _data=_data.map(lambda x: resize_images(x,image_columns,(args.image_size,args.image_size)),batched=False)
    train_data=_data["train"]
    test_data=_data["test"]
    n_cameras=len(train_data.features)-1 #-1 for time index
    use_shitty=False
    if len(args.shitty_dataset)>0:
        use_shitty=True
        _shit_data=load_dataset(args.shitty_dataset)
        _shit_data=_shit_data.map(lambda x: resize_images(x,image_columns,(args.image_size,args.image_size)),batched=False)
        train_shit_data=_shit_data["train"]
        test_shit_data=_shit_data["test"]


    #we have one main unet NOT conditioned on anything (no metadata), then we have VAE encoder/decoders for each one
    

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