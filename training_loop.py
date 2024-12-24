import os
import argparse
from experiment_helpers.gpu_details import print_details
from experiment_helpers.metadata_unet import MetaDataUnet
from accelerate import Accelerator
import time
from datasets import load_dataset
from diffusers import AutoencoderKL
from torchvision import transforms
from camera_locations import locations
from PIL import Image
import numpy as np
import torch

parser=argparse.ArgumentParser()

parser.add_argument("--n_epochs",type=int,default=2)
parser.add_argument("--mixed_precision",type=str,default="no")
parser.add_argument("--project_name",type=str,default="hands")
#parser.add_argument("--n_cameras",type=int,default=4)
parser.add_argument("--hub_dataset",type=str,default="jlbaker361/processed-james")
parser.add_argument("--image_size",type=int,default=256)
parser.add_argument("--task",type=str,default="image",help="one of image or video or ???")
parser.add_argument("--use_metadata",action="store_true")
parser.add_argument("--use_vae",action="store_true")
parser.add_argument("--parallel_vae",action="store_true",help="whether to use vaes for each camera")
parser.add_argument("--encoder_mismatch",action="store_true",help="if using autoencoders, to mismatch them")
#parser.add_argument("--use_unet",action="store_true",help="whether to use unet")
parser.add_argument("--n_unet",type=int,default=1,help="how many unets sequntially")
parser.add_argument("--batch_size",type=int,default=2)
parser.add_argument("--gradient_accumulation_steps",default=4,type=int)
parser.add_argument("--denoise_steps",type=int,default="number of steps to use if denoising")
parser.add_argument("--pretrained_unet_path",type=str,default="",help="if not blank start with a pretrained unet checkpoint")
parser.add_argument("--pretrained_vae_path",type=str,default="",help="if non empty, use this pretrained vae path")
parser.add_argument("--use_shitty",action="store_true")
parser.add_argument("--shitty_dataset",type=str,default="jlbaker361/processed-james",help="use this for shitty AI versions of all the existing hands and use them as inputs")
#given hand, and same hand in different pose, 




def get_image_columns(data):
    return [col for col, dtype in data['train'].features.items() if dtype.__class__.__name__ == "Image"]

def main(args):

    dtype={
        "no":torch.float32,
        "fp16":torch.float16
    }[args.mixed_precision]

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
                example[col]=transforms.ToTensor()(example[col]).to(dtype)
        return example


    accelerator=Accelerator(log_with="wandb",mixed_precision=args.mixed_precision,gradient_accumulation_steps=args.gradient_accumulation_steps)
    accelerator.init_trackers(project_name=args.project_name,config=vars(args))
    _data=load_dataset(args.hub_dataset)
    image_columns=get_image_columns(_data)
    _data=_data.map(lambda x: resize_images(x,image_columns,(args.image_size,args.image_size)),batched=True,batch_size=args.batch_size)
    try:
        train_data=_data["train"]
        test_data=_data["test"]
    except:
        _data=train_data.train_test_split(test_size=0.2, seed=42)
        train_data=_data["train"]
        test_data=_data["test"]
    n_cameras=len(train_data.features)-1 #-1 for time index
    use_shitty=False
    train_shit_data=[]
    test_shit_data=[]
    
    _shit_data=load_dataset(args.shitty_dataset)
    _shit_data=_shit_data.map(lambda x: resize_images(x,image_columns,(args.image_size,args.image_size)),batched=True,batch_size=args.batch_size)
    try:
        train_shit_data=_shit_data["train"]
        test_shit_data=_shit_data["test"]
    except:
        _shit_data=train_shit_data.train_test_split(test_size=0.2, seed=42)
        train_shit_data=_shit_data["train"]
        test_shit_data=_shit_data["test"]


    #build models
    if args.use_vae:
        vae_list = [
            AutoencoderKL.from_pretrained(args.pretrained_vae_path) if args.pretrained_vae_path != "" 
            else AutoencoderKL() 
            for _ in range(n_cameras)
        ]
        

        for vae in vae_list:
            vae.to(accelerator.device)

        if not args.parallel_vae:
            vae=vae_list[0]
    
        vae_optimizer_list=[torch.optim.AdamW(vae.parameters()) for vae in vae_list]

    unet_list=[MetaDataUnet.from_pretrained(args.pretrained_unet_path) if args.pretrained_unet_path != ""
    else MetaDataUnet()
    for _ in range(args.n_unet)]


    for unet in unet_list:
        unet.to(accelerator.device)

    unet_optimizer=torch.optim.AdamW([item for sublist in [unet.parameters() for unet in unet_list] for item in sublist])

    def encode_vae(vae,batch):
        model_input = vae.encode(batch).latent_dist.sample()
        model_input = model_input * vae.config.scaling_factor
        return model_input

    def decode_vae(vae,batch):
        return vae.decode(batch/vae.config.scaling_factor,return_dict=False)

    #training loop
    for e in range(args.n_epochs):
        with accelerator.accumulate():
            for good_batch,shitty_batch in zip(train_data, train_shit_data):
                if args.parallel_vae:
                    latents=[encode_vae(vae,good_batch[f"camera_{k}"]) for k,vae in enumerate(vae_list)]
                    if use_shitty:
                        shitty_latents=[encode_vae(vae,shitty_batch[f"camera_{k}"]) for k,vae in enumerate(vae_list)]
                else:
                    latents=[vae(torch.cat([good_batch[f"camera_{k}"] for k in range(n_cameras)]),dim=0)]
                    if use_shitty:
                        shitty_latents=[vae(torch.cat([shitty_batch[f"camera_{k}"] for k in range(n_cameras)]),dim=0)]
                new_latents=[]
                new_shitty_latents=[]
                for column,shitty_column in zip(latents,shitty_latents):
                    for unet in unet_list:
                        column=unet(column)
                    new_latents.append(column)
                    if args.use_shitty:
                        for unet in unet_list:
                            shitty_column=unet(shitty_column)
                        new_shitty_latents.append(shitty_column)
                if args.parallel_vae:
                    decoded_latents=[decode_vae(vae,batch) for vae,batch in zip(vae_list, new_latents)]
                
                



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