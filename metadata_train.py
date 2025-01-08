import os
import argparse
from experiment_helpers.gpu_details import print_details
from accelerate import Accelerator
from datasets import load_dataset
import time

parser=argparse.ArgumentParser()

parser.add_argument("--mixed_precision",type=str,default="no")
parser.add_argument("--project_name",type=str,default="evaluation-creative")
parser.add_argument("--start",default=0,type=int)
parser.add_argument("--end",type=int,default=2)
parser.add_argument("--dataset",type=str, default="jlbaker361/processed-james")
parser.add_argument("--blur",action="store_true")



def main(args):
    accelerator=Accelerator(log_with="wandb",mixed_precision=args.mixed_precision)
    accelerator.init_trackers(project_name=args.project_name,config=vars(args))

    dataset=load_dataset(args.dataset,split="train")
    #get list of names
    #alpahbaetixe
    #take the start:limit
    subject_name_set=set(dataset["subject_name"])
    subject_name_list=sorted(list(subject_name_set))[args.start:args.end]
    print("subjects are ",subject_name_list)
    split_dataset = dataset["train"].filter(lambda row: row["subject_name"] in subject_name_list).train_test_split(test_size=0.2, seed=42)

    # Access the train and test splits
    train_dataset = split_dataset["train"]
    test_dataset = split_dataset["test"]

    #filter out 
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