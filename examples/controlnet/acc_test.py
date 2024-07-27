import torch
# from accelerate import Accelerator
from datasets import load_dataset
import os

def main():
    # accelerator = Accelerator()
    # print(torch.cuda.is_available())
    dataset = load_dataset("csv", data_files="/home/mayanze/PycharmProjects/diffusers/examples/controlnet/one_plane_label/one_plane_dataset.csv")
    print(dataset)
if __name__ == "__main__":
    main()
