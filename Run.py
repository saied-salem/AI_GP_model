from train_without_loader import train_model
from Models.basic_unet import UNet
import torch
from pathlib import Path
import fnmatch
import os
import json 

model = UNet(n_channels=3, n_classes=1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


configerations= None
with open('model_config.json') as json_file:
    configerations = json.load(json_file)
 
    # Print the type of data variable
    print("Type:", type(configerations))
    print(configerations)



def sort_key_mask(s):
    # Extract the number following the "mask" prefix
    num_str = s[4:-4]
    # Convert the number string to an integer
    return int(num_str)
def sort_key_image(s):
    # Extract the number following the "mask" prefix
    num_str = s[5:-4]
    # Convert the number string to an integer
    return int(num_str)

def adjust_path(img_list, path):
    for i, img in enumerate(img_list):
      img_list[i] = path + img

path = '/content/drive/MyDrive/GP_AI/ready_data'
train_image =sorted(os.listdir( "/content/drive/MyDrive/GP_AI/ready_data/train/train_images"), key = sort_key_image)
train_mask =sorted(os.listdir( "/content/drive/MyDrive/GP_AI/ready_data/train/train_MasksNorm3"), key = sort_key_mask)
val_image = sorted(os.listdir("/content/drive/MyDrive/GP_AI/ready_data/val/val_images"), key = sort_key_image)
val_mask =sorted(os.listdir( "/content/drive/MyDrive/GP_AI/ready_data/val/val_MasksNorm3"), key = sort_key_mask)

adjust_path(train_image, "/content/drive/MyDrive/GP_AI/ready_data/train/train_images/")
adjust_path(train_mask , "/content/drive/MyDrive/GP_AI/ready_data/train/train_MasksNorm3/")
adjust_path(val_image, "/content/drive/MyDrive/GP_AI/ready_data/val/val_images/")
adjust_path(val_mask, "/content/drive/MyDrive/GP_AI/ready_data/val/val_MasksNorm3/")


if __name__ == '__main__':
    train_model (
        wb_project_name = configerations['wb_project_name'],
        wb_run_name = configerations['wb_run_name'],
        model = model,
        device = device,
        weights_dir = configerations['saving_weights_dir'],
        train_images_list = train_image,
        train_targets_list = train_mask,
        val_images_list = val_image,
        val_targets_list = val_mask ,
        input_channels = 3,
        output_classes = 1,
        epochs = configerations['epochs'],
        batch_size = configerations['batch_size'],
        learning_rate = configerations['learning_rate'],
        val_percent = configerations['val_percent'],
        img_scale = configerations['img_scale']
        )
