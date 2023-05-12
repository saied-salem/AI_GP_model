from train import train_model
from Models.basic_unet import UNet
import torch
from pathlib import Path
import fnmatch
import os
import json 
from sklearn.model_selection import train_test_split



configerations= None
with open('model_config.json') as json_file:
    configerations = json.load(json_file)
 
    # Print the type of data variable
    print("Type:", type(configerations))
    print(configerations)

num_class = 1
if configerations['is_multi_class'] == True:
   num_class=3
else:
   num_class=1
   
model = UNet(n_channels=3, n_classes=num_class)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
training_data =None
val_data =None

with open('/content/drive/MyDrive/GP_AI/New_data/training.json') as f:
    training_data = json.load(f)  
with open('/content/drive/MyDrive/GP_AI/New_data/val.json') as f:
    val_data = json.load(f)

training_images, training_masks = training_data['images'], training_data["masks"]
val_images, val_masks = val_data['images'], val_data["masks"]

##################################################################################
                                #  Multi class
##################################################################################

print('training ',len(training_images))
print("val ",len(val_images))
             

if __name__ == '__main__':
    train_model (
        wb_project_name = configerations['wb_project_name'],
        wb_run_name = configerations['wb_run_name'],
        model = model,
        device = device,
        load_weights = configerations['load_weights'],
        weights_dir = configerations['saving_weights_dir'],
        train_images_list = training_images[:10],
        train_targets_list = training_masks[:10],
        val_images_list = val_images[:10],
        val_targets_list = val_masks[:10],
        input_channels = 3,
        output_classes = num_class,
        epochs = configerations['epochs'],
        batch_size = configerations['batch_size'],
        learning_rate = configerations['learning_rate'],
        val_percent = configerations['val_percent'],
        img_scale = configerations['img_scale']
        )
