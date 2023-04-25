from train import train_model
from Models.basic_unet import UNet
import torch
from pathlib import Path
import fnmatch
import os
import json 

model = UNet(n_channels=3, n_classes=3)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


configerations= None
with open('model_config.json') as json_file:
    configerations = json.load(json_file)
 
    # Print the type of data variable
    print("Type:", type(configerations))
    print(configerations)

path = '/content/drive/MyDrive/GP_AI/Dataset_BUSI_with_GT/'
classes = ['malignant','benign', 'normal']
all_images = []
all_masks = []
for cls in classes:

    for file in sorted(os.listdir(path + cls)):
        if fnmatch.fnmatch(file, '*mask*.png'):
            all_masks.append(path+ cls+'/' + file )
        else :
            all_images.append(path+ cls+'/'+ file)
print(len(all_images))
print(len(all_masks))
if __name__ == '__main__':
    train_model (
        model = model,
        device = device,
        weights_dir = configerations['saving_weights_dir'],
        images_list = all_images,
        targets_list = all_masks,
        input_channels = 3,
        output_classes = 3,
        epochs = configerations['epochs'],
        batch_size = configerations['batch_size'],
        learning_rate = configerations['learning_rate'],
        val_percent = configerations['val_percent'],
        img_scale = configerations['img_scale']
        )
