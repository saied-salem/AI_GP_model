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

##################################################################################
                                #  Multi class
##################################################################################

path = '/content/drive/MyDrive/GP_AI/ready_data'
original_data_path = '/content/drive/MyDrive/GP_AI/Dataset_BUSI_with_GT/'
classes = ['malignant','benign', 'normal']
all_images = []
all_masks = []
for cls in classes:
    for file in sorted(os.listdir(original_data_path + cls)):
        if fnmatch.fnmatch(file, '*mask*.png'):
            all_masks.append(original_data_path+ cls+'/' + file )
        else :
            all_images.append(original_data_path+ cls+'/'+ file)

# extra Data
extra_path = '/content/drive/MyDrive/GP_AI/ExtraMultiClass/'
# images_and_masks = ['Images','Masks']
extra_images = []
extra_masks = []
for cls in classes:
  # print(images_and_masks)
  curr_path = extra_path + cls
  for file in sorted(os.listdir(curr_path)):
    # images_and_masks = os.listdir(extra_path + cls)
    if file =='Images': 
      for img in sorted(os.listdir(curr_path+'/'+file)):  
        extra_images.append(curr_path+'/'+file+'/' + img )
    else:
      for img in sorted(os.listdir(curr_path+'/'+file)):  
        extra_masks.append(curr_path+'/' + file+'/'+img)

all_images.extend(extra_images)
all_masks.extend(extra_masks)
all_images= sorted(all_images)
all_masks= sorted(all_masks)

all_images_train, all_images_val, all_masks_train, all_masks_val = train_test_split(all_images, 
                                            all_masks, test_size=0.15, random_state=42)
print(len(all_images))
print(len(all_masks))
             

if __name__ == '__main__':
    train_model (
        wb_project_name = configerations['wb_project_name'],
        wb_run_name = configerations['wb_run_name'],
        model = model,
        device = device,
        load_weights = configerations['load_weights'],
        weights_dir = configerations['saving_weights_dir'],
        train_images_list = all_images_train,
        train_targets_list = all_masks_train,
        val_images_list = all_images_val,
        val_targets_list = all_masks_val ,
        input_channels = 3,
        output_classes = num_class,
        epochs = configerations['epochs'],
        batch_size = configerations['batch_size'],
        learning_rate = configerations['learning_rate'],
        val_percent = configerations['val_percent'],
        img_scale = configerations['img_scale']
        )
