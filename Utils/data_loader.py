import logging
import numpy as np
import torch
from PIL import Image
from functools import lru_cache
from functools import partial
from itertools import repeat
from multiprocessing import Pool
from os import listdir
from os.path import splitext, isfile, join
from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm
from torch.nn.functional import one_hot
import fnmatch
import matplotlib.pyplot as plt

def load_image(filename):
    ext = splitext(filename)[1]
    if ext == '.npy':
        return Image.fromarray(np.load(filename))
    elif ext in ['.pt', '.pth']:
        return Image.fromarray(torch.load(filename).numpy())
    else:
        return Image.open(filename)


def unique_mask_values(idx, mask_dir, mask_suffix):
    mask_file = list(mask_dir.glob(idx + mask_suffix + '.*'))[0]
    mask = np.asarray(load_image(mask_file))
    if mask.ndim == 2:
        return np.unique(mask)
    elif mask.ndim == 3:
        mask = mask.reshape(-1, mask.shape[-1])
        return np.unique(mask, axis=0)
    else:
        raise ValueError(f'Loaded masks should have 2 or 3 dimensions, found {mask.ndim}')


class BasicDataset(Dataset):
    def __init__(self, images_dir: str, mask_dir: str, scale: float = 1.0):
        self.images_dir = images_dir
        self.mask_dir = mask_dir
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale


    def __len__(self):
        return len(self.images_dir)

    @staticmethod
    def preprocess( pil_img, file_name, scale, is_mask):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        # print(np.unique(img))

        img = np.asarray(pil_img).astype('int64')
        # print("neeeew",np.unique(img))

        if is_mask:
            #For  segmentation and classes Clasifications
            if(img.ndim > 2):
                # print("HHHHHHHHHHHHHHHHHHHHHHH")

                img= img[:,:,0]
                # print("mask shape: " ,img.shape )

            if 'malignant' in file_name:
                # print("HHHHHHHHHHHHHHHHHHHHHHH")
                img[img == 1] = 2
                # print(np.unique(img))
            elif "benign" in file_name:
                img[img == 1] = 1
            elif 'normal' in file_name:
                img= np.zeros((img.shape[0],img.shape[1]))

            num_classes = 3
            # print(img)
            img = torch.tensor(img).to(torch.int64)
            # img = one_hot(img, num_classes)           
            # img = img.permute((2, 0, 1))
            #For segemtation background and for ground only  
            # if 'malignant' in file_name:
            #     img[img == 1] = 1
            # elif "benign" in file_name:
            #     img[img == 1] = 1
            # elif 'normal' in file_name:
            #     img= np.zeros((img.shape[0],img.shape[1]))

            return img

        else:
            img= img/255
            if img.ndim == 2:
                img = img[np.newaxis, ...]
            else:
                img = img[:,:,:3].transpose((2, 0, 1))


            return torch.tensor(img)

    def __getitem__(self, idx):

        img_file = self.images_dir[idx]
        mask_file = self.mask_dir[idx]
        # print(img_file)
        # assert len(img_file) == 1, f'Either no image or multiple images found for the ID {img_file}: {img_file}'
        # assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {img_file}: {mask_file}'
        mask = load_image(mask_file)
        img = load_image(img_file)
        # print("load_image: ", np.unique(img))

        assert img.size == mask.size, \
            f'Image and mask {img_file} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess( img,img_file, self.scale, is_mask=False)
        mask = self.preprocess( mask,mask_file, self.scale, is_mask=True  )

        return {
            'image': img,
            'mask': mask 
        }
        # return img , mask


if __name__ == "__main__":
    path = './Dataset_BUSI_with_GT/malignant/'
    images_and_maskes_list  =  listdir(path)
    images_list , maskes_list = [],[]
    for file in images_and_maskes_list:
        if fnmatch.fnmatch(file, '*mask.png'):
            maskes_list.append(path+ file)
        else :
            images_list.append(path + file )
    # print('masks_list :',maskes_list)
    # print("-----------------")
    # print("images_list", images_list)
    data_loader = BasicDataset(images_list,maskes_list,1)
    image , mask =next(iter(data_loader))
    # image , mask =next(iter(data_loader))
    # image , mask =next(iter(data_loader))

    print(image.ndim)
    print("\n")
    print(mask.ndim)
    # plt.imshow(mask[0])
    # image= image.numpy()
    image =torch.moveaxis(image,0,-1)
    mask =torch.moveaxis(mask,0,-1)

    print("hhhhhhhhhh",image.shape)
    print("hhhhhhhhhh",mask.shape)

    # print("hhhhhhhhhh",np.unique(image))
    print(len(images_list))
    print(len(maskes_list))
    print(torch.unique(image))


    plt.imshow(image)
    plt.show()


    for img , mask  in zip(sorted(images_list),sorted(maskes_list)):
        # print("image file: " ,img , "      mask file: " , mask)
        i = load_image(img)
        m = load_image(mask)
        # print("image dim: " ,np.array(i).ndim, "      mask dim: " , np.array(m).ndim)


