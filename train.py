import argparse
import logging
import os
import random
import sys
import fnmatch

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

import wandb
from Metrics.evaluate import evaluate
from Models.basic_unet import UNet
from Utils.data_loader import BasicDataset
from Metrics.dice_score import dice_loss
 


dir_checkpoint = Path('./checkpoints/')


def train_model(
        wb_project_name,
        wb_run_name,
        model,
        device,
        weights_dir,
        train_images_list,
        train_targets_list,
        val_images_list,
        val_targets_list,
        input_channels,
        output_classes,
        load_weights =False,
        epochs: int = 5,
        batch_size: int = 32,
        learning_rate: float = 1e-5,
        val_percent: float = 0.1,
        save_checkpoint: bool = True,
        img_scale: float = 0.5,
        amp: bool = False,
        weight_decay: float = 1e-8,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0,
        
):
   
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixe
    model = model.to(memory_format=torch.channels_last)

    logging.info(f'Network:\n'
                 f'\t{input_channels} input channels\n'
                 f'\t{output_classes} output channels (classes)\n'
                #  f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling'
                 )

    if load_weights:
        state_dict = torch.load(weights_dir, map_location=device)
        del state_dict['mask_values']
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded from {weights_dir}')


    model.to(device=device)
    # 1. Create dataset
    if output_classes == 1:
        multi_class = False
    else:
        multi_class = True
    # dataset = BasicDataset(images_list, targets_list, img_scale, multi_class)

    # 2. Split into train / validation partitions
    # n_val = int(len(dataset) * val_percent)
    # n_train = len(dataset) - n_val
    # train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))
    train_set = BasicDataset(train_images_list, train_targets_list, img_scale, multi_class)
    val_set = BasicDataset(val_images_list, val_targets_list, img_scale, multi_class)
    
    n_train = len(train_set) 
    n_val = int(len(val_set))
    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    # (Initialize logging)
    experiment = wandb.init(project='U-Net',resume='allow', anonymous='must')
    experiment.config.update(
        dict(epochs=epochs, 
             batch_size=batch_size, 
             learning_rate=learning_rate,
             val_percent=val_percent, 
             save_checkpoint=save_checkpoint, 
             img_scale=img_scale, 
             amp=amp)
    )

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.AdamW(model.parameters(),
                              lr=learning_rate, foreach=True, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max',factor= 0.7, patience=3)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss() if output_classes > 1 else nn.BCEWithLogitsLoss()
    global_step = 0
    best_metric =-1
    log_table = []
    # 5. Begin training
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images, true_masks = batch['image'], batch['mask']

                assert images.shape[1] == input_channels, \
                    f'Network has been defined with {input_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device, dtype=torch.int64)

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    masks_pred = model(images)
                    if output_classes == 1:
                        loss = dice_loss(F.sigmoid(masks_pred.squeeze(1)), true_masks.float(), multiclass=False)
                        # loss += criterion(masks_pred.squeeze(1), true_masks.float())
                    else:
                        # print(true_masks.shape[0,0])
                        loss = dice_loss(
                            F.softmax(masks_pred, dim=1).float(),
                            F.one_hot(true_masks, output_classes).permute(0, 3, 1, 2).float(),
                            multiclass=True
                        )
                        # loss += criterion(masks_pred, true_masks)

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                experiment.log({
                    'train loss': loss.item(),
                    'step': global_step,
                    'epoch': epoch
                })
                pbar.set_postfix(**{'loss (batch)': loss.item()})
                log_table.append({'image':  wandb.Image(images[0].cpu()), 
                                  'predictied_mask': wandb.Image(masks_pred.argmax(dim=1)[0].float().cpu()), 
                                  "true_mask":  wandb.Image(true_masks[0].float().cpu())})
                log_df = pd.DataFrame(log_table)

                # Evaluation round
                # division_step = (n_train // (5 * batch_size))
                # if division_step > 0:
                #     if global_step % division_step == 0:
            histograms = {}
            for tag, value in model.named_parameters():
                tag = tag.replace('/', '.')
                if not (torch.isinf(value) | torch.isnan(value)).any():
                    histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                if not (torch.isinf(value.grad) | torch.isnan(value.grad)).any():
                    histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

            val_score = evaluate(model,output_classes, val_loader, device, amp)
            scheduler.step(val_score)
            logging.info('Validation Dice score: {}'.format(val_score))
            if val_score>best_metric :
                Path(weights_dir).mkdir(parents=True, exist_ok=True)
                state_dict = model.state_dict()
                state_dict['mask_values'] = train_set.mask_values
                saving_path = weights_dir + 'best_checkpoint_dice_val_score.pth'
                torch.save(state_dict, saving_path)
                logging.info(f'Checkpoint {epoch} saved!')
                best_metric = val_score
            try:
                experiment.log({
                    'learning rate': optimizer.param_groups[0]['lr'],
                    'validation Dice': val_score,
                    'image_mask_table': wandb.Table(dataframe=log_df),
                    'step': global_step,
                    'epoch': epoch,
                    **histograms
                })
            except:
                pass



def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    model = UNet(n_channels=3, n_classes=3)
    model = model.to(memory_format=torch.channels_last)

    logging.info(f'Network:\n'
                 f'\t{model.n_channels} input channels\n'
                 f'\t{model.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')

    if args.load:
        state_dict = torch.load(args.load, map_location=device)
        del state_dict['mask_values']
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded from {args.load}')

    model.to(device=device)
    try:
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp
        )
    except torch.cuda.OutOfMemoryError:
        logging.error('Detected OutOfMemoryError! '
                      'Enabling checkpointing to reduce memory usage, but this slows down training. '
                      'Consider enabling AMP (--amp) for fast and memory efficient training')
        torch.cuda.empty_cache()
        model.use_checkpointing()
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp
        )
