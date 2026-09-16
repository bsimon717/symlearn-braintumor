import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL import Image
from pathlib import Path
from torchvision import transforms
from torchvision.datasets import ImageFolder

def get_data_loaders(batch_size):

    BASE_DIR = Path('data/brain-tumor-mri-deduplicated/data')
    TRAIN_DIR = BASE_DIR / 'train'
    VAL_DIR   = BASE_DIR / 'val'
    TEST_DIR  = BASE_DIR / 'test'

    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.functional.rgb_to_grayscale,
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.functional.rgb_to_grayscale,
    ])
    
    train_ds = ImageFolder(str(TRAIN_DIR), transform=train_tf)
    val_ds   = ImageFolder(str(VAL_DIR),   transform=eval_tf)
    test_ds  = ImageFolder(str(TEST_DIR),  transform=eval_tf)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=2)

    print(f'Target Labels: {train_ds.classes}')
    print()
    print('Number of Batches:')
    print(f'\ttrain={len(train_ds)} val={len(val_ds)} test={len(test_ds)}')
    

    return train_loader, val_loader, test_loader