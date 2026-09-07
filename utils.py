import numpy as np
from torch.utils.data import DataLoader
from PIL import Image
from pathlib import Path
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm

import torch.multiprocessing as multiprocessing
from sklearn.metrics import accuracy_score

def epoch_summary(reports, epoch, tags, label_lookup):
    print(f'Summary:')
    for i, report in enumerate(reports):
        tag = tags[i]

        print(f'\t- {tag}:')
        for label in report.keys():
            if report[label] == None:
                continue
            else:
                if label != 'accuracy':
                    print(f'\t\t-- {label_lookup[label]}: {report[label]:.4}')
                else:
                    print()
                    print(f'\t\t-- {label_lookup[label]}: {report[label]:.4}')
        print()

    return

def fill_lt_reports(lt_reports, reports, phase):

    for lt_report, report in zip(lt_reports, reports):
            keys = list(report.keys())
            for key in keys:
                if key not in lt_report[phase].keys():
                    lt_report[phase][key] = [report[key]]
                else:
                    lt_report[phase][key].append(report[key])
                    
    return


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