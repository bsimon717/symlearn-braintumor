import argparse
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from model import SimpleCNN, Readout
from sklearn.metrics import accuracy_score
import json
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import torch.multiprocessing as multiprocessing
from copy import deepcopy
import numpy as np

from utils import *

def main():
    global device
    global tags
    
    device = torch.device("cuda")
    
    parser = argparse.ArgumentParser(prog='importance',
                    description='importance.py',
                    epilog='Full description TBD.')

    parser.add_argument('--load_path', default=False, type=str, help='Path to folder containing pre-trained models.')
    
    args = parser.parse_args()
    load_path = args.load_path
    
    print(f'Loading settings from path {load_path}')
    with open(f'{load_path}/config.json', 'r') as file:
       config = json.load(file)

    batch_size = config['training']['batch_size']
    temp = config['training']['temp']
    lamb = config['training']['lamb']
    crit = config['training']['criterion']
    
    readout_hidden_dim = config['models']['Readout']['readout_hidden_dim']
    num_heads = config['models']['Readout']['num_heads']
        
    num_preR = config['models']['pre-Readout']['num_preR']
    collab_params = config['models']['pre-Readout']['collab_params']
    kernel_size = config['models']['pre-Readout']['kernel_size']
    kernel_stride = config['models']['pre-Readout']['kernel_stride']
    conv_channels = config['models']['pre-Readout']['conv_channels']
    out_channels = config['models']['pre-Readout']['out_channels']
    num_fc = config['models']['pre-Readout']['num_fc']

    tags = [f'Model_{i}' for i in range(num_preR)] + ['Readout']
    
    torch.cuda.empty_cache()

    if crit == 'CE':
        criterion = nn.CrossEntropyLoss()
    else:
        raise Exception(f'Unsupported Criterion: {crit}')

    _, val_loader, test_loader = get_data_loaders(batch_size)

    models = []

    for _ in range(num_preR):
        model_i = SimpleCNN(kernel_size=kernel_size, kernel_stride=kernel_stride, conv_channels=conv_channels, out_channels=out_channels, padding=1, num_fc=num_fc, num_preR=num_preR)
        models.append(model_i)

    print('Loading pre-trained models.')
    for i in range(num_preR):
        check = torch.load(f'{load_path}/Model_{i}.pt', map_location=torch.device('cpu'))
        models[i].load_state_dict(check['model'])

    readout = Readout(hidden_dim=readout_hidden_dim, num_heads=num_heads, num_preR=num_preR, preR_dim=out_channels)
    models.append(readout)
    readout_check = torch.load(f'{load_path}/Readout.pt', map_location=torch.device('cpu'))
    models[-1].load_state_dict(readout_check['model'])

    importance(models, collab_params, val_loader, criterion, temp, lamb)

def importance(models, collab_params, eval_loader, criterion, temp, lamb):
    frac_changes = []
    for i in range(len(models[:-1])):
        models_clone = [deepcopy(model).to(device) for model in models]

        init_valid_reports = eval_one_epoch(eval_loader, models_clone, collab_params, temp, 999, criterion, lamb=lamb, phase='Validation')
        init_readout_acc = init_valid_reports[-1]['accuracy']

        for layer in models_clone[i].children():
           if hasattr(layer, 'reset_parameters'):
               layer.reset_parameters()

        post_valid_reports = eval_one_epoch(eval_loader, models_clone, collab_params, temp, 999, criterion, lamb=lamb, phase='Validation')
        post_readout_acc = post_valid_reports[-1]['accuracy']

        frac_change = abs((post_readout_acc-init_readout_acc)/init_readout_acc)
        frac_changes.append(frac_change)
        
        torch.cuda.empty_cache()  

    rankings = list(np.argsort(frac_changes))
    rankings.reverse()

    print('Pre-Readout Model Importance:')
    for rank, i in enumerate(rankings):
        print(f'\t{rank+1}) Model_{i}: {frac_changes[i]:.4f}')

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    main()