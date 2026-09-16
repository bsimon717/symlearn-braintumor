import argparse
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.nn.utils.prune as prune
from sklearn.metrics import accuracy_score
import json
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import torch.multiprocessing as multiprocessing
from copy import deepcopy

from utils import *
from model import SimpleCNN
from bayesian_pruning import prune_preReadout

from symlearn.loss import *
from symlearn.classify.Readout import Readout
from symlearn.classify.utils import *

def count_nonzero_parameters(model):
    return sum((p != 0).sum().item() for p in model.parameters() if p.requires_grad)

def main():
    device = torch.device("cuda")
        
    parser = argparse.ArgumentParser(prog='test_pruned',
                    description='test_pruned.py',
                    epilog='Full description TBD.')
    
    parser.add_argument('--load_path', default=False, type=str, help='Path to folder containing pre-trained models.')
    parser.add_argument('--trial', type=int, help='Trial number of bayesian optimization.')
    parser.add_argument('--save', type=bool, default=False, help='Save pruned models.')
    
    args = parser.parse_args()
    load_path = args.load_path
    trial = args.trial
    save = args.save
    
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

    label_lookup = {}
    label_lookup['personal'] = 'Average Personal Loss'
    label_lookup['symbiotic'] = 'Average Symbiotic Loss'
    label_lookup['accuracy'] = 'Accuracy'
    label_lookup['embedding'] = 'Average Embedding Loss'
    label_lookup['blame'] = 'Average Blame Loss'
    
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

    num_before_prune = 0
    for model in models:
        num_before_prune += count_nonzero_parameters(model)
    
    models_clone = [deepcopy(model).to(device) for model in models]
    
    study = pd.read_json(f'{load_path}/all_trials.json')
    
    for i in range(num_preR):
        perc = study.iloc[trial][f"params_Model_{i} Unstructured Pruning Percentage"]    
        prune_preReadout(models_clone[i], perc)

    valid_reports = eval_one_epoch(val_loader, models_clone, collab_params, temp, 999, criterion, lamb=lamb, phase='Validation')
    torch.cuda.empty_cache()
    
    epoch_summary(valid_reports, 999, tags, label_lookup)

    test_reports = eval_one_epoch(test_loader, models_clone, collab_params, temp, 999, criterion, lamb=lamb, phase='Testing')
    torch.cuda.empty_cache()
    
    epoch_summary(test_reports, 999, tags, label_lookup)

    num_unpruned = 0

    for model in models_clone:
        num_unpruned += count_nonzero_parameters(model)

    print(f'Total Parameters Before Pruning: {num_before_prune}')
    print(f'Total Parameters After Pruning: {num_unpruned}')

    if save:
        for i, model in enumerate(models_clone):
            tag = tags[i]
            
            checkpoint = {
                'model': model.state_dict(),
            }
            
            torch.save(checkpoint, f'{save_path}/{tag}_pruned.pt')
            
    return
    
if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    main()

    