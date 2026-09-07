import argparse
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.nn.utils.prune as prune
import optuna
from model import SimpleCNN, Readout
from sklearn.metrics import accuracy_score
import json
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import torch.multiprocessing as multiprocessing
from copy import deepcopy

from utils import *

def main():
    global models
    global num_preR
    global collab_params
    global batch_size
    global val_loader
    global criterion
    global temp
    global lamb
    global tags
    global device

    global min_pruning
    global max_pruning
    
    device = torch.device("cuda")
    
    parser = argparse.ArgumentParser(prog='bayesian_pruning',
                    description='bayesian_pruning.py',
                    epilog='Full description TBD.')

    parser.add_argument('--n_trials', default=50, type=int, help='Number of trials in Bayesian Optimization')
    parser.add_argument('--load_path', default=False, type=str, help='Path to folder containing pre-trained models.')
    parser.add_argument('--min_pruning', default=False, type=float, help='Lower bound on the amount a model will be pruned.')
    parser.add_argument('--max_pruning', default=False, type=float, help='Upper bound on the amount a model will be pruned.')
    
    args = parser.parse_args()
    n_trials = args.n_trials
    load_path = args.load_path
    min_pruning = args.min_pruning
    max_pruning = args.max_pruning
    
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
    
    sampler = optuna.samplers.TPESampler(seed=717)
    study = optuna.create_study(directions=["maximize", "maximize"], sampler=sampler)
    study.optimize(objective, n_trials=n_trials)

    print("Number of finished trials: ", len(study.trials))

    print("Pareto front:")

    trials = sorted(study.best_trials, key=lambda t: t.values)
    
    for trial in trials:
        print(f"  Trial#{trial.number}")
        print(
            f"    Values: Values={trial.values}"
        )
        print(f"    Params: {trial.params}")

    fig = optuna.visualization.plot_pareto_front(study, target_names=['Average %Pruning', 'Readout Accuracy'])
    fig.write_image(f'{load_path}/pareto_front.png')

    with open(f"{load_path}/pareto_front.txt", "w", encoding="utf-8") as file:
        file.write('Pareto Front:\n')
        for trial in trials:
            file.write(f"  Trial#{trial.number}")
            file.write(
            f"    Values: Values={trial.values}"
            )
            file.write(f"    Params: {trial.params}")

    df = study.trials_dataframe()
    df.to_json(f"{load_path}/all_trials.json", orient="records", date_format="iso", indent=4)
    
def objective(trial):
    prune_percs = []

    label_lookup = {}
    label_lookup['personal'] = 'Average Personal Loss'
    label_lookup['symbiotic'] = 'Average Symbiotic Loss'
    label_lookup['accuracy'] = 'Accuracy'
    label_lookup['embedding'] = 'Average Embedding Loss'
    label_lookup['blame'] = 'Average Blame Loss'
    
    models_clone = [deepcopy(model).to(device) for model in models]

    for i in range(num_preR):
        x = trial.suggest_float(f"Model_{i} Unstructured Pruning Percentage",min_pruning,max_pruning)
        prune_percs.append(x)

        prune_preReadout(models_clone[i], x)
        
    valid_reports = eval_one_epoch(val_loader, models_clone, collab_params, temp, 999, criterion, lamb=lamb, phase='Validation')
    torch.cuda.empty_cache()
    
    epoch_summary(valid_reports, 999, tags, label_lookup)

    readout_acc = valid_reports[-1]['accuracy']
    
    return np.mean(prune_percs), readout_acc

def count_nonzero_parameters(model):
    return sum((p != 0).sum().item() for p in model.parameters())

def prune_preReadout(model, amount):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=amount)
            prune.remove(module, 'weight')

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    main()