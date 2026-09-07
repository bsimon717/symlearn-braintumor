import os
import torch.multiprocessing as multiprocessing
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import json
import argparse

from model import SimpleCNN
from utils import *

from symlearn.loss import *
from symlearn.classify.Readout import Readout
from symlearn.classify.utils import *

def train(epochs, models, opts, scheds, data_loaders, collab_params, temp, criterion, uplift=10, eps=1e-7, lamb=1.0):
    
    label_lookup = {}
    label_lookup['personal'] = 'Average Personal Loss'
    label_lookup['symbiotic'] = 'Average Symbiotic Loss'
    label_lookup['accuracy'] = 'Accuracy'
    label_lookup['embedding'] = 'Average Embedding Loss'
    label_lookup['blame'] = 'Average Blame Loss'

    train_loader, val_loader, test_loader = data_loaders
    
    len_train = len(train_loader)
    len_valid = len(val_loader)
    len_test = len(test_loader)

    lt_reports = []

    for i in range(len(models)):
        lt_report_i = {
        'Validation': {},
        'Testing': {}
        }

        lt_reports.append(lt_report_i)

    if load_at_uplift:
        epoch_range = range(uplift, epochs)
    else:
        epoch_range = range(0, epochs)

    for epoch in epoch_range:
        
        train_reports = train_one_epoch(train_loader, models, opts, scheds, collab_params, temp, epoch, criterion, uplift=uplift, eps=eps, lamb=lamb)
        epoch_summary(train_reports, epoch, tags, label_lookup)
        
        valid_reports = eval_one_epoch(val_loader, models, collab_params, temp, epoch, criterion, uplift=uplift, eps=eps, lamb=lamb, phase='Validation')
        epoch_summary(valid_reports, epoch, tags, label_lookup)
        fill_lt_reports(lt_reports, valid_reports, 'Validation')
                            
        if epoch%5 == 0 or epoch == epochs-1:
            test_reports = eval_one_epoch(test_loader, models, collab_params, temp, epoch, criterion, uplift=uplift, eps=eps, lamb=lamb, phase='Testing')
            epoch_summary(test_reports, epoch, tags, label_lookup)   
            fill_lt_reports(lt_reports, test_reports, 'Testing')
            
        if epoch == uplift-1 and save_before_uplift == True:
            for i, model in enumerate(models):
                tag = tags[i]
                opt = opts[i]
                sched = scheds[i]
                
                checkpoint = {
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'opt': opt.state_dict(),
                    'sched': sched.state_dict(),
                    'last_step': sched.last_epoch
                }
                
                torch.save(checkpoint, f'{save_path}/{tag}_pre-uplift.pt')
            

    for tag, lt_report in zip(tags, lt_reports):
        for phase in lt_report.keys():
            plot_phase(lt_report, label_lookup, epochs, uplift, date_and_time, phase=phase, tag=tag)

    if save_end:
        for i, model in enumerate(models):
            tag = tags[i]
            opt = opts[i]
            sched = scheds[i]
            
            checkpoint = {
                'epoch': epoch,
                'model': model.state_dict(),
                'opt': opt.state_dict(),
                'sched': sched.state_dict(),
                'last_step': sched.last_epoch
            }
            
            torch.save(checkpoint, f'{save_path}/{tag}.pt')
    return

def plot_phase(lt_report, label_lookup, epochs, uplift, date_and_time, phase='Validation', tag='Model_A'):

    if not load_at_uplift:
        if phase == 'Validation':
            full_axis = list(range(0, epochs))
            post_uplift_axis = list(range(uplift, epochs))
            
        elif phase == 'Testing':
            full_axis = list(np.arange(0,epochs,5)) + [epochs]
            post_uplift_axis = list(np.arange(math.ceil(uplift/5)*5,epochs,5)) + [epochs]
    else:
        if phase == 'Validation':
            full_axis = list(range(uplift, epochs))
            post_uplift_axis = list(range(uplift, epochs))
            
        elif phase == 'Testing':
            full_axis = list(np.arange(uplift,epochs,5)) + [epochs]
            post_uplift_axis = list(np.arange(math.ceil(uplift/5)*5,epochs,5)) + [epochs]
            
    
    for key in lt_report[phase].keys():
        if tag == 'Readout' or key == 'blame':
            x_axis = post_uplift_axis
        else:
            x_axis = full_axis

        data_clean = [x for x in lt_report[phase][key] if x is not None]
        
        if key == 'accuracy' and tag == 'Readout':
            idx_best = np.argmax(data_clean)
            best_epoch = x_axis[idx_best]
            best_acc = data_clean[idx_best]
            label = f'{tag}: Best Accuracy=\n{best_acc:.4f} at Epoch {best_epoch}'
        else:
            label = f'{tag}: {label_lookup[key]}'
        
        try:
            plt.plot(x_axis, data_clean, color='black', linestyle='-', label=label)
        except:
            print(f"Error Encountered Plotting {tag}'s {phase} {key.capitalize()} Report")
            print('x axis: ', x_axis)
            print('data: ', data_clean)
            continue
            
        plt.title(f'{tag} {phase}: {label_lookup[key]} Per Epoch')
        plt.xlabel('Epoch')
        plt.ylabel(f'{label_lookup[key]}')
        plt.grid()
        if tag != 'Readout' and uplift != 0: plt.axvline(x=uplift, linestyle='dashed', color='red', label=f'Uplift: Epoch {uplift}')
        plt.legend()
        plt.savefig(f'{save_path}/{tag}/{phase}_{key}.png')
        plt.clf()
        
    return
    
def main():
    now = datetime.now()

    global date_and_time
    date_and_time = now.strftime('%d_%m_%y_%H_%M_%S')
    
    global only_create_json
    
    global save_best
    global save_end
    global save_before_uplift
    global save_path

    global load_at_uplift
    global load_path
    
    global tags
    
    ### Command-line Argument Parsing

    parser = argparse.ArgumentParser(prog='train',
                    description='train.py: Prototype Symbiotic Learning Set-Up',
                    epilog='Full description TBD.')

    parser.add_argument('-n', '--num_preR', default=3, type=int)
    parser.add_argument('-c', '--collab_params', nargs='*', type=float, help='Collaboration parameters. Specify <num> values in the range (0,1).')
    parser.add_argument('-o', '--only_create_json', default=False, type=bool)
    
    parser.add_argument('--comment', default='', type=str)
    parser.add_argument('--config_path', default=False, help='Path to folder containing existing config.json file.')
    
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--uplift', default=25, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--criterion', default='CE', type=str)
    
    parser.add_argument('--kernel_size', default=16, type=int)
    parser.add_argument('--kernel_stride', default=2, type=int)
    parser.add_argument('--conv_channels', default=32, type=int)
    parser.add_argument('--out_channels', default=128, type=int)
    parser.add_argument('--readout_hidden_dim', default=32, type=int)
    parser.add_argument('--num_heads', default=1, type=int)
    parser.add_argument('--num_fc', default=3, type=int)
    parser.add_argument('--temp', default=1.0, type=float)
    parser.add_argument('--lamb', default=1.0, type=float, help='Responsibility parameter lambda. Enables blame loss term after uplift.')

    parser.add_argument('--save_before_uplift', default=False, type=bool)
    parser.add_argument('--save_best', default=False, type=bool, help='TBD')
    parser.add_argument('--save_end', default=False, type=bool)
    
    parser.add_argument('--load_at_uplift', default=False, type=bool, help='Load pre-trained models A,B,C. Specify path in load_path argument.')
    parser.add_argument('--load_path', default=os.getcwd(), type=str)
                        
    known_args, unknown_args = parser.parse_known_args()    

    if len(unknown_args) > 0:
        print('Received unrecognized arguments: ', unknown_args)
        print('Recognized arguments: ', known_args)
        print('Exiting.')
        return
                
    args = parser.parse_args()    

    ###

    num_preR = args.num_preR
    collab_params = args.collab_params
    
    only_create_json = args.only_create_json
    
    save_end = args.save_end
    save_before_uplift = args.save_before_uplift
    save_best = args.save_best
    
    load_at_uplift = args.load_at_uplift
    load_path = args.load_path

    if load_at_uplift:
        assert load_path!= False, 'Must specify load_path if load_before_uplift is enabled.'
        
    if args.comment != '':
        save_path = f'./sym_logs/{date_and_time}_{args.comment}'
    else:
        save_path = f'./sym_logs/{date_and_time}'
        
    os.mkdir(save_path)

    config_path = args.config_path
    crit = args.criterion
    
    epochs = args.epochs
    uplift = args.uplift
    batch_size = args.batch_size
    
    kernel_size = args.kernel_size
    kernel_stride = args.kernel_stride
    conv_channels = args.conv_channels
    out_channels = args.out_channels
    readout_hidden_dim = args.readout_hidden_dim
    num_heads = args.num_heads
    num_fc = args.num_fc
    temp = args.temp
    lamb = args.lamb

    if config_path == False:
        assert epochs > 0, 'Must specify non-zero number of epochs.'
        assert uplift <= epochs, 'Uplift must be less than or equal to the total number of epochs.'
        assert num_heads >= 1, 'Number of attention heads in Readout must be at least 1.'
        assert num_fc >= 1, 'pre-Readout Models require at least 1 fully-conencted layer.'
        
        bad_params = 0
        for param in collab_params:
            if param < 0 or param > 1:
                bad_params += 1
    
        assert bad_params == 0, f'All collaboration parameters must fall in the range (0,1). Current parameters {collab_params}.'
        
        print(f'Creating config.json')
    
        config = {}
    
        config['models'] = {}
        config['models']['pre-Readout'] = {}
        config['models']['Readout'] = {}
        
        config['training'] = {}
        
        config['training']['epochs'] = epochs
        config['training']['uplift'] = uplift
        config['training']['batch_size'] = batch_size
        config['training']['temp'] = temp
        config['training']['lamb'] = lamb
        config['training']['criterion'] = crit

        config['models']['pre-Readout']['num_preR'] = num_preR
        config['models']['pre-Readout']['collab_params'] = collab_params
        config['models']['pre-Readout']['kernel_size'] = kernel_size
        config['models']['pre-Readout']['kernel_stride'] = kernel_stride
        config['models']['pre-Readout']['conv_channels'] = conv_channels
        config['models']['pre-Readout']['out_channels'] = out_channels
        config['models']['pre-Readout']['num_fc'] = num_fc

        config['models']['Readout']['readout_hidden_dim'] = readout_hidden_dim
        config['models']['Readout']['num_heads'] = num_heads
        
        # Open file in write mode ('w')
        with open(f"{save_path}/config.json", "w") as file:
            json.dump(config, file, indent=4, sort_keys=True)
    
        print('Done.')

        if args.only_create_json:
            print(f'Path to config.json: {save_path}')
            print('Exiting.')
            return 
    else:

        print(f'Loading settings from path {config_path}')
        with open(f'{config_path}/config.json', 'r') as file:
           config = json.load(file)

        epochs = config['training']['epochs']
        uplift = config['training']['uplift']
        batch_size = config['training']['batch_size']
        temp = config['training']['temp']
        lamb = config['training']['lamb']
        crit = config['training']['criterion']
        
        readout_hidden_dim = config['models']['Readout']['readout_hidden_dim']
        num_heads = config['models']['Readout']['num_heads']

        if not load_at_uplift:
            num_preR = config['models']['pre-Readout']['num_preR']
            collab_params = config['models']['pre-Readout']['collab_params']
            kernel_size = config['models']['pre-Readout']['kernel_size']
            kernel_stride = config['models']['pre-Readout']['kernel_stride']
            conv_channels = config['models']['pre-Readout']['conv_channels']
            out_channels = config['models']['pre-Readout']['out_channels']
            num_fc = config['models']['pre-Readout']['num_fc']
        else:
            print('Loading settings from pre-trained models.')
            with open(f'{load_path}/config.json', 'r') as file:
                temp_config = json.load(file)
                
            num_preR = temp_config['models']['pre-Readout']['num_preR']
            collab_params = temp_config['models']['pre-Readout']['collab_params']
            kernel_size = temp_config['models']['pre-Readout']['kernel_size']
            kernel_stride = temp_config['models']['pre-Readout']['kernel_stride']
            conv_channels = temp_config['models']['pre-Readout']['conv_channels']
            num_fc = temp_config['models']['pre-Readout']['num_fc']

            config['models']['pre-Readout']['num_preR'] = num_preR
            config['models']['pre-Readout']['collab_params'] = collab_params
            config['models']['pre-Readout']['kernel_size'] = kernel_size
            config['models']['pre-Readout']['kernel_stride'] = kernel_stride
            config['models']['pre-Readout']['conv_channels'] = conv_channels
            config['models']['pre-Readout']['out_channels'] = out_channels
            config['models']['pre-Readout']['num_fc'] = num_fc

        with open(f"{save_path}/config.json", "w") as file:
            json.dump(config, file, indent=4, sort_keys=True)

    tags = [f'Model_{i}' for i in range(num_preR)] + ['Readout']
    for tag in tags:
        os.mkdir(f'{save_path}/{tag}')
        
    device = torch.device("cuda")
    torch.cuda.empty_cache()

    if crit == 'CE':
        criterion = nn.CrossEntropyLoss()
    else:
        raise Exception(f'Unsupported Criterion: {crit}')
    
    data_loaders = get_data_loaders(batch_size)

    num_train_batches = len(data_loaders[0])
    
    pct_start = (uplift/2) / epochs
    
    models = []
    opts = []
    scheds = []
    for _ in range(num_preR):
        model_i = SimpleCNN(kernel_size=kernel_size, kernel_stride=kernel_stride, conv_channels=conv_channels, out_channels=out_channels, padding=1, num_fc=num_fc, num_preR=num_preR).to(device)
        opt_i = optim.AdamW(model_i.parameters(), lr=1e-7, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
        sched_i = torch.optim.lr_scheduler.OneCycleLR(opt_i, max_lr=5e-5, steps_per_epoch=num_train_batches, epochs=epochs, pct_start=pct_start) 
        
        models.append(model_i)
        opts.append(opt_i)
        scheds.append(sched_i)
    
    if load_at_uplift:
        assert load_path != False, 'Option load_at_uplift is set to True without a load_path specified.'
        print('Loading pre-trained models. Training will begin at uplift.')

        for i in range(num_preR):
            check = torch.load(f'{load_path}/Model_{i}_pre-uplift.pt', map_location=torch.device('cpu'))
            models[i].load_state_dict(check['model'])
            opts[i].load_state_dict(check['opt'])
            scheds[i].load_state_dict(check['sched'])
        
    num_trainable_params = sum(p.numel() for p in models[0].parameters() if p.requires_grad)
    readout = Readout(hidden_dim=readout_hidden_dim, num_heads=num_heads, num_preR=num_preR, preR_dim=out_channels).to(device)
    models.append(readout)
    opt_F = optim.AdamW(models[-1].parameters(), lr=1e-7, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
    opts.append(opt_F)
    sched_F = torch.optim.lr_scheduler.OneCycleLR(opts[-1], max_lr=5e-5, steps_per_epoch=num_train_batches, epochs=epochs-uplift+1)
    scheds.append(sched_F)
    
    num_trainable_params_readout = sum(p.numel() for p in readout.parameters() if p.requires_grad)
        
    print()
    print('Total Number of Trainable Parameters: ', num_preR*num_trainable_params+num_trainable_params_readout)
    print('\t-Total pre-Readout Trainable Parameters: ', num_preR*num_trainable_params)
    print('\t\t-Number of Trainable Parameters per pre-Readout Model: ', num_trainable_params)
    print('\t-Number of Trainable Parameters in Readout Block: ', num_trainable_params_readout)
    print()

    with open(f"{save_path}/param_breakdown.txt", "w", encoding="utf-8") as file:
        file.write(f"Total Number of Trainable Parameters: {num_preR*num_trainable_params+num_trainable_params_readout}\n")
        file.write(f"\tTotal pre-Readout Trainable Parameters: {num_preR*num_trainable_params}\n")
        file.write(f"\t\t-Number of Trainable Parameters per pre-Readout Model: {num_trainable_params}\n")
        file.write(f"\tNumber of Trainable Parameters in Readout Block: {num_trainable_params_readout}")
    
    train(epochs, models, opts, scheds, data_loaders, collab_params, temp, criterion, uplift=uplift, lamb=lamb)
    print('Exiting.')
    return

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    main()