import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import json
import argparse

from model import SimpleCNN, Readout
from utils import *

def tri_collab_train(epochs, models, opts, scheds, readout, opt_F, sched_F, train_loader, val_loader, test_loader, collab_params, temp, uplift=10, eps=1e-7, contrastive=False, lamb=1.0):
    
    label_lookup = {}
    label_lookup['personal'] = 'Average Personal Loss'
    label_lookup['total'] = 'Average Total Loss'
    label_lookup['accuracy'] = 'Accuracy'
    label_lookup['contrastive'] = 'Average Contrastive Loss'
    label_lookup['blame'] = 'Average Blame Loss'
    
    len_train = len(train_loader)
    len_valid = len(val_loader)
    len_test = len(test_loader)

    lt_report_A = {
        'Validation': {},
        'Testing': {}
    }

    lt_report_B = {
        'Validation': {},
        'Testing': {}
    }
    
    lt_report_C = {
        'Validation': {},
        'Testing': {}
    }

    lt_report_F = {
        'Validation': {},
        'Testing': {}
    }

    lt_reports = [lt_report_A, lt_report_B, lt_report_C, lt_report_F]

    if load_at_uplift:
        epoch_range = range(uplift, epochs)
    else:
        epoch_range = range(0, epochs)

    for epoch in epoch_range:
        
        report_A, report_B, report_C, report_F = train_one_epoch(train_loader, models, opts, scheds, readout, opt_F, sched_F, collab_params, temp, epoch, uplift=uplift, eps=eps, contrastive=contrastive, lamb=lamb)
        train_reports = (report_A, report_B, report_C, report_F)
        epoch_summary(train_reports, epoch, tags, label_lookup)

        valid_reports = eval_one_epoch(val_loader, models, readout, collab_params, temp, epoch, uplift=uplift, eps=eps, phase='Validation', contrastive=contrastive, lamb=lamb)
        epoch_summary(valid_reports, epoch, tags, label_lookup)
        fill_lt_reports(lt_reports, valid_reports, 'Validation', epoch, uplift, lamb)
        
                            
        if epoch%5 == 0 or epoch == epochs-1:
            test_reports = eval_one_epoch(test_loader, models, readout, collab_params, temp, epoch, uplift=uplift, eps=eps, phase='Testing', contrastive=contrastive, lamb=lamb)
            epoch_summary(test_reports, epoch, tags, label_lookup)   
            fill_lt_reports(lt_reports, test_reports, 'Testing', epoch, uplift, lamb)
            
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

        readout_checkpoint = {
                'epoch': epoch,
                'model': readout.state_dict(),
                'opt': opt_F.state_dict(),
                'sched': sched_F.state_dict(),
                'last_step': sched_F.last_epoch
            }
        
        torch.save(readout.state_dict(), f'{save_path}/Readout.pt')
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

        if key == 'accuracy' and tag == 'Readout':
            idx_best = np.argmax(lt_report[phase][key])
            best_epoch = x_axis[idx_best]
            best_acc = lt_report[phase][key][idx_best]
            label = f'{tag}: Best Accuracy=\n{best_acc:.4f} at Epoch {best_epoch}'
        else:
            label = f'{tag}: {label_lookup[key]}'
        
        try:
            plt.plot(x_axis, lt_report[phase][key], color='black', linestyle='-', label=label)
        except:
            print(f"Error Encountered Plotting {tag}'s {phase} {key.capitalize()} Report")
            print('x axis: ', x_axis)
            print('data: ', lt_report[phase][key])
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
    tags = ['Model_A', 'Model_B', 'Model_C', 'Readout']
    
    ### Command-line Argument Parsing

    parser = argparse.ArgumentParser(prog='tri_collab',
                    description='tri_collab.py: Prototype Tri-Collaborative Learning Set-Up',
                    epilog='Full description TBD.')
    
    parser.add_argument('-o', '--only_create_json', default=False, type=bool)
    parser.add_argument('--comment', default='', type=str)
    
    parser.add_argument('--save_before_uplift', default=False, type=bool)
    parser.add_argument('--save_best', default=False, type=bool, help='TBD')
    parser.add_argument('--save_end', default=False, type=bool)

    parser.add_argument('--config_path', default=False, help='Path to existing config file. ')
    
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--uplift', default=25, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--contrastive', default=True, type=float, help='Enable contrastive loss term for individual embeddings.')
    parser.add_argument('--kernel_size', default=32, type=int)
    parser.add_argument('--kernel_stride', default=8, type=int)
    parser.add_argument('--out_channels', default=64, type=int)
    parser.add_argument('--readout_hidden_dim', default=64, type=int)
    parser.add_argument('--num_heads', default=1, type=int)
    parser.add_argument('--num_fc', default=1, type=int)
    
    parser.add_argument('--collab_params', nargs=3, default=[0.5,0.5,0.5], type=float, help='Collaboration parameters: alpha, beta, gamma. Specify three values in range [0,1].')
    parser.add_argument('--temp', default=1.0, type=float)
    parser.add_argument('--lamb', default=0.0, type=float, help='Responsibility parameter lambda. Enables blame loss term after uplift.')

    parser.add_argument('--load_at_uplift', default=False, type=bool, help='Load pre-trained models A,B,C. Specify path in load_path argument.')
    parser.add_argument('--load_path', default=os.getcwd(), type=str)
                        
    known_args, unknown_args = parser.parse_known_args()    

    if len(unknown_args) > 0:
        print('Received unrecognized arguments: ', unkown_args)
        print('Recognized arguments: ', known_args)
        print('Exiting.')
        return
                
    args = parser.parse_args()
    
    ###
    
    only_create_json = args.only_create_json
    
    save_end = args.save_end
    save_before_uplift = args.save_before_uplift
    save_best = args.save_best
    
    load_at_uplift = args.load_at_uplift
    load_path = args.load_path

    if load_at_uplift:
        assert load_path!= False, 'Must specify load_path if load_before_uplift is enabled.'
        
    if args.comment != '':
        save_path = f'./tri_collab_logs/{date_and_time}_{args.comment}'
    else:
        save_path = f'./tri_collab_logs/{date_and_time}'
        
    os.mkdir(save_path)

    for tag in tags:
        os.mkdir(f'{save_path}/{tag}')
    
    config_path = args.config_path

    epochs = args.epochs
    uplift = args.uplift

    batch_size = args.batch_size
    contrastive = args.contrastive
    
    kernel_size = args.kernel_size
    kernel_stride = args.kernel_stride
    out_channels = args.out_channels
    readout_hidden_dim = args.readout_hidden_dim
    num_heads = args.num_heads
    num_fc = args.num_fc

    collab_params = args.collab_params    
    temp = args.temp
    lamb = args.lamb

    if config_path == False:
        assert epochs > 0, 'Must specify non-zero number of epochs.'
        assert uplift <= epochs, 'Uplift must be less than or equal to the total number of epochs.'
        assert num_heads >= 1, 'Number of attention heads in Readout must be at least 1.'
        assert num_fc >= 1, 'ABC Models require at least 1 fully-conencted layer.'
        
        bad_params = 0
        for param in collab_params:
            if param < 0 or param > 1:
                bad_params += 1
    
        assert bad_params == 0, f'All collaboration parameters must fall in the range [0,1]. Current parameters {collab_params}.'
        
        print(f'Creating config.json')
    
        config = {}
    
        config['models'] = {}
        config['models']['ABC'] = {}
        config['models']['Readout'] = {}
        
        config['training'] = {}
        
        config['training']['epochs'] = epochs
        config['training']['uplift'] = uplift
        config['training']['batch_size'] = batch_size
        config['training']['contrastive'] = contrastive
        config['training']['temp'] = temp
        config['training']['lamb'] = lamb
    
        config['models']['ABC']['kernel_size'] = kernel_size
        config['models']['ABC']['kernel_stride'] = kernel_stride
        config['models']['ABC']['out_channels'] = out_channels
        config['models']['ABC']['num_fc'] = num_fc
        
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
        contrastive = config['training']['contrastive']
        temp = config['training']['temp']
        lamb = config['training']['lamb']
        
        readout_hidden_dim = config['models']['Readout']['readout_hidden_dim']
        num_heads = config['models']['Readout']['num_heads']

        if not load_at_uplift:
            kernel_size = config['models']['ABC']['kernel_size']
            kernel_stride = config['models']['ABC']['kernel_stride']
            out_channels = config['models']['ABC']['out_channels']
            num_fc = config['models']['ABC']['num_fc']
        else:
            print('Loading ABC settings from pre-trained model.')
            with open(f'{load_path}/config.json', 'r') as file:
                temp_config = json.load(file)

            kernel_size = temp_config['models']['ABC']['kernel_size']
            kernel_stride = temp_config['models']['ABC']['kernel_stride']
            out_channels = temp_config['models']['ABC']['out_channels']
            num_fc = temp_config['models']['ABC']['num_fc']

            config['models']['ABC']['kernel_size'] = kernel_size
            config['models']['ABC']['kernel_stride'] = kernel_stride
            config['models']['ABC']['out_channels'] = out_channels
            config['models']['ABC']['num_fc'] = num_fc

        with open(f"{save_path}/config.json", "w") as file:
            json.dump(config, file, indent=4, sort_keys=True)

    device = torch.device("cuda")
    torch.cuda.empty_cache()
    
    train_loader, val_loader, test_loader = get_data_loaders(batch_size)
    
    model_A = SimpleCNN(kernel_size=kernel_size, kernel_stride=kernel_stride, out_channels=out_channels, padding=1, num_fc=num_fc, aux_embeds=True).to(device)
    model_B = SimpleCNN(kernel_size=kernel_size, kernel_stride=kernel_stride, out_channels=out_channels, padding=1, num_fc=num_fc, aux_embeds=True).to(device)
    model_C = SimpleCNN(kernel_size=kernel_size, kernel_stride=kernel_stride, out_channels=out_channels, padding=1, num_fc=num_fc, aux_embeds=True).to(device)

    models = [model_A, model_B, model_C]
    
    if load_at_uplift:
        assert load_path != False, 'Option load_at_uplift is set to True without a load_path specified.'
        print('Loading pre-trained models A, B, and C. Training will begin at uplift.')

        A_check = torch.load(f'{load_path}/Model_A_pre-uplift.pt', map_location=torch.device('cpu'))
        B_check = torch.load(f'{load_path}/Model_B_pre-uplift.pt', map_location=torch.device('cpu'))
        C_check = torch.load(f'{load_path}/Model_B_pre-uplift.pt', map_location=torch.device('cpu'))

        checks = [A_check, B_check, C_check]

        for model, check in zip(models, checks):
            model.load_state_dict(check['model'])
        
    num_trainable_params = sum(p.numel() for p in model_A.parameters() if p.requires_grad)
    readout = Readout(hidden_dim=readout_hidden_dim, num_heads=num_heads).to(device)
    
    num_trainable_params_readout = sum(p.numel() for p in readout.parameters() if p.requires_grad)
        
    print()
    print('Total Number of Trainable Parameters: ', 3*num_trainable_params+num_trainable_params_readout)
    print('\t-Total sub-Model Trainable Parameters: ', 3*num_trainable_params)
    print('\t\t-Number of Trainable Parameters per sub-Model: ', num_trainable_params)
    print('\t-Number of Trainable Parameters in Readout Block: ', num_trainable_params_readout)
    print()

    with open(f"{save_path}/param_breakdown.txt", "w", encoding="utf-8") as file:
        file.write(f"Total Number of Trainable Parameters: {3*num_trainable_params+num_trainable_params_readout}\n")
        file.write(f"\tTotal sub-Model Trainable Parameters: {3*num_trainable_params}\n")
        file.write(f"\t\t-Number of Trainable Parameters per sub-Model: {num_trainable_params}\n")
        file.write(f"\tNumber of Trainable Parameters in Readout Block: {num_trainable_params_readout}")
    
    pct_start = (uplift/2) / epochs

    opt_A = optim.AdamW(model_A.parameters(), lr=1e-7, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
    opt_B = optim.AdamW(model_B.parameters(), lr=1e-7, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
    opt_C = optim.AdamW(model_C.parameters(), lr=1e-7, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)

    opts = [opt_A, opt_B, opt_C]
    
    if load_at_uplift:
        scheds = []
        for opt, check in zip(opts, checks):
            opt.load_state_dict(check['opt'])

            sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=5e-5, steps_per_epoch=len(train_loader), epochs=epochs, pct_start=pct_start, last_epoch=check['last_step']) 
            sched.load_state_dict(check['sched'])
            scheds.append(sched)
            
    else:
        
        sched_A = torch.optim.lr_scheduler.OneCycleLR(opt_A, max_lr=5e-5, steps_per_epoch=len(train_loader), epochs=epochs, pct_start=pct_start) 
        sched_B = torch.optim.lr_scheduler.OneCycleLR(opt_B, max_lr=5e-5, steps_per_epoch=len(train_loader), epochs=epochs, pct_start=pct_start)  
        sched_C = torch.optim.lr_scheduler.OneCycleLR(opt_C, max_lr=5e-5, steps_per_epoch=len(train_loader), epochs=epochs, pct_start=pct_start)

        scheds = [sched_A, sched_B, sched_C]

    
    opt_F = optim.AdamW(readout.parameters(), lr=1e-7, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
    sched_F = torch.optim.lr_scheduler.OneCycleLR(opt_F, max_lr=5e-5, steps_per_epoch=len(train_loader), epochs=epochs-uplift+1)

    tri_collab_train(epochs, models, opts, scheds, readout, opt_F, sched_F, train_loader, val_loader, test_loader, collab_params, temp, uplift=uplift, contrastive=contrastive, lamb=lamb)
    print('Exiting.')
    return

if __name__ == "__main__":
    main()