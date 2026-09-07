from functools import partial
import numpy as np
import torch
from torch._higher_order_ops import map
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image
from pathlib import Path
from model import SimpleCNN, Readout
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm
import torch.optim as optim
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
    
def embed_sim(x1, x2):
    cosine_sim = F.cosine_similarity(x1, x2, dim=0)
    return (cosine_sim+1)/2

def embed_summand(src, aux, delta=0.5):
    return torch.exp( (1/delta)*embed_sim(src, aux) ) - 1

def embed_loss(src, auxs):
    num_aux = len(auxs)
    temp_func = lambda aux: embed_summand(src, aux)
    temp_func = torch.vmap(temp_func)

    auxs = torch.stack(auxs)
    sum_terms = temp_func(auxs)
    sum_terms = torch.sum(sum_terms, dim=0)
    pre_factor = 1/num_aux
    
    L_embed = pre_factor*sum_terms

    return L_embed.mean()

def train_one_epoch(train_loader, models, opts, scheds, collab_params, temp, epoch, criterion, uplift=10, eps=1e-7, lamb=1.0):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    reports = [{} for model in models]
    personals = [[] for model in models]
    embs = [[] for model in models[:-1]]
    blames = [[] for model in models[:-1]]
    syms = [[] for model in models]
    
    for model in models:
        model.train()

    ## TRAINING LOOP
    pbar_train = tqdm(train_loader, total=len(train_loader))
    pbar_train.set_description(f'Epoch {epoch}: Training')
    for img, label in pbar_train:
        for opt in opts:
            opt.zero_grad()

        img = img.to(device)

        inds = [model.embed(img) for model in models[:-1]]
        
        ind_concat = torch.cat(inds, dim=1)
        ind_stack = torch.stack(inds)
    
        logits_list = [model(ind_concat.clone()).clone().cpu() for model in models[:-1]]
        
        probs_list = [F.softmax(logits.clone().detach(),dim=-1).to('cpu') for logits in logits_list]
        preds_list = [torch.argmax(probs.clone().detach(),dim=1) for probs in probs_list]
        
        L_is = []
        L_emb_is = []
        for i, personal in enumerate(personals[:-1]):
            L_i = criterion(logits_list[i], label)
            L_is.append(L_i)
            personal.append(float(L_i.clone().detach()))

            src = ind_stack[i].clone()
            auxs = [ind_stack[i].clone() for i in range(len(ind_stack))]
            auxs.pop(i)
            L_emb_i = embed_loss(src, auxs).cpu()
            L_emb_is.append(L_emb_i)
            embs[i].append(float(L_emb_i.clone().detach()))
        
        L_i_tensor = torch.stack(L_is)
        L_emb_tensor = torch.stack(L_emb_is)
        
        L_syms = []
        for i in range(len(L_is)):
            param = collab_params[i]
            aux_idxs = [j!=i for j in range(len(L_is))]
            aux_L = L_i_tensor.clone()[aux_idxs]

            L_sym_i = (1-param)*L_i_tensor[i] + param*torch.sum(aux_L) + (param**2)*L_emb_tensor[i]

            L_syms.append(L_sym_i)

        if epoch >= uplift:
            upstream_input = torch.cat(logits_list, dim=1).to(device)

            final_logits = models[-1](upstream_input, ind_concat.clone()).to('cpu')
            final_probs = F.softmax(final_logits.clone(), dim=-1)
            final_preds = torch.argmax(final_probs, dim=1)

            L_F = criterion(final_logits, label)
            personals[-1].append(float(L_F.clone().detach()))

            L_up_sum = eps + torch.sum(L_i_tensor).detach()

            L_sym_F = L_F*(1+torch.exp(eps-temp*L_up_sum))
            L_syms.append(L_sym_F)

            for i, L_i in enumerate(L_i_tensor.clone().detach()):
                L_blame_i = lamb*(L_i/L_up_sum)*L_F.clone()
                blames[i].append(float(L_blame_i.clone().detach()))
                L_syms[i] = L_syms[i] + L_blame_i

        for i, L_sym_i in enumerate(L_syms):
            syms[i].append(float(L_sym_i.clone().detach()))
            if i != len(L_syms):
                L_sym_i.backward(retain_graph=True)
            else:
                L_sym_i.backward()
            
        for i in range(len(syms)-1):
            opts[i].step()
            scheds[i].step()

        if epoch >= uplift:
            opts[-1].step()
            scheds[-1].step()
        
    ## FILL REPORTS
    for i, report in enumerate(reports):
        
        if i != len(reports)-1:
            report['personal'] = np.mean(personals[i])
            report['embedding']= np.mean(embs[i])
            if epoch >= uplift:
                report['blame'] = np.mean(blames[i])
            else:
                report['blame'] = None
                
            report['symbiotic'] = np.mean(syms[i])
        else:
            if epoch >= uplift:
                report['personal'] = np.mean(personals[i])
                report['symbiotic'] = np.mean(syms[i])
            else:
                report['personal'] = None
                report['symbiotic'] = None
    
    return reports
            
def eval_one_epoch(eval_loader, models, collab_params, temp, epoch, criterion, uplift=10, eps=1e-7, lamb=1.0, phase='Validation'):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    reports = [{} for model in models]
    personals = [[] for model in models]
    embs = [[] for model in models[:-1]]
    blames = [[] for model in models[:-1]]
    syms = [[] for model in models]
    preds = [[] for model in models]
    
    for model in models:
        model.eval()
        model.to(device)
        
    all_labels = []
    
    ## VALIDATION LOOP
    pbar_eval = tqdm(eval_loader, total=len(eval_loader))
    pbar_eval.set_description(f'Epoch {epoch}: {phase}')
    with torch.no_grad():
        for img, label in pbar_eval:
    
            img = img.to(device)
            all_labels += label.tolist()

            inds = [model.embed(img) for model in models[:-1]]
            
            ind_concat = torch.cat(inds, dim=1)
            ind_stack = torch.stack(inds)
            
            logits_list = [model(ind_concat.clone()).clone().cpu() for model in models[:-1]]
            
            probs_list = [F.softmax(logits.clone(),dim=-1).to('cpu') for logits in logits_list]
            preds_list = [torch.argmax(probs.clone(),dim=1).tolist() for probs in probs_list]
            
            L_is = []
            L_emb_is = []
            for i, personal in enumerate(personals[:-1]):
                L_i = criterion(logits_list[i], label)
                L_is.append(L_i)
                personal.append(float(L_i.clone()))

                preds[i] += preds_list[i]
                
                src = ind_stack[i].clone()
                auxs = [ind_stack[i].clone() for i in range(len(ind_stack))]
                auxs.pop(i)
                L_emb_i = embed_loss(src, auxs).cpu()
                L_emb_is.append(L_emb_i)
                embs[i].append(float(L_emb_i.clone()))
            
            L_i_tensor = torch.stack(L_is)
            L_emb_tensor = torch.stack(L_emb_is)
            
            L_syms = []
            for i in range(len(L_is)):
                param = collab_params[i]
                aux_idxs = [j!=i for j in range(len(L_is))]
                aux_L = L_i_tensor.clone()[aux_idxs]
    
                L_sym_i = (1-param)*L_i_tensor[i] + param*torch.sum(aux_L) + (param**2)*L_emb_tensor[i]
    
                L_syms.append(L_sym_i)
    
            if epoch >= uplift:
                upstream_input = torch.cat(logits_list, dim=1).to(device)
    
                final_logits = models[-1](upstream_input, ind_concat.clone()).to('cpu')
                final_probs = F.softmax(final_logits.clone(), dim=-1)
                final_preds = torch.argmax(final_probs, dim=1).tolist()
                preds[-1] += final_preds
                
                L_F = criterion(final_logits, label)
                personals[-1].append(float(L_F.clone()))
    
                L_up_sum = eps + torch.sum(L_i_tensor)
    
                L_sym_F = L_F*(1+torch.exp(eps-temp*L_up_sum))
                L_syms.append(L_sym_F)
    
                for i, L_i in enumerate(L_i_tensor.clone()):
                    L_blame_i = lamb*(L_i/L_up_sum)*L_F.clone()
                    blames[i].append(float(L_blame_i.clone()))
                    L_syms[i] = L_syms[i] + L_blame_i
    
            for i, L_sym_i in enumerate(L_syms):
                syms[i].append(float(L_sym_i.clone()))

    ## FILL REPORTS
    for i, report in enumerate(reports):
        
        if i != len(reports)-1:
            report['personal'] = np.mean(personals[i])
            report['embedding']= np.mean(embs[i])
            if epoch >= uplift:
                report['blame'] = np.mean(blames[i])
            else:
                report['blame'] = None
                
            report['symbiotic'] = np.mean(syms[i])
            report['accuracy'] = accuracy_score(all_labels, preds[i])
        else:
            if epoch >= uplift:
                report['personal'] = np.mean(personals[i])
                report['symbiotic'] = np.mean(syms[i])
                report['accuracy'] = accuracy_score(all_labels, preds[i])
            else:
                report['personal'] = None
                report['symbiotic'] = None
                report['accuracy'] = None
    
    return reports

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