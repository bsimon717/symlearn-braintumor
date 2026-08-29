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
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.optim as optim
from sklearn.metrics import accuracy_score
import torch.multiprocessing as multiprocessing
TORCHDYNAMO_VERBOSE=1

def epoch_summary(reports, epoch, tags, label_lookup):
    print(f'Summary:')
    for i, report in enumerate(reports):
        tag = tags[i]

        if tag == 'Readout' and len(report) == 0:
            continue
        
        print(f'\t- {tag}:')
        for label in report.keys():
            if label != 'accuracy':
                print(f'\t\t-- {label_lookup[label]}: {report[label]:.4}')
            else:
                print()
                print(f'\t\t-- {label_lookup[label]}: {report[label]:.4}')
        print()

    return

def fill_lt_reports(lt_reports, reports, phase, epoch, uplift, lamb):

    for lt_report, report in zip(lt_reports, reports):
            keys = list(report.keys())
            for key in keys:
                if key not in lt_report[phase].keys():
                    lt_report[phase][key] = [report[key]]
                else:
                    lt_report[phase][key].append(report[key])
                    
    return
    
def embed_sim(x1, x2):

    cosine_sim = F.cosine_similarity(x1, x2)

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
    
def train_one_epoch(train_loader, models, opts, scheds, readout, opt_F, sched_F, collab_params, temp, epoch, uplift=10, eps=1e-7, lamb=1.0, contrastive=False):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = nn.CrossEntropyLoss()

    report_A = {}
    report_B = {}
    report_C = {}
    
    personal_A = []
    personal_B = []
    personal_C = []

    if contrastive:
        con_A = []
        con_B = []
        con_C = []

    if lamb > 0.0:
        blame_A = []
        blame_B = []
        blame_C = []
    
    tot_A = []
    tot_B = []
    tot_C = []

    
    model_A, model_B, model_C, = models
    
    model_A.train()
    model_B.train()
    model_C.train()
    
    opt_A, opt_B, opt_C = opts
    sched_A, sched_B, sched_C, = scheds
    alpha, beta, gamma = collab_params

    personal_F = []
    tot_F = []
    report_F = {}
    readout.train()
    
    pbar_train = tqdm(train_loader, total=len(train_loader))
    pbar_train.set_description(f'Epoch {epoch}: Training')
    for img, label in pbar_train:
        
        opt_A.zero_grad()
        opt_B.zero_grad()
        opt_C.zero_grad()
        
        img = img.to(device)
        
        ind_A = model_A.embed(img)
        ind_B = model_B.embed(img)
        ind_C = model_C.embed(img)
            
        ind_concat = torch.cat([ind_A, ind_B, ind_C], dim=1)

        logits_A = model_A(ind_concat).clone().cpu()
        logits_B = model_B(ind_concat).clone().cpu()
        logits_C = model_C(ind_concat).clone().cpu()
        
        probs_A = F.softmax(logits_A.clone().detach(), dim=-1).to('cpu')
        pred_A = torch.argmax(probs_A.clone().detach(), dim=1)

        L_A = criterion(logits_A, label)
        personal_A.append(float(L_A.clone().detach()))
        
        probs_B = F.softmax(logits_B, dim=-1).to('cpu')
        pred_B = torch.argmax(probs_B, dim=1)
        L_B = criterion(logits_B, label)
        personal_B.append(float(L_B.clone().detach()))
        
        probs_C = F.softmax(logits_C, dim=-1).to('cpu')
        pred_C = torch.argmax(probs_C, dim=1)
        L_C = criterion(logits_C, label)
        personal_C.append(float(L_C.clone().detach()))

        ## Note: L_A, L_B, and L_C are all funtions of the parameters of each of A_e, B_e, C_e.
        
        ## As such, when the gradient of L_tot_A is calculated, for example, the parameters updated by its contribution
        ## dependent on L_B and L_C are specifically the learnable parameters of A_e. Thus, collab parameters can be thought of
        ## determining how much an individual model should be focusing on refining its initial embedding block,
        ## specifically towards the aim of improving the performance of the other models.
        
        L_tot_A = ( (1-alpha)*L_A + alpha*(L_B+L_C) )
        L_tot_B = ( (1-beta)*L_B + beta*(L_A+L_C) )
        L_tot_C = ( (1-gamma)*L_C + gamma*(L_A+L_B) )

        if contrastive:
            L_con_A = embed_loss(ind_A.clone(), [ind_B.clone(), ind_C.clone()])
            L_con_B = embed_loss(ind_B.clone(), [ind_A.clone(), ind_C.clone()])
            L_con_C = embed_loss(ind_C.clone(), [ind_A.clone(), ind_B.clone()])

            con_A.append(float(L_con_A.clone().detach().cpu()))
            con_B.append(float(L_con_B.clone().detach().cpu()))
            con_C.append(float(L_con_C.clone().detach().cpu()))
            
            L_tot_A = L_tot_A + (alpha**2)*L_con_A
            L_tot_B = L_tot_B + (beta**2)*L_con_B
            L_tot_C = L_tot_C + (gamma**2)*L_con_C

        if epoch >= uplift:
            opt_F.zero_grad()
            
            input_from_ABC = torch.cat([logits_A.clone(), logits_B.clone(), logits_C.clone()], dim=1).to(device)

            final_logits = readout(input_from_ABC, ind_concat.clone()).to('cpu')
            
            final_probs = F.softmax(final_logits.clone(), dim=-1).to('cpu')
            final_pred = torch.argmax(final_probs, dim=1)

            L_F = criterion(final_logits, label)
            personal_F.append(float(L_F.clone().detach()))

            L_sum = (eps+L_A.clone().detach() + L_B.clone().detach() + L_C.clone().detach())
            
            L_tot_F = L_F*( 1+torch.exp(eps-temp*(L_sum)) )
            tot_F.append(float(L_tot_F.clone().detach()))

            if lamb > 0.0:

                L_blame_A = lamb*( L_A.clone().detach()/L_sum )*L_F.clone()
                L_blame_B = lamb*( L_B.clone().detach()/L_sum )*L_F.clone()
                L_blame_C = lamb*( L_C.clone().detach()/L_sum )*L_F.clone()
                
                L_tot_A = L_tot_A + L_blame_A
                L_tot_B = L_tot_B + L_blame_B
                L_tot_C = L_tot_C + L_blame_C

                blame_A.append(float(L_blame_A.clone().detach()))
                blame_B.append(float(L_blame_B.clone().detach()))
                blame_C.append(float(L_blame_C.clone().detach()))

        tot_A.append(float(L_tot_A.clone().detach()))
        tot_B.append(float(L_tot_B.clone().detach()))
        tot_C.append(float(L_tot_C.clone().detach()))
        
        if epoch >= uplift: 
            L_tot_F.backward(retain_graph=True)
        L_tot_A.backward(retain_graph=True)
        L_tot_B.backward(retain_graph=True)
        L_tot_C.backward()

        if epoch >= uplift:
            opt_F.step()
            sched_F.step()
        opt_A.step()
        sched_A.step()
        opt_B.step()
        sched_B.step()
        opt_C.step()
        sched_C.step()
        
    report_A['total'] = np.mean(tot_A)
    report_A['personal'] = np.mean(personal_A)
    
    report_B['total'] = np.mean(tot_B)
    report_B['personal'] = np.mean(personal_B)
    
    report_C['total'] = np.mean(tot_C)
    report_C['personal'] = np.mean(personal_C)

    if contrastive:
        report_A['contrastive'] = np.mean(con_A)
        report_B['contrastive'] = np.mean(con_B)
        report_C['contrastive'] = np.mean(con_C)

    if epoch >= uplift:
        report_F['total'] = np.mean(tot_F)
        report_F['personal'] = np.mean(personal_F)
        if lamb > 0.0:
            report_A['blame'] = np.mean(blame_A)
            report_B['blame'] = np.mean(blame_B)
            report_C['blame'] = np.mean(blame_C)
        
    return report_A, report_B, report_C, report_F

def eval_one_epoch(eval_loader, models, readout, collab_params, temp, epoch, uplift=10, eps=1e-7, contrastive=False, phase='Validation', lamb=1.0):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = nn.CrossEntropyLoss()
    
    personal_A = []
    personal_B = []
    personal_C = []
    
    tot_A = []
    tot_B = []
    tot_C = []

    preds_A = []
    preds_B = []
    preds_C = []

    report_A = {}
    report_B = {}
    report_C = {}
    
    if contrastive:
        con_A = []
        con_B = []
        con_C = []

    if lamb > 0.0:
        blame_A = []
        blame_B = []
        blame_C = []
        
    model_A, model_B, model_C, = models
    
    model_A.eval()
    model_B.eval()
    model_C.eval()
    
    alpha, beta, gamma = collab_params
    
    personal_F = []
    tot_F = []
    report_F = {}
    preds_F = []
    readout.eval()
        
    all_labels = []
    
    pbar_eval = tqdm(eval_loader, total=len(eval_loader))
    pbar_eval.set_description(f'Epoch {epoch}: {phase}')
    with torch.no_grad():
        for img, label in pbar_eval:
            
            img = img.to(device)
            all_labels += list(label)
            
            ind_A = model_A.embed(img)
            ind_B = model_B.embed(img)
            ind_C = model_C.embed(img)
    
            ind_concat = torch.cat([ind_A, ind_B, ind_C], dim=1)
    
            logits_A = model_A(ind_concat).clone().cpu()
            logits_B = model_B(ind_concat).clone().cpu()
            logits_C = model_C(ind_concat).clone().cpu()
    
            probs_A = F.softmax(logits_A, dim=-1).to('cpu')
            pred_A = torch.argmax(probs_A, dim=1)
            L_A = criterion(logits_A, label)
            preds_A += list(pred_A)
            personal_A.append(float(L_A.clone().detach()))
            
            probs_B = F.softmax(logits_B, dim=-1).to('cpu')
            pred_B = torch.argmax(probs_B, dim=1)
            L_B = criterion(logits_B, label)
            preds_B += list(pred_B)
            personal_B.append(float(L_B.clone().detach()))
            
            probs_C = F.softmax(logits_C, dim=-1).to('cpu')
            pred_C = torch.argmax(probs_C, dim=1)
            L_C = criterion(logits_C, label)
            preds_C += list(pred_C)
            personal_C.append(float(L_C.clone().detach()))
    
            L_tot_A = ( (1-alpha)*L_A + alpha*(L_B+L_C) )
            L_tot_B = ( (1-beta)*L_B + beta*(L_A+L_C) )
            L_tot_C = ( (1-gamma)*L_C + gamma*(L_A+L_B) )

            if contrastive:
                L_con_A = embed_loss(ind_A.clone(), (ind_B.clone(), ind_C.clone()))
                L_con_B = embed_loss(ind_B.clone(), (ind_A.clone(), ind_C.clone()))
                L_con_C = embed_loss(ind_C.clone(), (ind_A.clone(), ind_B.clone()))
    
                con_A.append(float(L_con_A.clone().detach().cpu()))
                con_B.append(float(L_con_B.clone().detach().cpu()))
                con_C.append(float(L_con_C.clone().detach().cpu()))
                
                L_tot_A = L_tot_A + (alpha**2)*L_con_A
                L_tot_B = L_tot_B + (beta**2)*L_con_B
                L_tot_C = L_tot_C + (gamma**2)*L_con_C
            
            tot_A.append(float(L_tot_A.clone().detach()))
            tot_B.append(float(L_tot_B.clone().detach()))
            tot_C.append(float(L_tot_C.clone().detach()))
    
            if epoch >= uplift:
                L_sum = (eps+L_A.clone().detach() + L_B.clone().detach() + L_C.clone().detach())
                
                input_from_ABC = torch.cat( [logits_A.clone(), logits_B.clone(), logits_C.clone()], dim=1).to(device)
    
                final_logits = readout(input_from_ABC, ind_concat.clone()).cpu()
                final_probs = F.softmax(final_logits, dim=-1).to('cpu')
                final_pred = torch.argmax(final_probs, dim=1)
                preds_F += list(final_pred)
                
                L_F = criterion(final_logits, label)
                personal_F.append(float(L_F.clone().detach()))
                
                L_tot_F = L_F*( 1+torch.exp(eps-temp*(L_sum)) )
                tot_F.append(float(L_tot_F.clone().detach()))

                if lamb > 0.0:

                    L_blame_A = lamb*( L_A.clone().detach()/L_sum )*L_F.clone()
                    L_blame_B = lamb*( L_B.clone().detach()/L_sum )*L_F.clone()
                    L_blame_C = lamb*( L_C.clone().detach()/L_sum )*L_F.clone()
                    
                    L_tot_A = L_tot_A + L_blame_A
                    L_tot_B = L_tot_B + L_blame_B
                    L_tot_C = L_tot_C + L_blame_C
    
                    blame_A.append(float(L_blame_A.clone().detach()))
                    blame_B.append(float(L_blame_B.clone().detach()))
                    blame_C.append(float(L_blame_C.clone().detach()))

                
    acc_A = accuracy_score(all_labels, preds_A)
    acc_B = accuracy_score(all_labels, preds_B)
    acc_C = accuracy_score(all_labels, preds_C)

    if epoch >= uplift:
        acc_F = accuracy_score(all_labels, preds_F)
    else:
        acc_F = None

    report_A['total'] = np.mean(tot_A)
    report_A['personal'] = np.mean(personal_A)

    report_B['total'] = np.mean(tot_B)
    report_B['personal'] = np.mean(personal_B)

    report_C['total'] = np.mean(tot_C)
    report_C['personal'] = np.mean(personal_C)
    
    if contrastive:
        report_A['contrastive'] = np.mean(con_A)
        report_B['contrastive'] = np.mean(con_B)
        report_C['contrastive'] = np.mean(con_C)
        
    if epoch >= uplift:
        report_F['total'] = np.mean(tot_F)
        report_F['personal'] = np.mean(personal_F)
        report_F['accuracy'] = acc_F
        if lamb > 0.0:
            report_A['blame'] = np.mean(blame_A)
            report_B['blame'] = np.mean(blame_B)
            report_C['blame'] = np.mean(blame_C)

    report_A['accuracy'] = acc_A
    report_B['accuracy'] = acc_B
    report_C['accuracy'] = acc_C
    
    return report_A, report_B, report_C, report_F

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