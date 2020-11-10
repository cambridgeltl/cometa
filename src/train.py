import torch
import torch.nn.init
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.utils.clip_grad import clip_grad_norm_
import copy
from .evaluation import *

def val_or_test(text_model, dloader, search_space_embeddings, \
    epoch=0, typ="dev ", bsz=256, GPU_INDEX=0, ensemble=False):
    text_model.eval()
    accs = evalrank(dloader, text_model, search_space_embeddings, \
            topks=(1,10,100),bsz=bsz, GPU_INDEX=GPU_INDEX, ensemble=ensemble)
    acc_sum = sum(accs)
    print ("[epoch %d][%s] Acc@1: %.3f Acc@10: %.3f Acc@100: %.3f AccSum: %.3f" % \
           (epoch+1,typ,accs[0],accs[1],accs[2],acc_sum))
    return accs

# val before epoch 1
def train(text_model, train_params, optimizer, criterion, \
          train_loader, val_loader, valset, test_loader, testset, \
          num_epoch=20, grad_clip=2., dor=0.0, GPU_INDEX=0):

    val_or_test(text_model, val_loader, testset.search_space_embeddings, \
                epoch=-1, typ="dev ", bsz=500, GPU_INDEX=GPU_INDEX)

    # train
    val_max = 0.0
    best_sd = None
    for e in range(num_epoch):
        text_model.train()
        loss_sum = 0.
        for b in train_loader:
            inputs, targets, labels_, indices = b
            # filter out samples with identical labels
            labels, valid_indices = np.unique(labels_, return_index=True)
            if len(valid_indices) < 2: continue # escape from bad batch

            inputs = inputs[valid_indices]
            targets = targets[valid_indices]

            out = text_model(inputs.cuda(GPU_INDEX), dor=dor)
            loss = criterion(out.float(), targets.float().cuda(GPU_INDEX))
            if grad_clip > 0:
                clip_grad_norm_(train_params, grad_clip)
        
            loss.backward()
            optimizer.step()
            loss_sum = loss_sum + loss.item()
        
        
        #print ("[epoch %d] loss: %.2f" % (e+1, loss_sum))
        # val & test
        #if (e+1) % 5 == 0:
        accs = val_or_test(text_model, val_loader, testset.search_space_embeddings, \
                              epoch=e, typ="dev ", bsz=500, GPU_INDEX=GPU_INDEX)
        
        if sum(accs) > val_max: 
            val_max = sum(accs)
            best_sd = copy.deepcopy(text_model.state_dict())
            val_or_test(text_model, test_loader, testset.search_space_embeddings,\
                 epoch=e, typ="test", GPU_INDEX=GPU_INDEX)
            print ("[best epoch: %d]" % (e+1))
    return best_sd


# val before epoch 1
def train_joint(text_model, train_params, optimizer, criterion, \
          train_loader, val_loader, valset, test_loader, testset, \
          num_epoch=20, grad_clip=2., dor=0.0, GPU_INDEX=0):
    
    val_or_test(text_model, val_loader, testset.search_space_embeddings, \
                epoch=-1, typ="dev ", bsz=500, GPU_INDEX=GPU_INDEX, ensemble=True)

    # train
    val_max = 0.0
    best_sd = None
    for e in range(num_epoch):
        text_model.train()
        loss_sum = 0.

        #if (e+1) == 20:
        #    optimizer = torch.optim.AdamW(text_model.parameters(), lr=1e-5/5)

        for b in train_loader:
            inputs1, inputs2, targets, labels_, indices = b
            # filter out samples with identical labels
            labels, valid_indices = np.unique(labels_, return_index=True)
            if len(valid_indices) < 2: continue # escape from bad batch

            inputs1 = inputs1[valid_indices]
            inputs2 = inputs2[valid_indices]
            targets = targets[valid_indices]
            
        
            out = text_model(inputs1.cuda(GPU_INDEX), inputs2.cuda(GPU_INDEX), dor=dor)
            loss = criterion(out.float(), targets.float().cuda(GPU_INDEX))
            if grad_clip > 0:
                clip_grad_norm_(train_params, grad_clip)
        
            loss.backward()
            optimizer.step()
            loss_sum = loss_sum + loss.item()
        
        
        #print ("[epoch %d] loss: %.2f" % (e+1, loss_sum))
        # val & test
        #if (e+1) % 5 == 0:
        
        accs = val_or_test(text_model, val_loader, testset.search_space_embeddings, \
                              epoch=e, typ="dev ", bsz=500, GPU_INDEX=GPU_INDEX, ensemble=True)
        #accs = val_or_test(text_model, test_loader, testset.search_space_embeddings, \
        #    epoch=e, typ="test", bsz=500, GPU_INDEX=GPU_INDEX, ensemble=True)

        #if sum(accs) > val_max: 
        if accs[0] > val_max: 
            #val_max = sum(accs)
            val_max = accs[0]
            best_sd = copy.deepcopy(text_model.state_dict())
            val_or_test(text_model, test_loader, testset.search_space_embeddings,\
                 epoch=e, typ="test", GPU_INDEX=GPU_INDEX, ensemble=True)
            print ("[best epoch: %d]" % (e+1))
    return best_sd
