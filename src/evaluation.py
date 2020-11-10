import torch
import torch.nn.init
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import faiss
from tqdm.notebook import tqdm
from multiprocessing import Pool
import os
import sys
import pathlib
import networkx as nx
import copy

from .data import *
from .utils import *

from .Snomed import Snomed

SNOMED_PATH = '../../data/SnomedCT_201907/'
snomed = Snomed(SNOMED_PATH, taxonomy=False)
snomed.load_snomed()

""" Acc@K"""

def compute_nn_accuracy(x_src_, x_labels_, tgt_faiss_index, topks=(1,10,100), bsz=256):
    lexicon_size = x_src_.shape[0]
    accs = [0.0]*len(topks)
    
    for si in tqdm(np.arange(0,lexicon_size,bsz),disable=True):
        if si+bsz <= lexicon_size:
            x_src = x_src_[si: si+bsz]
            x_labels = x_labels_[si: si+bsz]
        else:
            x_src = x_src_[si: lexicon_size]
            x_labels = x_labels_[si: lexicon_size]
        
        #x_src /= np.linalg.norm(x_src, axis=1)[:, np.newaxis] + 1e-8
        #tgt /= np.linalg.norm(tgt, axis=1)[:, np.newaxis] + 1e-8
        #scores = np.dot(x_src, tgt.T)
        
        #scores = scipy.spatial.distance.cdist(x_src, tgt, metric='cosine')
        
        #_, rank_matrix = tgt_faiss_index.search(x_src_, 350834)
        _, rank_matrix = tgt_faiss_index.search(x_src.astype('float32'), topks[-1])
        
        for j in range(x_src.shape[0]):
            #ranks = scores[j, :].argsort()
            ranks = rank_matrix[j]
            #rank_index = np.where(ranks == x_labels[j])[0][0]
            #MRR = MRR + 1/(rank_index+1)
            for k in range(len(topks)):
                if x_labels[j] in ranks[:topks[k]]:
                    accs[k] = accs[k] + 1
        
    return [acc / lexicon_size for acc in accs]


def evalrank(data_loader, model, tgt, topks=(1,10,100), dor=0.0, \
    bsz=256, GPU_INDEX=0, ensemble=False):

    tgt /= np.linalg.norm(tgt, axis=1)[:, np.newaxis] + 1e-8
    tgt_faiss_index = faiss.IndexFlatIP(tgt.shape[1])   # build the index
    tgt_faiss_index = faiss.index_cpu_to_all_gpus(tgt_faiss_index) # multiple GPU index
    #res = faiss.StandardGpuResources()  # use a single GPU
    #tgt_faiss_index = faiss.index_cpu_to_gpu(res, 0, tgt_faiss_index)# move to GPU
    tgt_faiss_index.add(tgt.astype('float32')) # add vectors to the index
    
    preds_, labels_ = compute_embeddings(data_loader, model, GPU_INDEX=GPU_INDEX, ensemble=ensemble)
    
    preds_ = preds_.cpu().detach().numpy()   
    preds_ /= np.linalg.norm(preds_, axis=1)[:, np.newaxis] + 1e-8
    return compute_nn_accuracy(preds_, labels_, tgt_faiss_index, topks=topks, bsz=bsz)


""" MRR """

def multi_run_wrapper(args):
    return tiny_computer(*args)

def tiny_computer(ranks, x_labels):
    MRR = 0.0
    #scores = scipy.spatial.distance.cdist(x_src, tgt, metric='cosine')
    
    for j in range(ranks.shape[0]):
        #ranks = scores[j, :].argsort()
        rank_index = np.where(ranks[j] == x_labels[j])[0][0]
        #print (rank_index)
        MRR = MRR + 1/(rank_index+1)    
    print ("[process done]")
    return MRR

def compute_MRR(x_src_, x_labels_, tgt, bsz=256):
    lexicon_size = x_src_.shape[0]

    MRR = 0.0
    
    pool = Pool(os.cpu_count()-1)   
    args = []
    
    for si in tqdm(np.arange(0,lexicon_size,bsz)):
        if si+bsz <= lexicon_size:
            x_src = x_src_[si: si+bsz]
            x_labels = x_labels_[si: si+bsz]
        else:
            x_src = x_src_[si: lexicon_size]
            x_labels = x_labels_[si: lexicon_size]
        #scores = 1-F.cosine_similarity(x_src, tgt, dim=1)
        scores = 1-torch.mm(x_src, tgt.transpose(0,1))
        ranks = torch.argsort(scores, dim=1)
        ranks = ranks.cpu().detach().numpy()
        args.append((ranks, x_labels)) # for multip
        
    res = pool.map(multi_run_wrapper, args) # for multip
    MRR = sum(res) # for multip

    return MRR / lexicon_size



def compute_MRR_dumb(x_src_, x_labels_, tgt, bsz=256):
    """
    assuming x_src_ and tgt are torch.tensor and already normalized.
    """
    lexicon_size = x_src_.shape[0]

    MRR = 0.0
    
    for si in tqdm(np.arange(0,lexicon_size,bsz)):
        if si+bsz <= lexicon_size:
            x_src = x_src_[si: si+bsz]
            x_labels = x_labels_[si: si+bsz]
        else:
            x_src = x_src_[si: lexicon_size]
            x_labels = x_labels_[si: lexicon_size]
        
        #scores = scipy.spatial.distance.cdist(x_src, tgt, metric='cosine')
        #scores = 1-F.cosine_similarity(x_src, tgt, dim=1)
        scores = 1-torch.mm(x_src, tgt.transpose(0,1))
        ranks = torch.argsort(scores, dim=1)
        ranks = ranks.cpu().detach().numpy()
        for j in range(x_src.shape[0]):
            #ranks = scores[j, :].argsort()
            rank_index = np.where(ranks[j] == x_labels[j])[0][0]
            #print (rank_index)
            MRR = MRR + 1/(rank_index+1)   
        
    return MRR / lexicon_size

def evalMRR(data_loader, model, tgt, bsz=256, \
     dumb=True, GPU_INDEX=0, ensemble=False):
    
    preds_, labels_ = compute_embeddings(data_loader, model, GPU_INDEX=GPU_INDEX, ensemble=ensemble)
    
    if dumb:
        tgt = torch.Tensor(tgt).to(GPU_INDEX)
        preds_ = F.normalize(preds_) #preds_ / preds_.norm(dim=1)[:, None]
        tgt = F.normalize(tgt) #tgt / tgt.norm(dim=1)[:, None]
        return compute_MRR_dumb(preds_, labels_, tgt, bsz=bsz)
    
    # if not dumb (parallel)
    preds_ = preds_.cpu().detach().numpy()
    preds_ /= np.linalg.norm(preds_, axis=1)[:, np.newaxis] + 1e-8
    tgt /= np.linalg.norm(tgt, axis=1)[:, np.newaxis] + 1e-8
    res = compute_MRR(preds_, labels_, tgt, bsz=bsz)
    del tgt
    del preds_
    return res

""" GD """
def multi_run_wrapper2(args):
    return tiny_computerGD(*args)

def tiny_computerGD(x_src, rank_matrix, x_labels, search_space_ids):
    mGD_sum, mGD_k_sum = 0.0,0.0
    for j in range(x_src.shape[0]):
        ranks = rank_matrix[j]
        gt_snomed_id = search_space_ids[x_labels[j]]
        # look for these top ranked labels
        top1_snomed_id = search_space_ids[ranks[0]]
        mGD_sum = mGD_sum + 1./(1.+snomed.distance(top1_snomed_id, gt_snomed_id))
        #print (len(ranks))
        for r in ranks:
            top_snomed_id = search_space_ids[r]
            mGD_k_sum = mGD_k_sum + 1./(1.+snomed.distance(top_snomed_id, gt_snomed_id))
    return mGD_sum, mGD_k_sum

def compute_GD(x_src_, x_labels_, tgt_faiss_index, dataset, \
                        topk=10, bsz=256):
    lexicon_size = x_src_.shape[0]
    
    mGD_sum,mGD_k_sum = 0.0,0.0
    
    args = []
    for si in tqdm(np.arange(0,lexicon_size,bsz)):
        if si+bsz <= lexicon_size:
            x_src = x_src_[si: si+bsz]
            x_labels = x_labels_[si: si+bsz]
        else:
            x_src = x_src_[si: lexicon_size]
            x_labels = x_labels_[si: lexicon_size]
        
        _, rank_matrix = tgt_faiss_index.search(x_src.astype('float32'), topk)
        
        args.append((x_src, rank_matrix, x_labels, dataset.search_space_ids)) # for multip
    
    pool = Pool(os.cpu_count()-1)   
    res = pool.map(multi_run_wrapper2, args) # for multip
    #print (res)
    mGD_sum = sum([p[0] for p in res])
    mGD_k_sum = sum([p[1] for p in res])

    return mGD_sum / lexicon_size, mGD_k_sum /lexicon_size/topk

def compute_GD_dumb(x_src_, x_labels_, tgt, dataset, topk=10,  bsz=256):
    
    lexicon_size = x_src_.shape[0]

    mGD_sum,mGD_k_sum = 0.0,0.0
    
    for si in tqdm(np.arange(0,lexicon_size,bsz)):
        if si+bsz <= lexicon_size:
            x_src = x_src_[si: si+bsz]
            x_labels = x_labels_[si: si+bsz]
        else:
            x_src = x_src_[si: lexicon_size]
            x_labels = x_labels_[si: lexicon_size]
        
        scores = 1-torch.mm(x_src, tgt.transpose(0,1))
        ranks = torch.argsort(scores, dim=1)
        ranks = ranks.cpu().detach().numpy()
        for j in range(x_src.shape[0]):
            gt_snomed_id = dataset.search_space_ids[x_labels[j]]
            # look for these top ranked labels
            rank_index = np.where(ranks[j] == x_labels[j])[0][0]
            top1_snomed_id = dataset.search_space_ids[ranks[j][0]]
            mGD_sum = mGD_sum + 1./(1.+snomed.distance(top1_snomed_id, gt_snomed_id))
            #print (len(ranks))
            for r in ranks[j][:topk]:
                top_snomed_id = dataset.search_space_ids[r]
                mGD_k_sum = mGD_k_sum + 1./(1.+snomed.distance(top_snomed_id, gt_snomed_id))

    return mGD_sum / lexicon_size, mGD_k_sum /lexicon_size/topk

def evalGD(data_loader, dataset, model, topk=10, \
    dor=0.0, bsz=256, dumb=False, GPU_INDEX=0, ensemble=False):
    tgt = dataset.search_space_embeddings
    
    if not dumb:
        tgt /= np.linalg.norm(tgt, axis=1)[:, np.newaxis] + 1e-8
        tgt_faiss_index = faiss.IndexFlatIP(tgt.shape[1])   # build the index
        tgt_faiss_index = faiss.index_cpu_to_all_gpus(tgt_faiss_index) # multiple GPU index
        #res = faiss.StandardGpuResources()  # use a single GPU
        #tgt_faiss_index = faiss.index_cpu_to_gpu(res, 0, tgt_faiss_index)# move to GPU
        tgt_faiss_index.add(tgt.astype('float32')) # add vectors to the index
    
    preds_, labels_ = compute_embeddings(data_loader, model, GPU_INDEX=GPU_INDEX, ensemble=ensemble)


    if dumb:
        tgt = torch.Tensor(tgt).to(GPU_INDEX)
        preds_ = F.normalize(preds_)
        tgt = F.normalize(tgt)
        return compute_GD_dumb(preds_, labels_, tgt, dataset, topk=topk, bsz=bsz)
    # if not dumb
    preds_ = preds_.cpu().detach().numpy()
    preds_ /= np.linalg.norm(preds_, axis=1)[:, np.newaxis] + 1e-8
    return compute_GD(preds_, labels_, tgt_faiss_index, dataset, topk=topk, bsz=bsz)

""" term monitor """

def term_monitor(full_chv_path, chv_track_path, term_vec_path, snomed_vec_path,\
                 model, knn=10, gran="general", GPU_INDEX=0):
    """
    given a sample, this function retrieves 
    and displays its topK candidate terms 
    in the search space.
    """
    data_loader, dataset = get_loader_single(full_chv_path, chv_track_path, term_vec_path, snomed_vec_path, \
                      batch_size=64, shuffle=False, num_workers=10, gran=gran, load_target=True)
    
    
    preds_, labels_ = compute_embeddings(data_loader, model, GPU_INDEX=GPU_INDEX)
    
    acc = 0.0
    #preds_ /= np.linalg.norm(preds_, axis=1)[:, np.newaxis] + 1e-8
    #tgt /= np.linalg.norm(tgt, axis=1)[:, np.newaxis] + 1e-8
    tgt = torch.Tensor(dataset.search_space_embeddings).cuda(GPU_INDEX)
    preds_ = F.normalize(preds_)
    tgt = F.normalize(tgt)

    #scores = np.dot(preds_, tgt.T)
    scores = 1-torch.mm(preds_, tgt.transpose(0,1))
    sorted_ranks = torch.argsort(scores, dim=1)
    sorted_ranks = sorted_ranks.cpu().detach().numpy()

    snmd_id2label = dataset.snomed_id_to_label
    label2snmd_id = {}
    for k,v in snmd_id2label.items():
        label2snmd_id[v] = k 
    
    terms = dataset.data_table["Term"].tolist()
    contexts = dataset.data_table["Example"].tolist()
    for j in range(preds_.shape[0]):
        ranklist = sorted_ranks[j]
        topk = sorted_ranks[j][:knn]
        # find hit
        hitat = np.where(ranklist == labels_[j])[0][0] + 1
        candidate_terms = [snomed[label2snmd_id[t]]['desc'] for t in topk]
        print ("** term **")
        print (terms[j])
        print ("\n")
        print ("** ground truth **")
        print (snomed[label2snmd_id[labels_[j]]]['desc'])
        print ("\n")
        print ("** context **")
        print (contexts[j])
        print ("\n")
        print ("** top %d candidates **" % knn)
        for ct in candidate_terms: print (ct)
        print ("\n")
        print ("** Hit at **")
        print (hitat)
        print ("\n")
        print ("---------------------------------------------------")
        


""" back-off metrics """

def compute_nn_accuracy_backoff(x_src_, x_labels_, ids_, dataset, tgt_faiss_index, prev_acc, num_solved, \
    topks=(1,10,100), lexicon_size=-1, bsz=256):

    if lexicon_size < 0:
        lexicon_size = x_src_.shape[0]
    accs = [0.0]*len(topks)
    prediction_dict = {}
    for si in np.arange(0,lexicon_size, bsz):
        if si+bsz <= lexicon_size:
            x_src = x_src_[si: si+bsz]
            x_labels = x_labels_[si: si+bsz]
            ids = ids_[si: si+bsz]
        else:
            x_src = x_src_[si: lexicon_size]
            x_labels = x_labels_[si: lexicon_size]
            ids = ids_[si: lexicon_size]

        _, rank_matrix = tgt_faiss_index.search(x_src.astype('float32'), topks[-1])
        
        for j in range(x_src.shape[0]):
            ranks = rank_matrix[j]
            predicted_id = dataset.label_to_snomed_id[ranks[0]]
            #gt_id = dataset.label_to_snomed_id[x_labels[j]]
            prediction_dict[ids[j]] = predicted_id

            for k in range(len(topks)):
                if x_labels[j] in ranks[:topks[k]]:
                    accs[k] = accs[k] + 1
    return [(acc+prev_acc)/(lexicon_size+num_solved) for acc in accs], prediction_dict


def compute_MRR_backoff(x_src_, x_labels_, tgt, num_solved, bsz=256):

    lexicon_size = x_src_.shape[0]

    MRR = 0.0
        
    for si in np.arange(0,lexicon_size,bsz):
        if si+bsz <= lexicon_size:
            x_src = x_src_[si: si+bsz]
            x_labels = x_labels_[si: si+bsz]
        else:
            x_src = x_src_[si: lexicon_size]
            x_labels = x_labels_[si: lexicon_size]
        
        scores = 1-torch.mm(x_src, tgt.transpose(0,1))
        ranks = torch.argsort(scores, dim=1)
        ranks = ranks.cpu().detach().numpy()
        for j in range(x_src.shape[0]):
            rank_index = np.where(ranks[j] == x_labels[j])[0][0]
            MRR = MRR + 1/(rank_index+1)   
        
    return (MRR+num_solved) / (lexicon_size+num_solved)

def compute_GD_backoff(x_src_, x_labels_, tgt_faiss_index, dataset, num_solved, \
    topk=10, bsz=256):

    lexicon_size = x_src_.shape[0]
    mGD_sum,mGD_k_sum = 0.0,0.0
    
    args = []
    for si in np.arange(0,lexicon_size,bsz):
        if si+bsz <= lexicon_size:
            x_src = x_src_[si: si+bsz]
            x_labels = x_labels_[si: si+bsz]
        else:
            x_src = x_src_[si: lexicon_size]
            x_labels = x_labels_[si: lexicon_size]
        
        _, rank_matrix = tgt_faiss_index.search(x_src.astype('float32'), topk)
        
        args.append((x_src, rank_matrix, x_labels, dataset.search_space_ids)) # for multip
    
    pool = Pool(os.cpu_count()-1)   
    res = pool.map(multi_run_wrapper2, args) # for multip
    #print (res)
    mGD_sum = sum([p[0] for p in res])
    mGD_k_sum = sum([p[1] for p in res])

    return (mGD_sum+num_solved) / (lexicon_size+num_solved), \
        (mGD_k_sum+num_solved*10) / (lexicon_size+num_solved) / topk


def compute_metrics_backoff(data_loader, dataset, model, sf2id=None, train_dict=None, \
    ed_dict=None, topks=(1,10,100), bsz=256, GPU_INDEX=0, ensemble=False):
    
    preds_, labels_ = compute_embeddings(data_loader, model,GPU_INDEX=GPU_INDEX, ensemble=ensemble)
    
    # apply string matching & dictionary
    to_be_acessed = []
    acc = 0.0
    prediction_dict = {}
    terms = dataset.data_table["Term"].tolist()
    ids = np.array(dataset.data_table["ID"].tolist())
    for i in range(len(dataset)):
        gt_id = dataset.label_to_snomed_id[labels_[i]]
        term = terms[i].lower()
        sample_id = ids[i]
        #print (term)
        if train_dict is not None:
            if term not in train_dict.keys():
                pass
            else:
                prediction_dict[sample_id] = train_dict[term]
                if gt_id == train_dict[term]:
                    acc = acc + 1
                continue
        
        if sf2id is not None:
            if term not in sf2id.keys():
                pass
            else:
                prediction_dict[sample_id] = sf2id[term]
                if gt_id == sf2id[term]:
                    acc = acc + 1
                continue
        
        if ed_dict is not None:
            #if len(term) > 4 and ed_dict[i][0][1] < 0.07:
            #if i not in ed_dict.keys(): continue
            #print (ed_dict[i][0][1])
            #ed_dict[i]
            #if (len(term)-term.count(" ")) > 5  and ed_dict[i][0][1][0] < 0.07:
            #if True:
            if ed_dict[i][0][1][1] < .16:
                pred = sf2id[ed_dict[i][0][0]]
                prediction_dict[sample_id] = pred
                if gt_id == pred:
                    acc = acc + 1
                continue

        to_be_acessed.append(i)
    
    ids_ = ids[to_be_acessed]
    preds_ = preds_[to_be_acessed]
    labels_ = np.array(labels_)[to_be_acessed]
    num_solved = float(len(dataset) - len(to_be_acessed))
    print ("ratio attempted by heuristics: %.1f%%" % (100 * num_solved / len(dataset)))

    print (acc/len(dataset))
    
    # process targets & predictions
    preds_torch = preds_
    preds_cpu = preds_.cpu().detach().numpy()   
    preds_cpu /= np.linalg.norm(preds_cpu, axis=1)[:, np.newaxis] + 1e-8

    tgt = dataset.search_space_embeddings
    tgt_torch = torch.Tensor(tgt).cuda(GPU_INDEX)
    tgt /= np.linalg.norm(tgt, axis=1)[:, np.newaxis] + 1e-8
    
    tgt_faiss_index = faiss.IndexFlatIP(tgt.shape[1])   # build the index
    tgt_faiss_index = faiss.index_cpu_to_all_gpus(tgt_faiss_index) # multiple GPU index
    tgt_faiss_index.add(tgt.astype('float32')) # add vectors to the index


    tgt_torch = F.normalize(tgt_torch)
    preds_torch = F.normalize(preds_torch)
    
    accs, prediction_dict_neural = compute_nn_accuracy_backoff(preds_cpu, labels_, ids_, dataset, tgt_faiss_index, acc, num_solved, topks=topks, bsz=bsz)
    prediction_dict.update(prediction_dict_neural)
    #mrgd, mrgdk = compute_GD_backoff(preds_cpu, labels_, tgt_faiss_index, dataset, num_solved, topk=10, bsz=bsz)
    #mrr = compute_MRR_backoff(preds_torch, labels_, tgt_torch, num_solved, bsz=bsz)

    return accs, prediction_dict #, mrr, (mrgd, mrgdk) # other metric turned off for back-off models
    



