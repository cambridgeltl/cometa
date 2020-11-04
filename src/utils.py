import torch

def compute_embeddings(data_loader, model, GPU_INDEX=0, ensemble=False):
    preds_, labels_ = None, None
    for c in data_loader:
        if ensemble:
            inputs1, inputs2, _, labels, indices = c
            out = model(inputs1.cuda(GPU_INDEX), inputs2.cuda(GPU_INDEX))
        else:
            inputs, _, labels, indices = c
            out = model(inputs.cuda(GPU_INDEX))
        
        labels = labels.cpu().detach().numpy().tolist()
        
        if preds_ is None:
            preds_, labels_ = out, labels
        else:
            preds_ = torch.cat([preds_, out])
            labels_ = labels_ + labels

    return preds_, labels_