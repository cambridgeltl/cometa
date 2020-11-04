import torch
import torch.nn.init
import torch.nn as nn
from torch.nn import functional as F

def cosine_sim(im, s):
    """Cosine similarity between all the image and sentence pairs
    """
    return im.mm(s.t())

def order_sim(im, s):
    """Order embeddings similarity measure max(0,s−im)
    """
    YmX = (s.unsqueeze(1).expand(s.size(0), im.size(0), s.size(1))
           - im.unsqueeze(0).expand(s.size(0), im.size(0), s.size(1)))
    score = -YmX.clamp(min=0).pow(2).sum(2).sqrt().t()
    return score

def l2norm(X):
    """L2-normalize columns of X
    """    
    norm = torch.pow(X, 2).sum(dim=1, keepdim=True).sqrt()
    a = norm.expand_as(X)    
    X = torch.div(X, a)    
    return X


class TripletLoss(nn.Module):

    def __init__(self, margin=0, measure=False, max_violation=False,\
        device=0):
        super(TripletLoss, self).__init__()
        self.margin = margin
        if measure == 'order':
            self.sim = order_sim
        else:
            self.sim = cosine_sim

        self.max_violation = max_violation
        self.device = device

    def forward(self, left, right):
        
        #l = F.normalize(left)
        #r = F.normalize(right)

        l = l2norm(left)
        r = l2norm(right)
        # compute prediction-target score matrix
        scores = self.sim(l, r)
        diagonal = scores.diag().view(l.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_r = (self.margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_l = (self.margin + scores - d2).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        I = mask.cuda(self.device)
        
        cost_r = cost_r.masked_fill_(I, 0)
        cost_l = cost_l.masked_fill_(I, 0)
        
        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_r = cost_r.max(1)[0]
            cost_l = cost_l.max(0)[0]

        #return cost_r.sum() + cost_l.sum()
        return cost_r.sum()
        
class NCA_loss(nn.Module):
    def __init__(self, alpha, beta, ep, device=0):
        super(NCA_loss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.ep = ep
        self.sim = cosine_sim
        self.device = device

    def forward(self, im, s):
        im = l2norm(im)
        s = l2norm(s)
        bsize = im.size()[0]
        # compute prediction-target score matrix
        scores = self.sim(im, s) #+ 1
        tmp  = torch.eye(bsize).cuda(self.device)
        s_diag = tmp * scores * self.beta
        scores_ori = scores - s_diag # clear diagnal

        alpha = self.alpha
        ep = self.ep
        S_ = torch.exp(alpha * (scores_ori-ep))
        
        loss = torch.sum(
                torch.log(1 + S_.sum(0)) / alpha \
                #+torch.log(1 + S_.sum(1)) / alpha \
                - torch.log(1 + F.relu(s_diag.sum(0)))
                ) / bsize
        return loss