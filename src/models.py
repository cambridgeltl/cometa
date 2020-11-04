import torch
import torch.nn.init
import torch.nn as nn
from torch.nn import functional as F
import pandas as pd
import math
from torch.nn.utils.rnn import pad_packed_sequence as pad
from torch.nn.utils.rnn import PackedSequence


class SelfAttention(nn.Module):
    def __init__(self, hidden_size, batch_first=False):
        super(SelfAttention, self).__init__()

        self.hidden_size = hidden_size
        self.batch_first = batch_first

        self.att_weights = nn.Parameter(torch.Tensor(1, hidden_size),
                                        requires_grad=True)

        nn.init.xavier_uniform_(self.att_weights.data)

    def get_mask(self):
        pass

    def forward(self, inputs):

        if isinstance(inputs, PackedSequence):
            # unpack output
            inputs, lengths = pad(inputs, batch_first=self.batch_first)
        if self.batch_first:
            batch_size, max_len = inputs.size()[:2]
        else:
            max_len, batch_size = inputs.size()[:2]
            inputs = inputs.permute(1, 0, 2)

        # att = torch.mul(inputs, self.att_weights.expand_as(inputs))
        # att = att.sum(-1)
        weights = torch.bmm(inputs,
                            self.att_weights  # (1, hidden_size)
                            .permute(1, 0)  # (hidden_size, 1)
                            .unsqueeze(0)  # (1, hidden_size, 1)
                            # (batch_size, hidden_size, 1)
                            .repeat(batch_size, 1, 1)
                            )


        attentions = F.softmax(F.relu(weights.squeeze()), dim=-1)

        # apply weights
        weighted = torch.mul(
            inputs, attentions.unsqueeze(-1).expand_as(inputs))

        # get the final fixed vector representations of the sentences
        representations = weighted.sum(1).squeeze()

        return representations, attentions
        #return weighted, attentions

class multilevel_attention(nn.Module):
    def __init__(self, input_size=768, target_size=768, lin=False):
        #super(aligner, self).__init__()
        super().__init__()
        
        self.self_attn = SelfAttention(hidden_size=768, batch_first=True)
        self.lin = lin
        if self.lin is True:
            self.fc = nn.Linear(768, 768)

    def forward(self, x, dor=0.0):
        
        #x = self.norm(x.transpose(1,2))
        x = x.transpose(1,2)

        #x = F.dropout(x, dor)

        x, attn = self.self_attn(x)
        #print (x.shape, attn.shape)
        
        
        #print (x.shape)
        #weighted_x = self.attn(x, x, x)
        #x = x + weighted_x
        #print (weighted_x.shape)
        #print (torch.stack([x,weighted_x], -1).shape)
        #x = torch.max(torch.stack([x,weighted_x], -1), -1)[0]
        #print (x.shape)
        #x = weighted_x
        #x = x.transpose(1,2)

        #x = torch.mean(x, -1)
        #x = torch.max(x, -1)[0]
        if self.lin:
            x = self.fc(x) #.squeeze(-1)

        #x = F.normalize(x)
        
        return x

class fc_aligner(nn.Module):
    def __init__(self, input_size=768, target_size=768):
        super().__init__()
        
        self.fc1 = nn.Linear(input_size, target_size)

    def forward(self, x, dor=0.0):
        
        x = F.dropout(x, dor)
        #x = self.fc1(x)
        x = F.relu(self.fc1(x))
        
        return x

class mla_bert_ft_ensemble(nn.Module):
    def __init__(self, target_len=1068):
        super().__init__()
        
        self.mla_bert = multilevel_attention(768, 768, lin=False)
        self.fc = nn.Linear(300+768, target_len)
        self.fc_aligner = fc_aligner(300, 300)

    def forward(self, ft_x, bert_x, dor=0.0):
        #with torch.no_grad():
        bert_x = self.mla_bert(bert_x)
        ft_x = self.fc_aligner(ft_x)
        #bert_x = torch.mean(bert_x, -1)
        #print (bert_x.shape, ft_x.shape)

        x = torch.cat([ft_x, bert_x], dim=1)
        x = self.fc(x)
        #x = self.fc(x)

        return x

# build snomed surface->node_id dict
def build_surface_to_snomed_id(snomed):
    sf2id = {}
    for node_id in snomed.graph.nodes:
        sfs = snomed.index_definition[node_id]
        for sf in sfs:
            sf2id[sf.lower()] = int(node_id)
    return sf2id

# # build term->node_id dict from train set
# def build_train_dict(train_path, gran="specific"):
#     train_dict = {}
#     data_table = pd.read_csv(train_path, sep='\t', encoding='utf8')
#     for index in range(len(data_table)):
#         row = data_table.iloc[index]
#         term = row["Term"].lower()
#         if gran == "specific":
#             snomed_id = row["Specific SNOMED ID"]
#         if gran == "general":    
#             snomed_id = row["General SNOMED ID"]
#         train_dict[term] = snomed_id
#     return train_dict

def build_train_dict(train_path, gran="specific"):
    """
    Collect term-node_id mappings from train set.
    If there is a tie, use majority vote.
    """
    train_dict = {}
    data_table = pd.read_csv(train_path, sep='\t', encoding='utf8')
    stats = {}
    for index in range(len(data_table)):
        row = data_table.iloc[index]
        term = row["Term"].lower()
        if gran == "specific":
            snomed_id = row["Specific SNOMED ID"]
        if gran == "general":    
            snomed_id = row["General SNOMED ID"]
        #if term in train_dict.keys():
        if term in stats.keys():
            if snomed_id in stats[term].keys():
                stats[term][snomed_id] += 1
            else:
                stats[term][snomed_id] = 1
        else:
            stats[term] = {}
            stats[term][snomed_id] = 1
    for term in stats.keys():
        sorted_ids = sorted(stats[term], key=stats[term].get, reverse=True)
        train_dict[term] = sorted_ids[0]
         
    return train_dict