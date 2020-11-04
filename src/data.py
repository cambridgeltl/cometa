
import torch.utils.data as data_
import torch
import pandas as pd
import pickle as pkl
import pickle
import numpy as np

class myDataset(data_.Dataset):
    def __init__(self, full_chv_path, chv_data_path, term_vec_path, snomed_vec_path, 
                 granularity="specific", load_target=False):
        """
        Args:
            transform: transformer for image.
        """

        self.gran = granularity
        # load chv data
        self.data_table = pd.read_csv(chv_data_path, sep='\t', encoding='utf8')
        
        # load target vecs
        self.snomed_vec_dict = pickle.load(open(snomed_vec_path,"rb"))
        #print ("SNOMED vec shape:",np.array(list(self.snomed_vec_dict.values())).shape)
        
        # load contextual term vec
        self.term_vec_dict = pickle.load(open(term_vec_path,"rb"))
        
        # all snomed ids used
        ful_path = full_chv_path
        self.data_table_full = pd.read_csv(ful_path, sep='\t', encoding='utf8')
        self.search_space_ids = list(self.snomed_vec_dict.keys())
        
        self.snomed_id_to_label = {}
        self.label_to_snomed_id = {}
        self.search_space_embeddings = []
        for i,k in enumerate(self.search_space_ids):
            self.snomed_id_to_label[k] = i
            self.label_to_snomed_id[i] = k
            if load_target:
                self.search_space_embeddings.append(self.snomed_vec_dict[k])
        if load_target:
            self.search_space_embeddings = np.array(self.search_space_embeddings)
            print ("[target embeddings loaded, search space size: {}]".format(len(self.search_space_ids)))

    def __getitem__(self, index):
        """This function returns a tuple that is further passed to collate_fn
        """

        row = self.data_table.iloc[index]
        term_vec = self.term_vec_dict[row["ID"]]

        if self.gran == "specific":
            snomed_id = row["Specific SNOMED ID"]
        if self.gran == "general":
            snomed_id = row["General SNOMED ID"]

        target_vec = self.snomed_vec_dict[snomed_id]
        label = self.snomed_id_to_label[snomed_id]
        
        return torch.tensor(term_vec), torch.tensor(target_vec), label, index


    def __len__(self):
        return len(self.data_table)

class myDataset_mla(data_.Dataset):
    def __init__(self, full_chv_path, chv_data_path, term_vec_path1, term_vec_path2, 
                snomed_vec_path, granularity="specific", load_target=False):
        """
        Args:
            transform: transformer for image.
        """

        self.gran = granularity
        # load chv data
        self.data_table = pd.read_csv(chv_data_path, sep='\t', encoding='utf8')
        
        # load target vecs
        self.snomed_vec_dict = pickle.load(open(snomed_vec_path,"rb"))
        #print ("SNOMED vec shape:",np.array(list(self.snomed_vec_dict.values())).shape)
        
        # load contextual term vec
        self.term_vec_dict1 = pickle.load(open(term_vec_path1,"rb"))
        self.term_vec_dict2 = pickle.load(open(term_vec_path2,"rb"))
        
        # all snomed ids used
        ful_path = full_chv_path
        self.data_table_full = pd.read_csv(ful_path, sep='\t', encoding='utf8')
        self.search_space_ids = list(self.snomed_vec_dict.keys())
        
        self.snomed_id_to_label = {}
        self.label_to_snomed_id = {}
        self.search_space_embeddings = []
        for i,k in enumerate(self.search_space_ids):
            self.snomed_id_to_label[k] = i
            self.label_to_snomed_id[i] = k
            if load_target:
                self.search_space_embeddings.append(self.snomed_vec_dict[k])
        if load_target:
            self.search_space_embeddings = np.array(self.search_space_embeddings)
            print ("[target embeddings loaded, search space size: {}]".format(len(self.search_space_ids)))

    def __getitem__(self, index):
        """This function returns a tuple that is further passed to collate_fn
        """

        row = self.data_table.iloc[index]
        term_vec1 = self.term_vec_dict1[row["ID"]]
        term_vec2 = self.term_vec_dict2[row["ID"]]

        if self.gran == "specific":
            snomed_id = row["Specific SNOMED ID"]
        if self.gran == "general":
            snomed_id = row["General SNOMED ID"]
        
        target_vec = self.snomed_vec_dict[snomed_id]
        label = self.snomed_id_to_label[snomed_id]
        
        return torch.tensor(term_vec1), torch.tensor(term_vec2), torch.tensor(target_vec), label, index


    def __len__(self):
        return len(self.data_table)


def get_loader_single(full_chv_path, chv_data_path, term_vec_path, snomed_vec_path, gran="specific", \
                      batch_size=128, shuffle=True, \
                      num_workers=10, load_target=False):

    dataset = myDataset(full_chv_path, chv_data_path, term_vec_path, snomed_vec_path, \
                        granularity=gran, load_target=load_target)

    # It crashes when using CPU-only and pin_memory 
    pin_memory = True
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              pin_memory=pin_memory
                                              )
                                              #, collate_fn=collate_fn)
    return data_loader, dataset

def get_loader_mla(full_chv_path, chv_data_path, term_vec_path1, term_vec_path2, snomed_vec_path, gran="specific", \
                      batch_size=128, shuffle=True, \
                      num_workers=10, load_target=False):

    dataset = myDataset_mla(full_chv_path, chv_data_path, term_vec_path1, term_vec_path2, snomed_vec_path, \
                        granularity=gran, load_target=load_target)

    # It crashes when using CPU-only and pin_memory 
    pin_memory = True
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              pin_memory=pin_memory
                                              )
                                              #, collate_fn=collate_fn)
    return data_loader, dataset

