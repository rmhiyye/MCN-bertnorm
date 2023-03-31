####################
# 2019-n2c2-MCN dataset loader:
####################
import os
import torch
from torch.utils.data import DataLoader
import my_global

'''
Create empty file list dictionary
'''
def file_list_loader(file_list_path): # ./dataset/train/train_file_list.txt
    with open(f"{file_list_path}", "r") as fl:
        lines = fl.readlines()
    file_dict = dict()
    keys = []
    for line in lines:
        line = line.strip()
        keys.append(line)
    file_dict = dict.fromkeys(keys)
    return file_dict

def norm_list_loader(norm_list_path): # ./dataset/train/train_norm.txt
    with open(f"{norm_list_path}", "r") as fl:
        lines = fl.readlines()
    norm_list = []
    for line in lines:
        line = line.strip()
        norm_list.append(line)
    return norm_list

'''
file_dict:
    {"0034":
            {"N000":
                    {"cui": .. ,
                     "mention", ..},
             "N003":
             ...
             "text": '054478430 ELMVH\n79660638\n1835979\n12/11/2005..'},
     "0038":
     ...
     }
     
n_span_split:
    '2': norm_id || cui || start || end || start || end || start || end
    '1': norm_id || cui || start || end || start || end 
    '0': norm_id || cui || start || end 
'''
def mention2concept(note_path, norm_path, file_dict, with_text = True):
    n_span_split_keys = ["2", "1", "0"]
    n_span_split = {key: 0 for key in n_span_split_keys}
    cui_less_dict = dict()
    for key in file_dict.keys():
        filepath = os.getcwd() + "/dataset/train"
        sub_dict = dict()
        with open(f"{note_path}/{key}.txt", "r") as fl:
            texts = str(fl.read())
        with open(f"{norm_path}/{key}.norm", "r") as fl:
            lines = fl.readlines()
        for line in lines:
            subsub_dict = dict()
            line = line.strip()
            # norm_id, cui, start, end = line.split("||")
            line = line.split("||")
            norm_id = line[0]
            if line[1] != 'CUI-less':
                cui = line[1]
                line[2:] = [int(x) for x in line[2:]] # convert str into int
                span_split = (len(line)-2)/2 - 1 # (number_terms - number_id_and_cui)/2 - 1
                if span_split == 2:
                    n_span_split["2"] += 1
                    mention = texts[line[2]: line[3]] + texts[line[4]: line[5]] + texts[line[6]: line[7]]
                elif span_split == 1:
                    n_span_split["1"] += 1
                    mention = texts[line[2]: line[3]] + texts[line[4]: line[5]]
                elif span_split == 0:
                    n_span_split["0"] += 1
                    mention = texts[line[2]: line[3]]
                else:
                    raise ValueError("there are more than 2 span splits.")
                subsub_dict["cui"] = cui
                subsub_dict["mention"] = mention
                sub_dict[norm_id] = subsub_dict
            else:
                cui_less_dict[key] = 'CUI-less'
        if with_text:
            sub_dict["text"] = texts
        file_dict[key] = sub_dict
    return file_dict, cui_less_dict, n_span_split

def encoder(dataset, tokenizer):
    # Constructs two dictionnaries containing tokenized mentions (X) and associated labels (Y) respectively.
    max_length = my_global.get_value("max_length")
    X = dict()
    y = dict()
    for i, idx in enumerate(dataset.keys()):
        X[i] = tokenizer.encode(dataset[idx]['mention'], padding="max_length", max_length=max_length, truncation=True, add_special_tokens = True, return_tensors="pt")
        y[i] = tokenizer.encode(dataset[idx]['cui'], padding="max_length", max_length=max_length, truncation=True, add_special_tokens = True, return_tensors="pt")
    nbMentions = len(X.keys())
    print("Number of mentions:", nbMentions)
    return X, y

class Dataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.mention = X
        self.label = y

    def __getitem__(self, idx):
        mention = self.mention[idx]
        label = self.label[idx]
        sample = (mention, label)
        return sample

    def __len__(self):
        return len(self.mention)