####################
# id_combination, lowercaser_mentions
####################

def id_combination(norm_dict):
    '''
    input:
        {"0034":
            {"N000":
                {"cui": .. ,
                 "mention", ..}
            }
        }
    output:
        {"0034_N000":
            {"cui": .. ,
             "mention", ..}
        }
    '''
    combin_dict = dict()
    for file_id in norm_dict.keys():
        for norm_id in norm_dict[file_id].keys():
            combin_id = file_id + "_" +norm_id
            combin_dict[combin_id] = norm_dict[file_id][norm_id]
    return combin_dict

def lowercaser_mentions(train_dict):
    for key in train_dict.keys():
        train_dict[key]["mention"] = train_dict[key]["mention"].lower()
    return train_dict