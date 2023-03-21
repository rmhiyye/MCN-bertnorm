####################
# test prediction
####################

import torch
import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import cdist
import my_global

def tokenize(sentence):
    max_length = my_global.get_value('max_length')
    tokenizer = my_global.get_value('tokenizer')
    device = my_global.get_value('device')
    return tokenizer.encode(sentence, padding="max_length", max_length=max_length, truncation=True, add_special_tokens=True, return_tensors="pt").to(device) # Tokenize input into ids.

def inference(norm_list, basenorm, dd_test):
    print("Embedding ontology concept labels...")

    ######
    # Build labels/tags embeddings from ontology:
    ######
    device = my_global.get_value('device')
    max_length = my_global.get_value('max_length')
    tokenizer = my_global.get_value('tokenizer')
    model = my_global.get_value('model')
    embbed_size = my_global.get_value('embbed_size')
    cui_encode = dict()
    with torch.no_grad():
        for cui in norm_list:
            cui_encode[cui] = tokenizer.encode(cui, padding="max_length", max_length=max_length, truncation=True, add_special_tokens=True, return_tensors="pt")
            if embbed_size == None:
                embbed_size = len(cui_encode[cui][0])
    print("Number of concepts in ontology:", len(norm_list))
    print("Done.\n")

    ######
    # Build mention embeddings from testing set:
    ######
    X_pred = np.zeros((len(dd_test.keys()), embbed_size))
    with torch.no_grad():
        for i, id in tqdm(enumerate(dd_test.keys()), desc ='Building embeddings from cui list'):
            tokenized_mention = torch.tensor(tokenize(dd_test[id]['mention']).to(device))
            X_pred[i] = basenorm(model(tokenized_mention)[0][:,0]).cpu().detach().numpy()

    ######
    # Nearest neighbours calculation:
    ######
    dd_predictions = dict()
    for id in dd_test.keys():
        dd_predictions[id] = dict()
        dd_predictions[id]["pred_cui"] = []

    CUIVectorMatrix = np.zeros((len(norm_list), embbed_size)) # len(norm_list) x embbed_size
    i = 0
    for cui in cui_encode.keys():
        CUIVectorMatrix[i] = cui_encode[cui]
        i += 1

    print('\tDistance matrix calculation...')
    scoreMatrix = cdist(X_pred, CUIVectorMatrix, 'cosine')  # cdist() is an optimized algo to distance calculation.
    # (doc: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html)

    print("\tDone.")

    # For each mention, find back the nearest cui vector, then attribute the associated cui:
    i=0
    for i, id in enumerate(dd_test.keys()):
        minScore = min(scoreMatrix[i])
        j = -1
        stopSearch = False
        for cui in cui_encode.keys():
            if stopSearch == True:
                break
            j += 1
            if scoreMatrix[i][j] == minScore:
                dd_predictions[id]["pred_cui"] = [cui]
                stopSearch = True
                break
    del cui_encode

    return dd_predictions
