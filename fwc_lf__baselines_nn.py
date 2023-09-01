import numpy as np

from sklearn.metrics import classification_report

from sklearn.model_selection import KFold
import os
import pandas as pd
from stop_words import get_stop_words
import pprint
import json
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

import random
from collections import Counter
from sentence_transformers import SentenceTransformer
from collections import Counter



# init params and encodings
TARGETS_ENCODING = {
    "FWC": 0,
    "LF": 1
}


SEED = 2
N_FOLDS = 3

random.seed(SEED)


data_dir_path = "syntaxTree_parsed_ccs_laz_exl_fix_drop/"

SAVE_DIR = "FWC_LF_classification_results"
os.makedirs(SAVE_DIR, exist_ok = True)

DEVICE = "cuda"

embedding_model = SentenceTransformer('hiiamsid/sentence_similarity_spanish_es', device="cuda")
input_size = 768




def read_jsonl(path):
    output = []
    with open(path, "r") as f:
        for line in f.read().split("\n"):
            if line:
                output.append(json.loads(line))
    return output


# read data and save for each class
sentences = {}
lfs_data = []

fwc_path = os.path.join(data_dir_path, "FWC")
for filename in tqdm(os.listdir(fwc_path)):
    contents = read_jsonl(os.path.join(fwc_path, filename))
    sentences[filename] = {}
    sentences[filename]["FWC"] = contents
    lfs_data.append((filename, "FWC", TARGETS_ENCODING["FWC"]))


for dirname in tqdm(os.listdir(data_dir_path)):
    if dirname != "FWC" and dirname != "ERROR":
        dir_path = os.path.join(data_dir_path, dirname)
        for filename in os.listdir(dir_path):
            contents = read_jsonl(os.path.join(dir_path, filename))
            sentences[filename] = {}
            sentences[filename]["LF"] = contents
            lfs_data.append((filename, "LF", TARGETS_ENCODING["LF"]))




fold_split = KFold(n_splits=N_FOLDS, random_state=SEED, shuffle=True)



BATCH_SIZE = 128
LR = 1e-3
NUM_CLASSES = max(TARGETS_ENCODING.values()) + 1
NUM_ITERS = 3_500

class NNClassifier(nn.Module):
    def __init__(self, input_size):
        super(NNClassifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, NUM_CLASSES),
            nn.Softmax()
            )

    def forward(self, x):
        return self.fc(x)



def get_weights(labels):
    counts = Counter(labels)
    weights = [0 for x in range(len(labels))]
    for i, l in enumerate(labels):
        weights[i] = 1/(counts[l] + 1e-3)
    return weights


def generate_uniform_batch_indexes(labels, max_label, batch_size, weights=None):

    idx = random.choices(
            list(
                range(len(labels))), weights=weights, k=batch_size)
    return idx

    # Calculate the number of samples per class in a batch
    samples_per_class = batch_size // (max_label + 1)

    # Create a list of indexes with uniform distribution of labels
    batch_indexes = []
    for label in list(TARGETS_ENCODING.values()):
        
        label_indexes = random.choices(np.where(labels == label)[0], k=samples_per_class)
        batch_indexes.extend(label_indexes)

    shuffled_idx = sorted(batch_indexes, key=lambda x: random.uniform(0,1))
    return shuffled_idx



def divide_chunks(l, n=1028):
    # looping till length l
    for i in range(0, len(l), n):
        yield l[i:i + n]



class TrainerNN:
  @classmethod
  def train(cls, save_path, train_features, test_features, train_labels, test_labels, embedding_model, input_size, device):
    model = NNClassifier(input_size).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    weights = get_weights(train_labels)

    model.train()
    for i_iter in range(NUM_ITERS):
      batch_idx = generate_uniform_batch_indexes(train_labels, NUM_CLASSES, BATCH_SIZE, weights)
      batch_features = train_features[batch_idx]
      batch_features = embedding_model.encode(batch_features)
      batch_features = torch.Tensor(batch_features).to(device)

      batch_labels = torch.Tensor(train_labels[batch_idx]).long().to(device)
      optimizer.zero_grad()
      outputs = model(batch_features)
      loss = criterion(outputs, batch_labels)
      loss.backward()
      optimizer.step()

      print(f"Iter [{i_iter+1}/{NUM_ITERS}], Loss: {loss.item()}")

    model.eval()
    preds = []
    with torch.no_grad():
        for batch_idx in divide_chunks(list(range(len(test_labels)))):
          batch_features = test_features[batch_idx]
          batch_features = embedding_model.encode(batch_features)
          batch_features = torch.Tensor(batch_features).to(device)

          outputs = model(batch_features)
          outputs = outputs.detach().cpu()
          _, predicted = torch.max(outputs.data, 1)
          preds += predicted.numpy().tolist()

    report = classification_report(
              y_true=test_labels,
              y_pred=preds

          )
    json_report = classification_report(
            y_true=test_labels,
            y_pred=preds,
            output_dict=True
        )
    print(report)

    with open(save_path, "w") as f:
      json.dump({"report": report, "json_report": json_report}, f)



def load_lfs_texts(lf_names_labels):
    texts, labels = [], []

    for cc, label_name, label_id in lf_names_labels:
        contents = sentences[cc][label_name]

        for cont_dict in contents:
            end_from = max(cont_dict["start_cc_index"], cont_dict["end_cc_index"])
            tokens = cont_dict["tokens"].copy()
            tokens[cont_dict["start_cc_index"]] = "[MASK]"
            tokens = tokens[:cont_dict["end_cc_index"]] + ["|", cont_dict["tokens"][cont_dict["end_cc_index"]], "|"] + tokens[cont_dict["end_cc_index"]+1:] 
            texts.append(" ".join(tokens))
            labels.append(label_id)

    return texts, labels




for i, (train_index, test_index) in enumerate(fold_split.split(lfs_data)):

  print("fold ", i)
  train_lfs = [lfs_data[i] for i in train_index]
  test_lfs = [lfs_data[i] for i in test_index]

  train_texts, train_labels = load_lfs_texts(train_lfs)
  test_texts, test_labels = load_lfs_texts(test_lfs)


  train_texts = np.array(train_texts)
  test_texts = np.array(test_texts)

  train_labels = np.array(train_labels)
  test_labels = np.array(test_labels)


  save_path = os.path.join(SAVE_DIR, f"fold_{str(i)}.json")
  TrainerNN.train(save_path, train_texts, test_texts, train_labels, test_labels, embedding_model, input_size, DEVICE)
