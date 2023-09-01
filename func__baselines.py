import numpy as np

from sklearn.metrics import classification_report

from sklearn.model_selection import KFold
import os
import pandas as pd
from stop_words import get_stop_words
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
import pprint
import json
from tqdm import tqdm
import random

from sklearn.naive_bayes import GaussianNB

from collections import Counter



TARGETS_ENCODING = {

    "Func0": 0,
    "Func1": 1


}

SEED = 2
N_FOLDS = 2

data_dir_path = "syntaxTree_parsed_ccs_laz_exl_fix_drop"

SAVE_DIR = "Func_classification_results"
os.makedirs(SAVE_DIR, exist_ok = True)

WS = [4]


def read_jsonl(path):
    output = []
    with open(path, "r") as f:
        for line in f.read().split("\n"):
            if line:
                output.append(json.loads(line))
    return output


sentences = {}
lfs_data = []

for dirname in tqdm(os.listdir(data_dir_path)):
  if dirname in TARGETS_ENCODING:

    dir_path = os.path.join(data_dir_path, dirname)

    for filename in os.listdir(dir_path):
      contents = read_jsonl(os.path.join(dir_path, filename))
      sentences[filename] = {}
      sentences[filename][dirname] = contents

      lfs_data.append(
          (filename, dirname, TARGETS_ENCODING[dirname])
          )



fold_split = KFold(n_splits=N_FOLDS, random_state=SEED, shuffle=True)




def load_lfs(ws, lf_names_labels):
    texts, labels = [], []

    for cc, label_name, label_id in lf_names_labels:
        contents = sentences[cc][label_name]

        for cont_dict in contents:
            start_from = min(cont_dict["start_cc_index"], cont_dict["end_cc_index"])
            end_from = max(cont_dict["start_cc_index"], cont_dict["end_cc_index"])

            contexts = cont_dict["lemmas"][max(0, start_from - ws):start_from]
            contexts += cont_dict["lemmas"][end_from + 1: end_from + 1 + ws]

            texts.append(" ".join(contexts))
            labels.append(label_id)

    return texts, labels



def resample(labels, drop_labels_probs_dict):
  random.seed(2)

  keep_idx = []
  for i, l in enumerate(labels):

    if l in drop_labels_probs_dict:
      coin = random.uniform(0,1)
      if coin <= drop_labels_probs_dict[l]:
        keep_idx.append(i)
    else:
      keep_idx.append(i)
  return keep_idx





resample_probs = {
    0:0.05
}
svm_cw =  "balanced"



for ws in WS:
  for i, (train_index, test_index) in enumerate(fold_split.split(lfs_data)):


    print("folder ", i, "ws ", ws)
    vect = TfidfVectorizer(
        stop_words=get_stop_words("spanish"),
        binary=True
    )

    train_lfs = [lfs_data[i] for i in train_index]
    test_lfs = [lfs_data[i] for i in test_index]

    train_texts, train_labels = load_lfs(ws, train_lfs)
    test_texts, test_labels = load_lfs(ws, test_lfs)

    keep_train_idx = resample(
        train_labels,
        resample_probs
    )
    print("Original train:", Counter(train_labels))
    train_texts = [train_texts[k] for k in keep_train_idx]
    train_labels = [train_labels[k] for k in keep_train_idx]
    print("Resample train:", Counter(train_labels))
    print("test:", Counter(test_labels))

    train_features = vect.fit_transform(train_texts)
    test_features = vect.transform(test_texts)
    for model_arch, model_params, model_name in [
        (SVC, {
            "verbose": True,
            "random_state": SEED,
            "class_weight": svm_cw,
            "kernel": "linear"
            }, "svm"),
        (GaussianNB, {}, "nb")
        ]:


        print(model_name)
        model = model_arch(**model_params)
        model.fit(train_features.toarray(), train_labels)

        pred_classes = model.predict(test_features.toarray())
        report = classification_report(
            y_true=test_labels,
            y_pred=pred_classes

        )
        json_report = classification_report(
            y_true=test_labels,
            y_pred=pred_classes,
            output_dict=True
        )
        print(report)
        df = pd.DataFrame(json_report).transpose()
        df.to_csv(os.path.join(SAVE_DIR, f"{model_name}_fold_{str(i)}.csv"))


