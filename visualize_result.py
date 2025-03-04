from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import numpy as np
import os
from folder_structure import FolderStructure
import argparse

arg = argparse.ArgumentParser()
arg.add_argument(
    "dataset_name",
    default="",
    help="The dataset name, please pick one from 20ng, R8, R52, ohsumed, mr")
args = arg.parse_args()

dataset = args.dataset_name

fs = FolderStructure(dataset)

with open(fs.get_train_index_file(), 'r') as f:
    lines = f.readlines()

train_size = len(lines)

with open(fs.get_shuffled_doc_names_file(), 'r') as f:
    lines = f.readlines()

target_names = set()
labels = []
for line in lines:
    line = line.strip()
    temp = line.split('\t')
    labels.append(temp[2])
    target_names.add(temp[2])

target_names = list(target_names)

f = open(fs.get_doc_vector_file(), 'r')
lines = f.readlines()
f.close()

docs = []
for line in lines:
    temp = line.strip().split()
    values_str_list = temp[1:]
    values = [float(x) for x in values_str_list]
    docs.append(values)

fea = docs[train_size:]  # int(train_size * 0.9)
label = labels[train_size:]  # int(train_size * 0.9)
label = np.array(label)

fea = TSNE(n_components=2).fit_transform(fea)
cls = np.unique(label)

# cls=range(10)
fea_num = [fea[label == i] for i in cls]
plt.figure(figsize=(20, 10))
for i, f in enumerate(fea_num):
    if cls[i] in range(10):
        plt.scatter(f[:, 0], f[:, 1], label=cls[i], marker='+')
    else:
        plt.scatter(f[:, 0], f[:, 1], label=cls[i])
plt.legend(ncol=5, loc='lower left', bbox_to_anchor=(0, 1.02), fontsize=11)
plt.savefig("result.png")
plt.show()
