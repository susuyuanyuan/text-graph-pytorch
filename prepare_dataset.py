from nltk.corpus import stopwords
import nltk
from nltk.corpus import wordnet as wn

import sys
import shutil

from utils import clean_str
from folder_structure import FolderStructure
import argparse

datasets = ['20ng', 'R8', 'R52', 'ohsumed', 'mr']

arg = argparse.ArgumentParser()
arg.add_argument(
    "dataset_name",
    default="",
    help="The dataset name, please pick one from 20ng, R8, R52, ohsumed, mr")
args = arg.parse_args()

dataset = args.dataset_name

if dataset not in datasets:
    sys.exit("wrong dataset name")

fs = FolderStructure(dataset)
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
print(stop_words)

shutil.copyfile(fs.get_original_doc_name_file(), fs.get_doc_name_file())

doc_content_list = []
with open(fs.get_original_doc_words_file(), 'rb') as f:
    for line in f.readlines():
        doc_content_list.append(line.strip().decode('latin1'))

word_freq = {}  # to remove rare words

for doc_content in doc_content_list:
    temp = clean_str(doc_content)
    words = temp.split()
    for word in words:
        if word in word_freq:
            word_freq[word] += 1
        else:
            word_freq[word] = 1

clean_docs = []
for doc_content in doc_content_list:
    temp = clean_str(doc_content)
    words = temp.split()
    doc_words = []
    for word in words:
        # word not in stop_words and word_freq[word] >= 5
        if dataset == 'mr':
            doc_words.append(word)
        elif word not in stop_words and word_freq[word] >= 5:
            doc_words.append(word)

    doc_str = ' '.join(doc_words).strip()
    clean_docs.append(doc_str)

with open(fs.get_clean_doc_words_file(), 'w') as f:
    f.write('\n'.join(clean_docs))

#dataset = '20ng'
min_len = 10000
aver_len = 0
max_len = 0

with open(fs.get_clean_doc_words_file(), 'r') as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        temp = line.split()
        aver_len = aver_len + len(temp)
        if len(temp) < min_len:
            min_len = len(temp)
        if len(temp) > max_len:
            max_len = len(temp)

aver_len = 1.0 * aver_len / len(lines)
print('Min_len : ' + str(min_len))
print('Max_len : ' + str(max_len))
print('Average_len : ' + str(aver_len))
