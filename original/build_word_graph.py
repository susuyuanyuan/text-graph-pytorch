import os
import random
import numpy as np
import pickle as pkl
# import networkx as nx
import scipy.sparse as sp
from word_graph_builder import WordGraphBuilder
from math import log

import sys


def read_and_shuffle_dataset(dataset):
    print("Reading data")

    doc_name_list = []
    doc_train_list = []
    doc_test_list = []

    with open('./data/' + dataset + '.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            doc_name_list.append(line.strip())
            temp = line.split("\t")
            if temp[1].find('test') != -1:
                doc_test_list.append(line.strip())
            elif temp[1].find('train') != -1:
                doc_train_list.append(line.strip())

    doc_content_list = []
    with open('./data/corpus/' + dataset + '.clean.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            doc_content_list.append(line.strip())

    train_ids = []
    for train_name in doc_train_list:
        train_id = doc_name_list.index(train_name)
        train_ids.append(train_id)
    random.shuffle(train_ids)

    # uncomment this to get partial labeled data
    # train_ids = train_ids[:int(0.2 * len(train_ids))]

    train_ids_str = '\n'.join(str(index) for index in train_ids)
    with open('./data/' + dataset + '.train.index', 'w') as f:
        f.write(train_ids_str)

    test_ids = []
    for test_name in doc_test_list:
        test_id = doc_name_list.index(test_name)
        test_ids.append(test_id)
    random.shuffle(test_ids)

    test_ids_str = '\n'.join(str(index) for index in test_ids)
    with open('./data/' + dataset + '.test.index', 'w') as f:
        f.write(test_ids_str)

    ids = train_ids + test_ids

    shuffle_doc_name_list = []
    shuffle_doc_words_list = []
    for id in ids:
        shuffle_doc_name_list.append(doc_name_list[int(id)])
        shuffle_doc_words_list.append(doc_content_list[int(id)])

    with open('./data/' + dataset + '_shuffle.txt', 'w') as f:
        f.write('\n'.join(shuffle_doc_name_list))

    with open('./data/corpus/' + dataset + '_shuffle.txt', 'w') as f:
        f.write('\n'.join(shuffle_doc_words_list))

    # split to vector
    for doc_name_list in shuffle_doc_name_list:
        doc_name_list = doc_name_list.split('\t')

    for doc_words_list in shuffle_doc_words_list:
        doc_words_list = doc_words_list.split()

    return shuffle_doc_words_list, shuffle_doc_name_list, train_ids, test_ids


def main():
    if len(sys.argv) != 2:
        sys.exit("Use: python build_graph.py <dataset>")

    datasets = ['20ng', 'R8', 'R52', 'ohsumed', 'mr']
    # build corpus
    dataset = sys.argv[1]

    if dataset not in datasets:
        sys.exit("wrong dataset name")

    doc_words_list, doc_name_list, train_ids, test_ids = read_and_shuffle_dataset(
        dataset)

    train_size = len(train_ids)
    test_size = len(test_ids)
    # x: feature vectors of training docs, no initial features
    # select 90% training set
    val_size = int(0.1 * train_size)
    real_train_size = train_size - val_size

    graph_builder = WordGraphBuilder()
    # word co-occurrence with context windows
    window_size = 20
    row, col, weight = graph_builder.build_graph(doc_words_list, window_size,
                                                 train_size)

    node_size = train_size + len(graph_builder.vocab) + test_size
    adj = sp.csr_matrix((weight, (row, col)), shape=(node_size, node_size))

    label_list, all_labeled_y = graph_builder.extract_label(doc_name_list)

    real_train_y = all_labeled_y[:real_train_size]

    test_y = all_labeled_y[train_size:train_size + test_size]

    all_y = all_labeled_y[train_size]
    # add all the vocabulary y which is all zeros
    all_y = np.concatenate(all_y,
                           np.zeros(len(graph_builder.vocab), all_y.shape[1]))

    print(real_train_y.shape, test_y.shape, all_y.shape)

    print("Saving data to disk")
    # save all the vocab to disk
    with open('./data/corpus/' + dataset + '_vocab.txt', 'w') as f:
        f.write('\n'.join(graph_builder.vocab))

    with open('./data/corpus/' + dataset + '_labels.txt', 'w') as f:
        f.write('\n'.join(label_list))

    with open('./data/' + dataset + '.real_train.name', 'w') as f:
        f.write('\n'.join(doc_name_list[:real_train_size]))

    # dump objects
    with open("./data/ind.{}.real_train_y".format(dataset), 'wb') as f:
        pkl.dump(real_train_y, f)

    with open("./data/ind.{}.test_y".format(dataset), 'wb') as f:
        pkl.dump(test_y, f)

    with open("./data/ind.{}.all_y".format(dataset), 'wb') as f:
        pkl.dump(all_y, f)

    with open("./data/ind.{}.adj".format(dataset), 'wb') as f:
        pkl.dump(adj, f)


if __name__ == "__main__":
    main()