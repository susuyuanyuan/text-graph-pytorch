from __future__ import division
from __future__ import print_function

from sklearn import metrics
import random
import time
import sys
import os

import torch
import torch.nn as nn

import numpy as np
import scipy.sparse as sparse

import utils
from gcn import GCN
from mlp import MLP

from config import CONFIG
from graph_network_trainer import GraphNetworkTrainer


def print_and_save_result(embedding, train_size, test_size):
    word_embeddings = embedding[train_size:adj.shape[0] - test_size]
    train_doc_embeddings = embedding[:train_size]  # include val docs
    test_doc_embeddings = embedding[adj.shape[0] - test_size:]

    print('Embeddings:')
    print('\rWord_embeddings:' + str(len(word_embeddings)))
    print('\rTrain_doc_embeddings:' + str(len(train_doc_embeddings)))
    print('\rTest_doc_embeddings:' + str(len(test_doc_embeddings)))
    print('\rWord_embeddings:')
    print(word_embeddings)

    with open('./data/corpus/' + dataset + '_vocab.txt', 'r') as f:
        words = f.readlines()

    vocab_size = len(words)
    word_vectors = []
    for i in range(vocab_size):
        word = words[i].strip()
        word_vector = word_embeddings[i]
        word_vector_str = ' '.join([str(x) for x in word_vector])
        word_vectors.append(word + ' ' + word_vector_str)

    word_embeddings_str = '\n'.join(word_vectors)
    with open('./data/' + dataset + '_word_vectors.txt', 'w') as f:
        f.write(word_embeddings_str)

    doc_vectors = []
    doc_id = 0
    for i in range(train_size):
        doc_vector = train_doc_embeddings[i]
        doc_vector_str = ' '.join([str(x) for x in doc_vector])
        doc_vectors.append('doc_' + str(doc_id) + ' ' + doc_vector_str)
        doc_id += 1

    for i in range(test_size):
        doc_vector = test_doc_embeddings[i]
        doc_vector_str = ' '.join([str(x) for x in doc_vector])
        doc_vectors.append('doc_' + str(doc_id) + ' ' + doc_vector_str)
        doc_id += 1

    doc_embeddings_str = '\n'.join(doc_vectors)
    with open('./data/' + dataset + '_doc_vectors.txt', 'w') as f:
        f.write(doc_embeddings_str)


if __name__ == "__main__":

    cfg = CONFIG()

    if len(sys.argv) != 2:
        sys.exit("Use: python train.py <dataset>")

    datasets = ['20ng', 'R8', 'R52', 'ohsumed', 'mr']
    dataset = sys.argv[1]

    if dataset not in datasets:
        sys.exit("wrong dataset name")
    cfg.dataset = dataset

    # Set random seed
    seed = random.randint(1, 200)
    seed = 2019
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # Settings
    # os.environ["CUDA_VISIBLE_DEVICES"] = ""

    # Load data
    (adj, y_train, y_val, y_test, train_mask, val_mask, test_mask, train_size,
     test_size) = utils.load_corpus(cfg.dataset)

    embedding_size = 300
    features = sparse.identity(adj.shape[1])

    # Some preprocessing
    features = utils.preprocess_features(features)
    if cfg.model == 'gcn':
        support = [utils.preprocess_adj(adj)]
        num_supports = 1
        model_func = GCN
    elif cfg.model == 'dense':
        support = [utils.preprocess_adj(adj)]  # Not used
        num_supports = 1
        model_func = MLP
    else:
        raise ValueError('Invalid argument for model: ' + str(cfg.model))

    # Define placeholders
    t_features = torch.from_numpy(features)
    t_y_train = torch.from_numpy(y_train)
    t_y_val = torch.from_numpy(y_val)
    t_y_test = torch.from_numpy(y_test)
    t_train_mask = torch.from_numpy(train_mask.astype(np.float32))

    t_support = []
    for i in range(len(support)):
        t_support.append(torch.Tensor(support[i]))

    if torch.cuda.is_available():
        model_func = model_func.cuda()
        t_features = t_features.cuda()
        t_y_train = t_y_train.cuda()
        t_y_val = t_y_val.cuda()
        t_y_test = t_y_test.cuda()
        t_train_mask = t_train_mask.cuda()
        for i in range(len(support)):
            t_support = [t.cuda() for t in t_support if True]

    model = model_func(input_dim=features.shape[0],
                       support=t_support,
                       num_classes=y_train.shape[1])

    trainer = GraphNetworkTrainer(model, cfg)

    trainer.train(t_features, t_y_train, t_train_mask, t_y_val, val_mask)

    # Testing
    test_loss, test_acc, pred, labels, test_duration = trainer.evaluate(
        t_features, t_y_test, test_mask)
    print(
        "Test set results: \n\t loss= {:.5f}, accuracy= {:.5f}, time= {:.5f}".
        format(test_loss, test_acc, test_duration))

    test_pred = []
    test_labels = []
    for i in range(len(test_mask)):
        if test_mask[i]:
            test_pred.append(pred[i])
            test_labels.append(np.argmax(labels[i]))

    print("Test Precision, Recall and F1-Score...")
    print(metrics.classification_report(test_labels, test_pred, digits=4))
    print("Macro average Test Precision, Recall and F1-Score...")
    print(
        metrics.precision_recall_fscore_support(test_labels,
                                                test_pred,
                                                average='macro'))
    print("Micro average Test Precision, Recall and F1-Score...")
    print(
        metrics.precision_recall_fscore_support(test_labels,
                                                test_pred,
                                                average='micro'))

    # doc and word embeddings
    embedding = model.layer1.embedding.numpy()
    print_and_save_result(embedding, train_size, test_size)