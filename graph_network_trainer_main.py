from __future__ import division
from __future__ import print_function

from sklearn import metrics
import sys
import os

import torch

import numpy as np
import scipy.sparse as sparse

import utils
from gcn import GCN
from mlp import MLP

from optimization_config import OptimizationConfig
from graph_network_trainer import GraphNetworkTrainer
from folder_structure import FolderStructure

import argparse


def print_and_save_result(dataset, embedding, all_size, train_size, test_size):
    word_embeddings = embedding[train_size:all_size - test_size]
    train_doc_embeddings = embedding[:train_size]  # include val docs
    test_doc_embeddings = embedding[all_size - test_size:]

    print('Embeddings:')
    print('\rWord_embeddings:' + str(len(word_embeddings)))
    print('\rTrain_doc_embeddings:' + str(len(train_doc_embeddings)))
    print('\rTest_doc_embeddings:' + str(len(test_doc_embeddings)))
    print('\rWord_embeddings:')
    print(word_embeddings)

    fs = FolderStructure(dataset)
    with open(fs.get_vocab_file(), 'r') as f:
        words = f.readlines()

    vocab_size = len(words)
    word_vectors = []
    for i in range(vocab_size):
        word = words[i].strip()
        word_vector = word_embeddings[i]
        word_vector_str = ' '.join([str(x) for x in word_vector])
        word_vectors.append(word + ' ' + word_vector_str)

    word_embeddings_str = '\n'.join(word_vectors)
    with open(fs.get_word_vector_file(), 'w') as f:
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
    with open(fs.get_doc_vector_file(), 'w') as f:
        f.write(doc_embeddings_str)


def get_data(dataset):
    # Load output_data
    (adj, y_train, y_val, y_test, train_mask, val_mask, test_mask, train_size,
     test_size) = utils.load_data(dataset)

    features = sparse.identity(adj.shape[1])

    # Some preprocessing
    features = utils.preprocess_features(features)
    support = [utils.preprocess_adj(adj)]

    # Define placeholders
    t_features = torch.from_numpy(features)
    t_y_train = torch.from_numpy(y_train)
    t_y_val = torch.from_numpy(y_val)
    t_y_test = torch.from_numpy(y_test)
    t_train_mask = torch.from_numpy(train_mask.astype(np.float32))

    t_support = []
    for i in range(len(support)):
        t_support.append(torch.Tensor(support[i]))

    return (t_features, t_y_train, t_y_val, t_y_test, t_train_mask, t_support,
            val_mask, test_mask, train_size, test_size)


if __name__ == "__main__":
    config = OptimizationConfig()

    arg = argparse.ArgumentParser()
    arg.add_argument(
        "dataset_name",
        default="",
        help="The dataset name, please pick one from 20ng, R8, R52, ohsumed, mr"
    )
    args = arg.parse_args()

    dataset = args.dataset_name

    datasets = ['20ng', 'R8', 'R52', 'ohsumed', 'mr']
    if dataset not in datasets:
        sys.exit("wrong dataset name")
    config.dataset = dataset

    # Set random seed
    np.random.seed(2019)
    torch.manual_seed(2019)

    if config.model == 'gcn':
        model_func = GCN
    elif config.model == 'dense':
        model_func = MLP
    else:
        raise ValueError('Invalid argument for model: ' + str(config.model))

    (t_features, t_y_train, t_y_val, t_y_test, t_train_mask, t_support,
     val_mask, test_mask, train_size, test_size) = get_data(config.dataset)

    model = model_func(input_dim=t_features.shape[0],
                       support=t_support,
                       num_classes=t_y_train.shape[1])

    trainer = GraphNetworkTrainer(model, config)
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
    print_and_save_result(dataset, embedding, t_support[0].shape[0],
                          train_size, test_size)
