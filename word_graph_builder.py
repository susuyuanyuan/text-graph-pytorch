import numpy as np
from math import log
from collections import defaultdict, Counter


class WordGraphBuilder:
    def __init__(self):
        self.vocab = None
        self.word_id_map = None
        self.word_doc_freq = None
        self.word_window_freq = None
        self.word_pair_counter = None
        self.num_window = 0
        self.vocab_size = 0

    def build_graph(self, doc_words_list, window_size, train_size):
        print("Building Graph")
        self._build_vocab(doc_words_list)
        self._compute_word_frequency(doc_words_list, window_size)
        return self._compute_adjacency_matrix(doc_words_list, train_size)

    def extract_label(self, doc_name_list):
        """
        given the list of the doc name
        return all the labels of all the doc and it's corresponding one hot vector
        """
        label_id_map = {}
        index = 0
        all_labels = []
        for doc_names in doc_name_list:
            label = doc_names[2]
            all_labels.append(label)
            if label not in label_id_map:
                label_id_map[label] = index
                index = index + 1
        onehot_y = np.zeros((len(all_labels), len(label_id_map)))
        for i, label in enumerate(all_labels):
            onehot_y[i, label_id_map[label]] = 1
        return all_labels, onehot_y

    def _build_vocab(self, doc_words_list):
        """
        given a list of documents (list of string),
        where each document is a space separated words(string)
        initializes
        1. vocab: all the unique words (list of string)
        2. world_id_map: the id mapping to the each word (map of string to int)
        3. world_doc_freq: the frequency of how many times this word appears in different docs
                           (map of string to int)
        """
        print("Building vocabulary")
        # find out the frequency of the word appears in different doc
        # word doc freq stores how many times this word appeared in different docs
        self.word_doc_freq = defaultdict(int)
        for doc_words in doc_words_list:
            doc_unique_words = set(doc_words)
            for word in doc_unique_words:
                self.word_doc_freq[word] += 1

        # word id map is to convert the word into it's index in vocab
        self.word_id_map = {}
        self.vocab = []
        for word in self.word_doc_freq:
            self.word_id_map[word] = len(self.vocab)
            self.vocab.append(word)
        self.vocab_size = len(self.vocab)

    def _update_window_freq(self, window):
        appeared = set()
        for word in window:
            if word in appeared:
                continue
            self.word_window_freq[word] += 1
            appeared.add(word)

    def _update_word_pair_counter(self, window):
        # lookup ids for words in this window
        word_ids = []
        for word in window:
            word_ids.append(self.word_id_map[word])

        # for each pair of words in this window, update the counter
        for i in range(1, len(window)):
            for j in range(0, i):
                id1 = word_ids[i]
                id2 = word_ids[j]
                if id1 == id2:
                    continue
                # since we know the max vocab size, we can compute the pair hash by using the max
                self.word_pair_counter[id1 * self.vocab_size + id2] += 1
                self.word_pair_counter[id2 * self.vocab_size + id1] += 1

    def _add_window(self, window):
        self.num_window += 1
        self._update_window_freq(window)
        self._update_word_pair_counter(window)

    def _compute_word_frequency(self, doc_words_list, window_size):
        print("Computing word frequency")
        self.word_pair_counter = defaultdict(int)
        self.word_window_freq = defaultdict(int)

        # word co-occurrence with context windows
        self.num_window = 0

        for doc_words in doc_words_list:
            words_length = len(doc_words)
            if words_length <= window_size:
                self._add_window(doc_words)
                continue
            # sliding window
            for j in range(words_length - window_size + 1):
                self._add_window(doc_words[j:j + window_size])

    def _compute_pmi(self, world_pair_count, word_i_freq, word_j_freq):
        return log((1.0 * world_pair_count / self.num_window) /
                   (1.0 * word_i_freq * word_j_freq /
                    (self.num_window * self.num_window)))

    def _compute_adjacency_matrix(self, doc_words_list, train_size):
        print("Computing adjacency matrix")

        row = []
        col = []
        weight = []
        vocab_size = len(self.vocab)

        # use window frequency to compute the weight for testing samples
        for word_id_hash, pair_count in self.word_pair_counter.items():
            id1 = int(word_id_hash / self.vocab_size)
            id2 = int(word_id_hash - id1 * self.vocab_size)
            pmi = self._compute_pmi(pair_count,
                                    self.word_window_freq[self.vocab[id1]],
                                    self.word_window_freq[self.vocab[id2]])
            if pmi <= 0:
                continue
            row.append(train_size + id1)
            col.append(train_size + id2)
            weight.append(pmi)

        for doc_id, doc_words in enumerate(doc_words_list):
            doc_word_freq = Counter(doc_words)
            for word, in_doc_freq in doc_word_freq.items():
                word_id = self.word_id_map[word]
                if doc_id < train_size:
                    row.append(doc_id)
                else:
                    row.append(doc_id + vocab_size)
                col.append(train_size + word_id)
                idf = log(1.0 * len(doc_words_list) /
                          self.word_doc_freq[self.vocab[word_id]])
                weight.append(in_doc_freq * idf)

        return row, col, weight
