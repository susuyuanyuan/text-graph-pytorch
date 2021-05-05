import os


class FolderStructure:
    def __init__(self, dataset, root_dir="./"):
        self.dataset = dataset
        self.root_dir = root_dir
        if not os.path.exists(self.get_output_dir()):
            os.mkdir(self.get_output_dir())
        if not os.path.exists(self.get_input_dir()):
            os.mkdir(self.get_input_dir())

    def get_original_doc_name_file(self):
        return os.path.join(self.root_dir, 'dataset', self.dataset + '.txt')

    def get_output_dir(self):
        return os.path.join(self.root_dir, 'output_data', self.dataset)

    def get_input_dir(self):
        return os.path.join(self.root_dir, 'input_data')

    def get_original_doc_words_file(self):
        return os.path.join(self.root_dir, 'dataset',
                            'corpus/' + self.dataset + '.txt')

    def get_doc_name_file(self):
        return os.path.join(self.root_dir, 'input_data',
                            self.dataset + '_names.txt')

    def get_clean_doc_words_file(self):
        return os.path.join(self.root_dir, 'input_data',
                            self.dataset + '_words.txt')

    def get_train_index_file(self):
        return os.path.join(self.get_output_dir(), 'train_index.txt')

    def get_test_index_file(self):
        return os.path.join(self.get_output_dir(), 'test_index.txt')

    def get_shuffled_doc_names_file(self):
        return os.path.join(self.get_output_dir(), 'doc_names_shuffle.txt')

    def get_shuffled_doc_words_file(self):
        return os.path.join(self.get_output_dir(), 'doc_words_shuffle.txt')

    def get_vocab_file(self):
        return os.path.join(self.get_output_dir(), 'vocab.txt')

    def get_labels_file(self):
        return os.path.join(self.get_output_dir(), 'labels.txt')

    def get_real_train_name(self):
        return os.path.join(self.get_output_dir(), 'real_train_name.txt')

    def get_pickle_file(self, name):
        return os.path.join(self.get_output_dir(), name + ".pickle")

    def get_doc_vector_file(self):
        return os.path.join(self.get_output_dir(), 'doc_vectors.txt')

    def get_word_vector_file(self):
        return os.path.join(self.get_output_dir(), 'word_vectors.txt')