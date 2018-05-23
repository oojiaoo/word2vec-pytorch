import collections
import numpy as np
import math
import os
import random
import time
from six.moves import xrange

data_index = 0

class Options(object):
    def __init__(self, data_file, vocab_size):
        self.vocab_size = vocab_size
        self.save_path = ''
        self.vocab = self.read_data(data_file)
        data_idx, self.count, self.idx2word = self.build_dataset(self.vocab, self.vocab_size)
        self.train_data = self.subsampling(data_idx)
        self.sample_table = self.init_sample_table()
        self.save_vocab()

    def read_data(self, data_file):
        with open(data_file) as f:
            data = f.read().split()
            data = [x for x in data if x != 'eoood']
        return data

    def build_dataset(self, words, n_words):
        count = [['UNK', -1]]
        count.extend(collections.Counter(words).most_common(n_words - 1))
        dictionary = dict()

        #build dictionary，the higher word frequency is, the top word is
        for word, _ in count:
            dictionary[word] = len(dictionary)

        data = list()
        unk_count = 0

        #dataset labelled
        for word in words:
            if word in dictionary:
                index = dictionary[word]
            else:
                index = 0
                unk_count += 1
            data.append(index)

        count[0][1] = unk_count
        reversed_dict = dict(zip(dictionary.values(), dictionary.keys()))
        return data, count, reversed_dict

    #high frequency word subsampled
    #randomly discard common words, and keep the same frequency ranking
    def subsampling(self, data):
        count = [c[1] for c in self.count]
        frequency = count / np.sum(count)
        prob = dict()

        #calculate discard probability
        for idx, x in enumerate(frequency):
            y = (math.sqrt(x / 0.001) + 1) * 0.001 / x
            prob[idx] = y
        subsampled_data = list()
        for word in data:
            if random.random() < prob[word]:
                subsampled_data.append(word)
        return subsampled_data

    def init_sample_table(self):
        count = [ c[1] for c in self.count]
        pow_freq = np.array(count) ** 0.75
        pow_sum = np.sum(pow_freq)
        ratio = pow_freq / pow_sum

        table_size = 1e8
        count = np.round(ratio * table_size)
        sample_table = []

        for idx, x in enumerate(count):
            sample_table += [idx] * int(x)
        return np.array(sample_table)

    def save_vocab(self):
        with open(os.path.join(self.save_path, 'vocab.txt'), 'w') as f:
            for i in xrange(len(self.count)):
                vocab_word = self.idx2word[i]
                f.write('%s %d\n' % (vocab_word, self.count[i][1]))

    def generate_batch(self, window_size, batch_size, neg_sample_size):
        data = self.train_data
        global data_index

        span = 2 * window_size + 1
        context = np.ndarray(shape=(batch_size, 2 * window_size), dtype=np.int64)
        labels = np.ndarray(shape=(batch_size), dtype=np.int64)
        if data_index + span > len(data):
            data_index = 0
            self.process = False

        buffer = data[data_index : data_index + span]
        pos_u = []
        pos_v = []
        for i in range(batch_size):
            context[i, :] = buffer[:window_size] + buffer[window_size+1:]
            labels[i] = buffer[window_size]
            for j in range(span - 1):
                pos_u.append(labels[i])
                pos_v.append(context[i, j])

            data_index += 1
            if data_index + span > len(data):
                buffer = data[:span]
                data_index = 0
                self.process = False
            else:
                buffer = data[data_index : data_index + span]
        neg_v = np.random.choice(self.sample_table, size=(batch_size * 2 * window_size, neg_sample_size))
        return np.array(pos_u), np.array(pos_v), neg_v

