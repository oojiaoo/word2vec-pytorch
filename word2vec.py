import time

import torch
from torch.autograd import Variable
import torch.optim as optim

from model import SkipGram, CBOW
from preprocess_data import Options


class word2vec:
    def __init__(self, input_file, model_name, vocabulary_size=100000,
                 embedding_dim=200, epoch=10, batch_size=256, windows_size=5, neg_sample_size=10):
        self.model_name = model_name
        self.op = Options(input_file, vocabulary_size)
        self.vocabulary_size = vocabulary_size
        self.embedding_dim = embedding_dim
        self.epoch = epoch
        self.batch_size = batch_size
        self.windows_size = windows_size
        self.neg_sample_size = neg_sample_size

    def train(self):
        if self.model_name == 'SkipGram':
            model = SkipGram(self.vocabulary_size, self.embedding_dim)
        elif self.model_name == 'CBOW':
            return

        if torch.cuda.is_available():
            model.cuda()

        optimizer = optim.SGD(model.parameters(), lr=0.2)

        for epoch in range(self.epoch):
            start = time.time()
            self.op.process =True
            batch_num = 0
            batch_new = 0

            while self.op.process:
                pos_u, pos_v, neg_v = self.op.generate_batch(self.windows_size, self.batch_size, self.neg_sample_size)

                pos_u = Variable(torch.LongTensor(pos_u))
                pos_v = Variable(torch.LongTensor(pos_v))
                neg_v = Variable(torch.LongTensor(neg_v))


                if torch.cuda.is_available():
                    pos_u = pos_u.cuda()
                    pos_v = pos_v.cuda()
                    neg_v = neg_v.cuda()

                optimizer.zero_grad()
                loss = model(pos_u, pos_v, neg_v, self.batch_size)
                loss.backward()
                optimizer.step()

                if batch_num % 3000 == 0:
                    end = time.time()
                    print('epoch,batch = %2d %5d:   pair/sec = %4.2f  loss = %4.3f\r'
                          % (epoch, batch_num, (batch_num - batch_new)*self.batch_size/(end-start), loss.data[0]), end="\n")
                    batch_new = batch_num
                    start = time.time()
                batch_num += 1

        model.save_embeddings(self.op.idx2word, 'word_embdding.txt', torch.cuda.is_available())


if __name__ == '__main__':
    w2v = word2vec('text8', 'SkipGram')
    w2v.train()



