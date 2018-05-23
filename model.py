import torch
import torch.nn as nn
import torch.nn.functional as F

class CBOW(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(CBOW, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, inputs):
        X = torch.mean(self.embeddings(inputs), dim=0).view((1, -1))
        X = self.linear(X)
        X = F.log_softmax(X)
        return X

class SkipGram(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(SkipGram, self).__init__()
        self.u_embeddings = nn.Embedding(vocab_size, embedding_dim, sparse=True)
        self.v_embeddings = nn.Embedding(vocab_size, embedding_dim, sparse=True)
        self.embedding_dim = embedding_dim
        initrange = 0.5 / self.embedding_dim
        self.u_embeddings.weight.data.uniform_(-initrange, initrange)
        self.v_embeddings.weight.data.uniform_(-0, 0)

    def forward(self, u_pos, v_pos, v_neg, batch_size):
        embed_u = self.u_embeddings(u_pos)
        pos_embed_v = self.v_embeddings(v_pos)
        #print('embed_u.shape', embed_u.shape)
        #print('pos_embed_v.shape', pos_embed_v.shape)
        pos_score = torch.sum(torch.mul(embed_u, pos_embed_v), dim = 1)
        pos_output = F.logsigmoid(pos_score).squeeze()

        neg_embed_v = self.v_embeddings(v_neg)
        #print('neg_embed_v.shape', neg_embed_v.shape)
        neg_score = torch.bmm(neg_embed_v, embed_u.unsqueeze(2)).squeeze()
        neg_score = torch.sum(neg_score, dim = 1)
        neg_output = F.logsigmoid(-1*neg_score).squeeze() #1-sigma(x)=sigma(-x)

        cost = pos_output + neg_output
        return -1 * cost.sum() / batch_size

    def save_embeddings(self, id2word, file_name, use_cuda):
        if use_cuda:
            embedding = self.u_embeddings.weight.cpu().data.numpy()
        else:
            embedding = self.u_embeddings.weight.data.numpy()

        fout = open(file_name, 'w')
        fout.write('%d %d\n' % (len(id2word), self.embedding_dim))
        for wid, w in id2word.items():
            e = embedding[wid]
            e = ' '.join(map(lambda x: str(x), e))
            fout.write('%s %s\n' % (w, e))
