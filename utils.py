import random
import numpy as np
import torch


def normalize(adj):
    degree = adj.sum(1)
    d_hat = np.diag(np.power(degree, -0.5).flatten())
    return d_hat.dot(adj).dot(d_hat)


def make_adjacency(args):
    GCN_adj = []
    for i in range(args.num_scale):
        N = args.num_frames - i
        sub_adj = np.zeros(shape=(N, N), dtype=np.float32)
        for j in range(N):
            for k in range(N):
                if k-1 > j+i or j-1 > k+i:
                    continue
                else:
                    sub_adj[j][k] = 1
        sub_adj = normalize(sub_adj)
        sub_adj = torch.from_numpy(sub_adj)
        GCN_adj.append(sub_adj)

    GAT_adj = []
    for i in range(args.num_scale - 1):
        N = args.num_frames - i
        M = args.num_frames - i - 1
        sub_adj = np.zeros(shape=(N, M), dtype=np.float32)
        for j in range(N):
            for k in range(M):
                if j > k+i or j+i < k:
                    continue
                else:
                    sub_adj[j][k] = 1
        sub_adj = torch.from_numpy(sub_adj)
        GAT_adj.append(sub_adj)

    return GCN_adj, GAT_adj


def save_model(args, model, save_path, score_all, score_word=None):
    save_kwargs ={
        "args": args,
        "score_all": score_all,
        "score_word": score_word,
        "model": model.state_dict()
    }
    torch.save(save_kwargs, save_path)


def load_model(load_path):
    kwargs = torch.load(load_path)
    args = kwargs["args"]
    model_dict = kwargs["model"]
    args.train = False
    args.val = False
    args.test = True
    if args.dataset in ['msvd-qa', 'msrvtt-qa']:
        print("The best result of question words are {} !".format(kwargs["score_word"]))
    print("The best result is \033[1;31m {:.4f} \033[0m !".format(kwargs["score_all"]))
    return args, model_dict