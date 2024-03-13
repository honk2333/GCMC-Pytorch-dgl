import re
import os
import json
import torch
import numpy as np
import random
import pandas as pd
import dgl

from parse import parse_args

from dataset import ml1m
from model import GCMC
from evals import MAE, RMSE, MSE_loss, CE_loss
from sklearn.metrics import classification_report


def get_relation_dict(re_path):
    with open(re_path, 'r', encoding="utf-8") as f:
        line = f.readlines()[0]
        re_dict = json.loads(line)
    return re_dict


def get_key(dic, value):
    return [k for k, v in dic.items() if v == value][0]


if __name__ == '__main__':

    args = parse_args()
    re_dict = get_relation_dict('/home/wanghk/dataset_gui/meore/rel2id.json')

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    data = dgl.data.utils.load_graphs("graph.bin")[0]
    hetero_graph = data[0]
    # model = Model(768, 768 * 2, 768, hetero_graph.etypes)
    user_feats = hetero_graph.nodes['entity'].data['x']
    item_feats = hetero_graph.nodes['object'].data['x']

    # 划分子图
    train_hetero_graph = dgl.heterograph({
        ('entity', etype, 'object'): (hetero_graph.edges(etype=etype)[0][hetero_graph.edges[etype].data['train_edge_mask']], \
                                      hetero_graph.edges(etype=etype)[1][hetero_graph.edges[etype].data['train_edge_mask']])
        for etype in hetero_graph.etypes
    })
    train_hetero_graph.nodes['entity'].data['cj'] =
    train_hetero_graph.nodes['object'].data['ci'] =

    train_dec_graph = train_hetero_graph['entity', :, 'object']
    train_edge_label = train_dec_graph.edata[dgl.ETYPE]

    test_hetero_graph = dgl.heterograph({
        ('entity', etype, 'object'): (hetero_graph.edges(etype=etype)[0][hetero_graph.edges[etype].data['test_edge_mask']], \
                                      hetero_graph.edges(etype=etype)[1][hetero_graph.edges[etype].data['test_edge_mask']])
        for etype in hetero_graph.etypes
    })
    test_dec_graph = test_hetero_graph['entity', :, 'object']
    test_edge_label = test_dec_graph.edata[dgl.ETYPE]
    print("Loading data finished ...\n")

    enc_G = hetero_graph
    train_dec_G = train_dec_graph
    test_dec_G = test_dec_graph
    u_feat = user_feats
    i_feat = item_feats

    u_feat = torch.Tensor(u_feat)
    i_feat = torch.Tensor(i_feat)

    train_labels = train_edge_label
    test_labels = test_edge_label

    config = dict()
    config['dataset'] = args.data_name
    config['n_user'] = data.n_user
    config['n_item'] = data.n_item

    config['u_fea_dim'] = u_feat.shape[1]
    config['i_fea_dim'] = i_feat.shape[1]

    config['embed_dim'] = args.embed_size
    config['rating_values'] = data.rating_values
    config['n_relation'] = data.n_rel

    model = GCMC(config)

    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)

    early_stop_n = 0
    model.reset_parameters()

    for epoch in range(args.epoch):
        model.train()
        optimizer.zero_grad()

        logits = model(u_feat, i_feat, enc_G, train_dec_G)
        loss = CE_loss(logits, train_labels)

        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            test_logits = model(u_feat, i_feat, enc_G, test_dec_G)
            test_loss = CE_loss(test_logits, test_labels)
            pred_labels = torch.argmax(test_logits, dim=1)
            sk_result = classification_report(y_true=test_labels, y_pred=pred_labels,
                                              labels=list(re_dict.values())[1:],
                                              target_names=list(re_dict.keys())[1:], digits=4)

