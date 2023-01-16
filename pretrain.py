import argparse
#import torch
#import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
#from torch_geometric.loader import DataLoader
import math, random, sys
from tqdm import tqdm
import numpy as np
from optparse import OptionParser
from get_vocab import get_motifs
from sklearn.model_selection import train_test_split


from gnn_model import GNN, GNN_graphpred

from datautils import *
from rationale import rationale_motif_pred
from ogb.graphproppred import PygGraphPropPredDataset

import rdkit
from rdkit import Chem


def group_node_rep(node_rep, batch_index, batch_size):
    group = []
    count = 0
    for i in range(batch_size):
        num = sum(batch_index == i)
        group.append(node_rep[count:count + num])
        count += num
    return group


def train(args, model_list, loader, optimizer_list, device):

    encoder_model, rationale_model = model_list
    optimizer_encoder, optimizer_rationale = optimizer_list

    encoder_model.train()
    rationale_model.train()

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):

        mol = [x.split(",")[0] for x in batch]

        graph_batch = moltree_to_graph_data(mol)
        motif_batch = get_motifs(mol)
        motif_batch_graph = []
        for list in motif_batch:
            motif_batch_graph.append(moltree_to_graph_data(list))

        graph_batch.to(device)

        node_rep = encoder_model(graph_batch.x, graph_batch.edge_index, graph_batch.edge_attr)

        motif_node_rep = []
        for motif_list in motif_batch_graph:
            motif_rep = encoder_model(motif_list.x, motif_list.edge_index, motif_list.edge_attr)
            motif_node_rep.append(motif_rep)

        _, _, motif_loss = rationale_model(encoder_model, graph_batch, node_rep, motif_node_rep)

        optimizer_encoder.zero_grad()
        optimizer_rationale.zero_grad()

        motif_loss.backward()

        optimizer_encoder.step()
        optimizer_rationale.step()


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--decay', type=float, default=0,
                        help='weight decay (default: 0)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5).')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='embedding dimensions (default: 300)')
    parser.add_argument('--dropout_ratio', type=float, default=0.2,
                        help='dropout ratio (default: 0.2)')
    parser.add_argument('--graph_pooling', type=str, default="mean",
                        help='graph level pooling (sum, mean, max, set2set, attention)')
    parser.add_argument('--JK', type=str, default="last",
                        help='how the node features across layers are combined. last, sum, max or concat')
    parser.add_argument('--dataset', type=str, default='./data/zinc/all.txt',
                        help='root directory of dataset. For now, only classification.')
    parser.add_argument('--gnn_type', type=str, default="gin")
    parser.add_argument('--input_model_file', type=str, default='./saved_model/init', help='filename to read the model (if there is any)')
    parser.add_argument('--output_model_file', type=str, default='./saved_model/motif_pretrain',
                        help='filename to output the pre-trained model')
    parser.add_argument('--num_workers', type=int, default=8, help='number of workers for dataset loading')
    parser.add_argument("--hidden_size", type=int, default=300, help='hidden size')
    parser.add_argument("--latent_size", type=int, default=56, help='latent size')
    parser.add_argument("--vocab", type=str, default='./data/chembl/vocab.txt', help='vocab path')
    parser.add_argument('--order', type=str, default="bfs",
                        help='motif tree generation order (bfs or dfs)')
    parser.add_argument('--seed', type=int, default=42, help = "Seed for splitting the dataset.")
    parser.add_argument('--split', type = str, default="random", help = "random or scaffold or random_scaffold")
    args = parser.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    print(device)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    dataset = MoleculeDataset(args.dataset)

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=lambda x:x, drop_last=True)

    encoder_model = GNN(args.num_layer, args.emb_dim, device, JK=args.JK, drop_ratio=args.dropout_ratio, gnn_type=args.gnn_type).to(device)
    rationale_model = rationale_motif_pred(device, args.hidden_size, 1, 5, args.emb_dim, args.gnn_type, args.dropout_ratio, 0.4, False).to(device)

    #if not args.input_model_file == "":
    #    encoder_model.load_state_dict(torch.load(args.input_model_file + ".pth"))

    model_list = [encoder_model, rationale_model]

    optimizer_encoder = optim.Adam(encoder_model.parameters(), lr=args.lr, weight_decay=args.decay)
    optimizer_rationale = optim.Adam(rationale_model.parameters(), lr=args.lr, weight_decay=args.decay)

    optimizer_list = [optimizer_encoder, optimizer_rationale]

    for epoch in range(1, args.epochs + 1):
        print("====epoch " + str(epoch))

        train(args, model_list, loader, optimizer_list, device)

        torch.save(encoder_model.state_dict(), "saved_model/encoder.pth")
        torch.save(rationale_model.state_dict(), "saved_model/rationale.pth")

        #if not args.output_model_file == "":
        #    torch.save(encoder_model.state_dict(), args.output_model_file + ".pth")


if __name__ == "__main__":
    main()
