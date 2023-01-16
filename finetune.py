import argparse

from loader import MoleculeDataset
from datautils import moltree_to_graph_data
from torch_geometric.loader import DataLoader
from get_vocab import get_motifs

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
import numpy as np

from gnn_model import GNN, GNN_graphpred
from rationale import rationale_motif_pred
from sklearn.metrics import roc_auc_score

from splitters import scaffold_split, random_split
import pandas as pd
from rdkit import Chem

import os
import shutil

criterion = nn.BCEWithLogitsLoss(reduction = "none")

def train(args, model_list, device, loader, optimizer_list, smiles_list):

    graph_pred, encoder_model, rationale_model = model_list
    optimizer_pred, optimizer_encoder, optimizer_rationale = optimizer_list
    
    graph_pred.train()
    rationale_model.train()

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):

        batch = batch.to(device)

        smiles = [smiles_list[i] for i in batch.id]

        motif_batch = get_motifs(smiles)
        motif_batch_graph = []
        for list in motif_batch:
            motif_batch_graph.append(moltree_to_graph_data(list))

        motif_node_rep = []
        for motif_list in motif_batch_graph:
            motif_rep = encoder_model(motif_list.x, motif_list.edge_index, motif_list.edge_attr)
            motif_node_rep.append(motif_rep)

        node_rep = encoder_model(batch.x, batch.edge_index, batch.edge_attr)

        motif_rat_rep, motif_rat_rep_aug, motif_loss = rationale_model(encoder_model, batch, node_rep, motif_node_rep)

        y_pred = graph_pred(node_rep, batch.batch, motif_rat_rep)
        y_pred_aug = graph_pred(node_rep, batch.batch, motif_rat_rep_aug)
        y_true = batch.y.view(y_pred.shape).to(torch.float64)

        #Whether y is non-null or not.
        is_valid = y_true**2 > 0

        #Loss matrix
        loss_mat = criterion(y_pred.double(), (y_true+1)/2)
        loss_mat_aug = criterion(y_pred_aug.double(), (y_true+1)/2)
        #loss matrix after removing null target
        loss_mat = torch.where(is_valid, loss_mat, torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))
        loss_mat_aug = torch.where(is_valid, loss_mat_aug, torch.zeros(loss_mat_aug.shape).to(loss_mat_aug.device).to(loss_mat_aug.dtype))
            
        optimizer_pred.zero_grad()
        optimizer_rationale.zero_grad()

        loss = torch.sum(loss_mat)/torch.sum(is_valid)
        loss_aug = torch.sum(loss_mat_aug)/torch.sum(is_valid)
        full_loss = loss + loss_aug
        #print(loss)
        full_loss.backward()

        optimizer_pred.step()
        optimizer_rationale.step()


def eval(args, model_list, device, loader, smiles_list):

    graph_pred, encoder_model, rationale_model = model_list

    graph_pred.eval()
    encoder_model.eval()
    rationale_model.eval()

    y_true = []
    y_scores = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):

        batch = batch.to(device)

        smiles = [smiles_list[i] for i in batch.id]
        motif_batch = get_motifs(smiles)
        motif_batch_graph = []
        for list in motif_batch:
            motif_batch_graph.append(moltree_to_graph_data(list))

        motif_node_rep = []
        for motif_list in motif_batch_graph:
            motif_rep = encoder_model(motif_list.x, motif_list.edge_index, motif_list.edge_attr)
            motif_node_rep.append(motif_rep)

        with torch.no_grad():
            node_rep = encoder_model(batch.x, batch.edge_index, batch.edge_attr)
            motif_rat_rep, _, _ = rationale_model(encoder_model, batch, node_rep, motif_node_rep)
            y_pred = graph_pred(node_rep, batch.batch, motif_rat_rep)

        y_true.append(batch.y.view(y_pred.shape))

        y_scores.append(y_pred)

    y_true = torch.cat(y_true, dim = 0).cpu().numpy()
    y_scores = torch.cat(y_scores, dim = 0).cpu().numpy()

    roc_list = []
    for i in range(y_true.shape[1]):
        #AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:,i] == 1) > 0 and np.sum(y_true[:,i] == -1) > 0:
            is_valid = y_true[:,i]**2 > 0
            roc_list.append(roc_auc_score((y_true[is_valid,i] + 1)/2, y_scores[is_valid,i]))

    if len(roc_list) < y_true.shape[1]:
        print("Some target is missing!")
        print("Missing ratio: %f" %(1 - float(len(roc_list))/y_true.shape[1]))

    return sum(roc_list)/len(roc_list) #y_true.shape[1]



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
    parser.add_argument('--lr_scale', type=float, default=1,
                        help='relative learning rate for the feature extraction layer (default: 1)')
    parser.add_argument('--decay', type=float, default=0,
                        help='weight decay (default: 0)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5).')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='embedding dimensions (default: 300)')
    parser.add_argument('--dropout_ratio', type=float, default=0.5,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--graph_pooling', type=str, default="mean",
                        help='graph level pooling (sum, mean, max, set2set, attention)')
    parser.add_argument('--JK', type=str, default="last",
                        help='how the node features across layers are combined. last, sum, max or concat')
    parser.add_argument('--gnn_type', type=str, default="gin")
    parser.add_argument('--dataset', type=str, default = 'hiv', help='root directory of dataset. For now, only classification.')
    parser.add_argument('--input_model_file', type=str, default = './saved_model/motif_pretrain.pth', help='filename to read the model (if there is any)')
    parser.add_argument('--filename', type=str, default = '', help='output filename')
    parser.add_argument('--seed', type=int, default=42, help = "Seed for splitting the dataset.")
    parser.add_argument("--hidden_size", type=int, default=300, help='hidden size')
    parser.add_argument('--runseed', type=int, default=0, help = "Seed for minibatch selection, random initialization.")
    parser.add_argument('--split', type = str, default="scaffold", help = "random or scaffold or random_scaffold")
    parser.add_argument('--eval_train', type=int, default = 0, help='evaluating training or not')
    parser.add_argument('--num_workers', type=int, default = 4, help='number of workers for dataset loading')
    args = parser.parse_args()

    torch.manual_seed(args.runseed)
    np.random.seed(args.runseed)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.runseed)

    #Bunch of classification tasks
    if args.dataset == "tox21":
        num_tasks = 12
    elif args.dataset == "hiv":
        num_tasks = 1
    elif args.dataset == "pcba":
        num_tasks = 128
    elif args.dataset == "muv":
        num_tasks = 17
    elif args.dataset == "bace":
        num_tasks = 1
    elif args.dataset == "bbbp":
        num_tasks = 1
    elif args.dataset == "toxcast":
        num_tasks = 617
    elif args.dataset == "sider":
        num_tasks = 27
    elif args.dataset == "clintox":
        num_tasks = 2
    else:
        raise ValueError("Invalid dataset name.")

    #set up dataset
    dataset = MoleculeDataset("data/" + args.dataset, dataset=args.dataset)
    
    if args.split == "scaffold":
        smiles_list = pd.read_csv('data/' + args.dataset + '/processed/smiles.csv', header=None)[0].tolist()
        train_dataset, valid_dataset, test_dataset = scaffold_split(dataset, smiles_list, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1)
        print("scaffold")
    elif args.split == "random":
        train_dataset, valid_dataset, test_dataset = random_split(dataset, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1, seed = args.seed)
        print("random")
    elif args.split == "random_scaffold":
        smiles_list = pd.read_csv('data/' + args.dataset + '/processed/smiles.csv', header=None)[0].tolist()
        train_dataset, valid_dataset, test_dataset = random_scaffold_split(dataset, smiles_list, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1, seed = args.seed)
        print("random scaffold")
    else:
        raise ValueError("Invalid split option.")

    smiles_list = pd.read_csv('data/' + args.dataset + '/processed/smiles.csv', header=None)[0].tolist()

    #print(train_dataset[0])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)
    val_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)

    #set up model
    graph_pred_model = GNN_graphpred(args.num_layer, args.emb_dim, num_tasks, JK = args.JK, drop_ratio = args.dropout_ratio, graph_pooling = args.graph_pooling, gnn_type = args.gnn_type).to(device)

    #if not args.input_model_file == "":
    #    model.from_pretrained(args.input_model_file)

    # TODO TURN LOAD STATE DICT BACK ON ONCE PRETRAINING IS FINISHED

    encoder_model = GNN(args.num_layer, args.emb_dim, device, JK=args.JK, drop_ratio=args.dropout_ratio, gnn_type=args.gnn_type).to(device)
    #encoder_model.load_state_dict(torch.load("./saved_model/encoder.pth"))
    rationale_model = rationale_motif_pred(device, args.hidden_size, num_tasks, 5, args.emb_dim, args.gnn_type, args.dropout_ratio, 0.4, False).to(device)
    #rationale_model.load_state_dict(torch.load("./saved_model/rationale.pth"))

    model = [graph_pred_model, encoder_model, rationale_model]

    #optimizer_pred = optim.Adam(model_param_group, lr=args.lr, weight_decay=args.decay)
    optimizer_pred = optim.Adam(graph_pred_model.parameters(), lr=args.lr, weight_decay=args.decay)
    optimizer_encoder = optim.Adam(encoder_model.parameters(), lr=args.lr, weight_decay=args.decay)
    optimizer_rationale = optim.Adam(rationale_model.parameters(), lr=args.lr, weight_decay=args.decay)

    optimizer = [optimizer_pred, optimizer_encoder, optimizer_rationale]

    for epoch in range(1, args.epochs+1):
        print("====epoch " + str(epoch))
        
        train(args, model, device, train_loader, optimizer, smiles_list)

        val_acc = eval(args, model, device, val_loader, smiles_list)
        test_acc = eval(args, model, device, test_loader, smiles_list)

        print("val: %f test: %f" %(val_acc, test_acc))

if __name__ == "__main__":
    main()
