import argparse
import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.linalg as linalg
#from torch.utils.data import DataLoader
import torch.nn.functional as F
from datautils import DataLoaderMaskingPred
#from torch_geometric.loader import DataLoader
import math, random, sys
from tqdm import tqdm
import numpy as np
from optparse import OptionParser
from functools import partial

from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set, aggr

from gnn_model import GNN, GNNDecoder

from datautils import *
from loader import MoleculeDataset

import rdkit
from rdkit import Chem, DataStructs


def group_node_rep(node_rep, batch_index, batch_size):
    group = []
    count = 0
    for i in range(batch_size):
        num = sum(batch_index == i)
        group.append(node_rep[count:count + num])
        count += num
    return group

def sce_loss(x, y, alpha=1):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)

    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)
    loss = loss.mean()
    return loss

def train(args, model_list, loader, optimizer_list, device, smiles_list, alpha_l=1.0):

    encoder_model, atom_pred_decoder_model, chi_pred_decoder_model, both_pred_decoder_model = model_list
    optimizer_encoder, optimizer_dec_pred_atoms, optimizer_dec_pred_chi, optimizer_dec_pred_both  = optimizer_list

    encoder_model.train()

    if (args.to_predict == 'atom_type'):
        atom_pred_decoder_model.train()
    elif (args.to_predict == 'chirality'):
        chi_pred_decoder_model.train()
    elif (args.to_predict == 'both_one_decoder'):
        both_pred_decoder_model.train()
    elif (args.to_predict == 'both_two_decoder'):
        atom_pred_decoder_model.train()
        chi_pred_decoder_model.train()

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):

        batch = batch.to(device)
        smiles = [smiles_list[i] for i in batch.id]

        #print(batch.batch)
        node_rep = encoder_model(batch.x, batch.edge_index, batch.edge_attr)

        if (args.decoder == 'gnn'):
            pred_atom = atom_pred_decoder_model(node_rep, batch)
            pred_chi = chi_pred_decoder_model(node_rep, batch)
            pred_both = both_pred_decoder_model(node_rep, batch)
        if (args.decoder == 'mlp'):
            pred_atom = atom_pred_decoder_model(node_rep)
            pred_chi = chi_pred_decoder_model(node_rep)
            pred_both = both_pred_decoder_model(node_rep)

        masked_node_indices_atom = batch.masked_atom_indices_atom
        masked_node_indices_chi = batch.masked_atom_indices_chi
        label_atom = batch.node_attr_label
        label_chi = batch.node_attr_chi_label

        if (args.error_func == 'ce'):
            criterion = nn.CrossEntropyLoss()
        elif (args.error_func == 'mse'):
            criterion = nn.MSELoss()
        elif (args.error_func == 'sce'):
            criterion = partial(sce_loss, alpha=alpha_l)

        if (args.to_predict == 'atom_type'):
            node_loss_type = criterion(pred_atom.double()[masked_node_indices_atom], torch.Tensor.double(label_atom))
            node_loss = node_loss_type
        elif (args.to_predict == 'chirality'):
            node_loss_chi = criterion(pred_chi.double()[masked_node_indices_chi], torch.Tensor.double(label_chi))
            node_loss = node_loss_chi
        elif (args.to_predict == 'both_one_decoder'):
            node_loss_type = criterion(pred_both.double()[masked_node_indices_atom][:,:119], torch.Tensor.double(label_atom))
            node_loss_chi = criterion(pred_both.double()[masked_node_indices_chi][:,119:], torch.Tensor.double(label_chi))
            node_loss = (node_loss_type + node_loss_chi).double()
        elif (args.to_predict == 'both_two_decoder'):
            node_loss_type = criterion(pred_both.double()[masked_node_indices_atom][:,:119], torch.Tensor.double(label_atom))
            node_loss_chi = criterion(pred_both.double()[masked_node_indices_chi][:,119:], torch.Tensor.double(label_chi))
            node_loss = node_loss_type + node_loss_chi

        fingerprint_list = []

        embedding = global_mean_pool(node_rep, batch.batch)
        fingerprint_loss = 0

        for i in range(len(embedding)):
            mol = Chem.RDKFingerprint(Chem.MolFromSmiles(smiles[i]))
            for j in range(len(fingerprint_list)):
                finger_sim = DataStructs.FingerprintSimilarity(mol, fingerprint_list[j])
                emb_sim = (embedding[i].dot(embedding[j])) / (linalg.norm(embedding[i]) * linalg.norm(embedding[j]))
                fingerprint_loss += (finger_sim - emb_sim)**2

            fingerprint_list.append(mol)

        sim_loss = torch.sqrt(fingerprint_loss)
        full_loss = args.beta * sim_loss + (1-args.beta) * node_loss
        
        optimizer_encoder.zero_grad()

        if (args.to_predict == 'atom_type'):
            optimizer_dec_pred_atoms.zero_grad()
            full_loss.backward()
            optimizer_dec_pred_atoms.step()
        elif (args.to_predict == 'chirality'):
            optimizer_dec_pred_chi.zero_grad()
            full_loss.backward()
            optimizer_dec_pred_chi.step()
        elif (args.to_predict == 'both_one_decoder'):
            optimizer_dec_pred_both.zero_grad()
            full_loss.backward()
            optimizer_dec_pred_both.step()
        elif (args.to_predict == 'both_two_decoder'):
            optimizer_dec_pred_atoms.zero_grad()
            optimizer_dec_pred_chi.zero_grad()
            full_loss.backward()
            optimizer_dec_pred_atoms.step()
            optimizer_dec_pred_chi.step()

        optimizer_encoder.step()

    #torch.cuda.empty_cache()


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='input batch size for training (default: 256)')
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
    parser.add_argument('--dataset', type=str, default='zinc_standard_agent',
                        help='root directory of dataset. For now, only classification.')
    parser.add_argument('--gnn_type', type=str, default="gin")
    parser.add_argument('--output_model_file', type=str, default='encoder',
                        help='filename to output the pre-trained model')
    parser.add_argument('--num_workers', type=int, default=8, help='number of workers for dataset loading')
    parser.add_argument("--hidden_size", type=int, default=300, help='hidden size')
    parser.add_argument("--latent_size", type=int, default=56, help='latent size')
    parser.add_argument('--seed', type=int, default=42, help = "Seed for splitting the dataset.")
    parser.add_argument('--beta', type = float, default=0.5, help = "loss hyperparamter")

    parser.add_argument('--error_func', type = str, default='ce', help='sce, mse, ce')
    parser.add_argument('--decoder', type = str, default='mlp', help='gnn or mlp')
    parser.add_argument('--motif_to_mask_percent', type = float, default='0.15')
    parser.add_argument('--node_to_mask_percent', type = float, default='1')
    parser.add_argument('--mask_strat', type = str, default='node', help='node-wise masking or element-wise masking')
    parser.add_argument('--to_predict', type = str, default='atom_type', help='atom_type, chirality, both_one_decoder, both_two_decoder')

    args = parser.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    print(device)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    dataset = MoleculeDataset('dataset/' + args.dataset, dataset=args.dataset)
    smiles_list = pd.read_csv('dataset/' + args.dataset + '/processed/smiles.csv', header=None)[0].tolist()

    loader = DataLoaderMaskingPred(dataset, smiles_list, batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers, motif_mask_rate=args.motif_to_mask_percent, intermotif_mask_rate=args.node_to_mask_percent, masking_strategy=args.mask_strat, mask_edge=0)

    encoder_model = GNN(args.num_layer, args.emb_dim, device, JK=args.JK, drop_ratio=args.dropout_ratio, gnn_type=args.gnn_type).to(device)

    NUM_NODE_ATTR = 119
    NUM_CHIRALITY_ATTR = 3
    
    if (args.decoder == 'gnn'):
        atom_pred_decoder_model = GNNDecoder(args.emb_dim, NUM_NODE_ATTR, JK=args.JK, gnn_type=args.gnn_type).to(device)
        chi_pred_decoder_model = GNNDecoder(args.emb_dim, NUM_CHIRALITY_ATTR, JK=args.JK, gnn_type=args.gnn_type).to(device)
        both_pred_decoder_model = GNNDecoder(args.emb_dim, NUM_NODE_ATTR+NUM_CHIRALITY_ATTR, JK=args.JK, gnn_type=args.gnn_type).to(device)
    elif (args.decoder == 'mlp'):
        atom_pred_decoder_model = torch.nn.Linear(args.emb_dim, NUM_NODE_ATTR).to(device)
        chi_pred_decoder_model = torch.nn.Linear(args.emb_dim, NUM_CHIRALITY_ATTR).to(device)
        both_pred_decoder_model = torch.nn.Linear(args.emb_dim, NUM_NODE_ATTR+NUM_CHIRALITY_ATTR).to(device)

    model_list = [encoder_model, atom_pred_decoder_model, chi_pred_decoder_model, both_pred_decoder_model]

    optimizer_encoder = optim.Adam(encoder_model.parameters(), lr=args.lr, weight_decay=args.decay)
    optimizer_dec_pred_atoms = optim.Adam(atom_pred_decoder_model.parameters(), lr=args.lr, weight_decay=args.decay)
    optimizer_dec_pred_chi = optim.Adam(chi_pred_decoder_model.parameters(), lr=args.lr, weight_decay=args.decay)
    optimizer_dec_pred_both = optim.Adam(both_pred_decoder_model.parameters(), lr=args.lr, weight_decay=args.decay)

    optimizer_list = [optimizer_encoder, optimizer_dec_pred_atoms, optimizer_dec_pred_chi, optimizer_dec_pred_both]

    for epoch in range(1, args.epochs + 1):
        print("====epoch " + str(epoch))

        train(args, model_list, loader, optimizer_list, device, smiles_list)

        torch.save(encoder_model.state_dict(), 'saved_model/' + args.output_model_file + '.pth')

if __name__ == "__main__":
    main()
