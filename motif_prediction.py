import torch
import torch.nn as nn
from mol_tree import Vocab, MolTree
from nnutils import create_var
#from dfs import Motif_Generation_dfs
from bfs import Motif_Generation_bfs
#from chemutils import enum_assemble, set_atommap, copy_edit_mol, attach_mols, atom_equal, decode_stereo
import rdkit
import rdkit.Chem as Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
import copy, math


def set_batch_nodeID(mol_batch, vocab):
    tot = 0
    for mol_tree in mol_batch:
        for node in mol_tree.nodes:
            node.idx = tot
            node.wid = vocab.get_index(node.smiles)
            tot += 1

class Motif_Prediction_sub(nn.Module):

    def __init__(self, vocab, hidden_size, device):
        super(Motif_Prediction_sub, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab.size()
        self.vocab = vocab
        self.device = device

        # GRU Weights
        #self.W_z = nn.Linear(2 * hidden_size, hidden_size)
        #self.U_r = nn.Linear(hidden_size, hidden_size, bias=False)
        #self.W_r = nn.Linear(hidden_size, hidden_size)
        #self.W_h = nn.Linear(2 * hidden_size, hidden_size)

        # Feature Aggregate Weights
        self.W = nn.Linear(hidden_size, hidden_size)
        #self.U = nn.Linear(2 * hidden_size, hidden_size)

        # Output Weights
        #self.W_o = nn.Linear(hidden_size, self.vocab_size+1)
        # bfs add one stop node
        #self.U_s = nn.Linear(hidden_size, 1)

        # Loss Functions
        self.pred_loss = nn.MSELoss(size_average=False)
        #self.stop_loss = nn.BCEWithLogitsLoss(size_average=False)

    def forward(self, mol_batch, node_rep, motif_list):

        pred_hiddens, pred_targets = [], []

        pred_vecs = torch.FloatTensor(node_rep)
        pred_scores = nn.ReLU()(self.W(pred_vecs))
        pred_targets = create_var(torch.FloatTensor(node_rep))
        pred_loss = self.pred_loss(pred_scores, pred_targets) / len(mol_batch)

        #print(pred_loss)
        return pred_loss


class Motif_Prediction(nn.Module):

    def __init__(self, vocab, hidden_size, latent_size, depth, device, order):
        super(Motif_Prediction, self).__init__()
        self.vocab = vocab
        self.hidden_size = hidden_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.depth = depth
        self.device = device
        #self.decoder = Motif_Prediction_sub(vocab, hidden_size, self.device)

        self.W = nn.Linear(hidden_size, hidden_size)
        self.pred_loss = nn.MSELoss(size_average=False)

    #def forward(self, mol_batch, node_rep, motif_list):
    def forward(self, motif_list):

        #set_batch_nodeID(mol_batch, self.vocab)

        #word_loss, topo_loss, word_acc, topo_acc = self.decoder(mol_batch, node_rep)
        
        #loss = word_loss + topo_loss
        #loss = self.decoder(mol_batch, node_rep, motif_list)

        pred_hiddens, pred_targets = [], []
        #pred_vecs = torch.FloatTensor(node_rep)
        pred_vecs = torch.FloatTensor(motif_list)
        pred_scores = nn.ReLU()(self.W(pred_vecs))
        #pred_targets = create_var(torch.FloatTensor(node_rep))
        #pred_loss = self.pred_loss(pred_scores, pred_targets) / len(mol_batch)

        #print(loss, pred_loss)
        #loss = self.decoder(mol_batch)
        #return loss, word_acc, topo_acc
        #return loss
        #return pred_loss
        return pred_scores

class Motif_Prediction1(nn.Module):

    def __init__(self, vocab, hidden_size, latent_size, depth, device, order):
        super(Motif_Prediction1, self).__init__()
        self.vocab = vocab
        self.hidden_size = hidden_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.depth = depth
        self.device = device
        #self.decoder = Motif_Prediction_sub(vocab, hidden_size, self.device)

        self.W = nn.Linear(hidden_size, hidden_size)
        self.pred_loss = nn.MSELoss(size_average=False)

    def forward(self, mol_batch, node_rep, motif_list):
    #def forward(self, motif_list):

        #set_batch_nodeID(mol_batch, self.vocab)

        #word_loss, topo_loss, word_acc, topo_acc = self.decoder(mol_batch, node_rep)
        
        #loss = word_loss + topo_loss
        #loss = self.decoder(mol_batch, node_rep, motif_list)

        pred_hiddens, pred_targets = [], []
        pred_vecs = torch.FloatTensor(node_rep)
        #pred_vecs = torch.FloatTensor(motif_list)
        pred_scores = nn.ReLU()(self.W(pred_vecs))
        pred_targets = create_var(torch.FloatTensor(node_rep))
        print(type(pred_scores), type(pred_targets))
        pred_loss = self.pred_loss(pred_scores, pred_targets) / len(mol_batch)

        #print(loss, pred_loss)
        #loss = self.decoder(mol_batch)
        #return loss, word_acc, topo_acc
        #return loss
        return pred_loss
        #return pred_scores