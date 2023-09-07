import sys
import argparse 
import torch
import torch_geometric
from chemutils import *
from rdkit import Chem
from multiprocessing import Pool
import pandas as pd
from chemutils import brics_decomp, get_clique_mol

def process(data):
    vocab = set()
    for line in data:
        s = line.strip("\r\n ")
        hmol = MolGraph(s)
        for node,attr in hmol.mol_tree.nodes(data=True):
            smiles = attr['smiles']
            vocab.add( attr['label'] )
            for i,s in attr['inter_label']:
                vocab.add( (smiles, s) )
    return vocab

def get_motifs(data):

    Chem.SanitizeMol(data)
    motifs, edges = brics_decomp(data)
    return motifs

def get_motifs_edges(data):

    Chem.SanitizeMol(data)
    motifs, edges = brics_decomp(data)
    return motifs, edges


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--ncpu', type=int, default=1)
    parser.add_argument('--dataset', type=str, default="zinc")
    args = parser.parse_args()

    if (args.dataset == 'zinc'):
        smiles_list = pd.read_csv('dataset/' + args.dataset + '/raw/all.txt', header=None)[0].tolist()
    else:
        smiles_list = pd.read_csv('dataset/' + args.dataset + '/processed/smiles.csv', header=None)[0].tolist()

    data = smiles_list
    #data = ['CCOC(=O)c1cncn1C(C)c2ccccc2']

    vocab = []
    for i in range(len(data)):
        mol = Chem.MolFromSmiles(data[i])
        Chem.SanitizeMol(mol)
        motifs = get_motifs(mol)
        for motif in motifs:
            vocab.append(motif)

    for i, x in enumerate(sorted(vocab)):
        print(x)