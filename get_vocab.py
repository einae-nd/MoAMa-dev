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

    vocab = []

    for smiles in data:
        
        mol = Chem.MolFromSmiles(smiles)
        Chem.SanitizeMol(mol)

        motifs, edges = brics_decomp(mol)

        motif_mols = []
        for x in motifs:
            motif = get_clique_mol(mol, x)
            motif = Chem.MolToSmiles(motif)
            motif_mols.append(motif)

        vocab.append(motif_mols)

    return vocab

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--ncpu', type=int, default=1)
    args = parser.parse_args()

    data = [mol for line in sys.stdin for mol in line.split()[:2]]
    data = list(set(data))

    batch_size = len(data) // args.ncpu + 1
    batches = [data[i : i + batch_size] for i in range(0, len(data), batch_size)]

    pool = Pool(args.ncpu)
    vocab_list = pool.map(process, batches)
    vocab = [(x,y) for vocab in vocab_list for x,y in vocab]
    raw_vocab = vocab
    vocab = list(set(vocab))

    #count = pd.Series(raw_vocab).value_counts()
    #mol_name = count.index
    
    #motif_length_frequency = []

    #for elem in mol_name:
    #    m = Chem.MolFromSmiles(elem[0])
    #    x = m.GetNumAtoms()
    #    try:
    #         motif_length_frequency[x]
    #    except:
    #        while (len(motif_length_frequency) - 1 < x):
    #             motif_length_frequency.append(0)

    #    motif_length_frequency[x] += count[elem]

    #print(motif_length_frequency)
    #count.to_csv('count.csv')
    #pd.DataFrame(motif_length_frequency).to_csv('motif_count.csv')
    
    for x,y in sorted(vocab):
        print(x, y)