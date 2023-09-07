from itertools import count
from re import S
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import numpy as np
import pandas as pd
import random
import math
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem
from torch_geometric.data import Batch
from torch_geometric.data import Data
from get_vocab import get_motifs, get_motifs_edges
from loader import graph_data_obj_to_mol_simple
from chemutils import brics_decomp, get_clique_mol


# allowable node and edge features
allowable_features = {
    'possible_atomic_num_list': list(range(1, 119)),
    'possible_formal_charge_list': [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
    'possible_chirality_list': [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_OTHER
    ],
    'possible_hybridization_list': [
        Chem.rdchem.HybridizationType.S,
        Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2, Chem.rdchem.HybridizationType.UNSPECIFIED
    ],
    'possible_numH_list': [0, 1, 2, 3, 4, 5, 6, 7, 8],
    'possible_implicit_valence_list': [0, 1, 2, 3, 4, 5, 6],
    'possible_degree_list': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'possible_bonds': [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC
    ],
    'possible_bond_dirs': [  # only for double bond stereo information
        Chem.rdchem.BondDir.NONE,
        Chem.rdchem.BondDir.ENDUPRIGHT,
        Chem.rdchem.BondDir.ENDDOWNRIGHT
    ]
}


def mol_to_graph_data_obj_simple(mol):
    """
    Converts rdkit mol object to graph Data object required by the pytorch
    geometric package. NB: Uses simplified atom and bond features, and represent
    as indices
    :param mol: rdkit mol object
    :return: graph data object with the attributes: x, edge_index, edge_attr
    """
    # atoms

    index = []

    num_atom_features = 2  # atom type,  chirality tag
    atom_features_list = []
    for atom in mol.GetAtoms():
        index.append(atom.GetIdx())
        atom_feature = [allowable_features['possible_atomic_num_list'].index(
            atom.GetAtomicNum())] + [allowable_features[
                                         'possible_chirality_list'].index(atom.GetChiralTag())]
        atom_features_list.append(atom_feature)
    x = torch.tensor(np.array(atom_features_list), dtype=torch.long)

    # bonds
    num_bond_features = 2  # bond type, bond direction
    if len(mol.GetBonds()) > 0:  # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_feature = [allowable_features['possible_bonds'].index(
                bond.GetBondType())] + [allowable_features[
                'possible_bond_dirs'].index(
                bond.GetBondDir())]
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = torch.tensor(np.array(edge_features_list),
                                 dtype=torch.long)
    else:  # mol has no bonds
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    return data, index


def moltree_to_graph_data(batch):
    graph_data_batch = []
    for mol in batch:
        graph_data_batch.append(mol_to_graph_data_obj_simple(Chem.MolFromSmiles(mol)))
    new_batch = Batch().from_data_list(graph_data_batch)
    return new_batch


class DataLoaderMasking(torch.utils.data.DataLoader):
    r"""Data loader which merges data objects from a
    :class:`torch_geometric.data.dataset` to a mini-batch.
    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How may samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch (default: :obj:`True`)
    """

    def __init__(self, dataset, batch_size=1, shuffle=True, **kwargs):
        super(DataLoaderMasking, self).__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=lambda data_list: BatchMasking.from_data_list(data_list),
            **kwargs)


class DataLoaderMaskingPred(torch.utils.data.DataLoader):
    r"""Data loader which merges data objects from a
    :class:`torch_geometric.data.dataset` to a mini-batch.
    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How may samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch (default: :obj:`True`)
    """

    def __init__(self, dataset, smiles_list, batch_size=1, shuffle=True, motif_mask_rate=0.25, intermotif_mask_rate=1, masking_strategy='node', mask_edge=0.0, **kwargs):
        self._transform = MaskAtom(num_atom_type = 119, num_edge_type = 5, mask_rate = motif_mask_rate, inter_mask_rate = intermotif_mask_rate, mask_strat = masking_strategy, mask_edge=mask_edge)
        self.smiles_list = smiles_list
        super(DataLoaderMaskingPred, self).__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=self.collate_fn,
            **kwargs)
    
    def collate_fn(self, batches):
        batchs = [self._transform(x, self.smiles_list[x.id]) for x in batches]
        return BatchMasking.from_data_list(batchs)


class BatchMasking(Data):
    r"""A plain old python object modeling a batch of graphs as one big
    (dicconnected) graph. With :class:`torch_geometric.data.Data` being the
    base class, all its methods can also be used here.
    In addition, single graphs can be reconstructed via the assignment vector
    :obj:`batch`, which maps each node to its respective graph identifier.
    """

    def __init__(self, batch=None, **kwargs):
        super(BatchMasking, self).__init__(**kwargs)
        self.batch = batch

    @staticmethod
    def from_data_list(data_list):
        r"""Constructs a batch object from a python list holding
        :class:`torch_geometric.data.Data` objects.
        The assignment vector :obj:`batch` is created on the fly."""
        keys = [set(data.keys) for data in data_list]
        keys = list(set.union(*keys))
        assert 'batch' not in keys

        batch = BatchMasking()

        for key in keys:
            batch[key] = []
        batch.batch = []

        cumsum_node = 0
        cumsum_edge = 0

        for i, data in enumerate(data_list):
            num_nodes = data.num_nodes
            batch.batch.append(torch.full((num_nodes, ), i, dtype=torch.long))
            for key in data.keys:
                item = data[key]
                if key in ['edge_index', 'masked_atom_indices']:
                    item = item + cumsum_node
                elif key  == 'connected_edge_indices':
                    item = item + cumsum_edge
                batch[key].append(item)

            cumsum_node += num_nodes
            cumsum_edge += data.edge_index.shape[1]

        for key in keys:
            batch[key] = torch.cat(
                batch[key], dim=data_list[0].__cat_dim__(key, batch[key][0]))
        batch.batch = torch.cat(batch.batch, dim=-1)
        return batch.contiguous()

    def cumsum(self, key, item):
        r"""If :obj:`True`, the attribute :obj:`key` with content :obj:`item`
        should be added up cumulatively before concatenated together.
        .. note::
            This method is for internal use only, and should only be overridden
            if the batch concatenation process is corrupted for a specific data
            attribute.
        """
        return key in ['edge_index', 'face', 'masked_atom_indices', 'connected_edge_indices']

    @property
    def num_graphs(self):
        """Returns the number of graphs in the batch."""
        return self.batch[-1].item() + 1

class MaskAtom:
    def __init__(self, num_atom_type, num_edge_type, mask_rate, inter_mask_rate, mask_strat, mask_edge):
        """
        Randomly masks an atom, and optionally masks edges connecting to it.
        The mask atom type index is num_possible_atom_type
        The mask edge type index in num_possible_edge_type
        :param num_atom_type:
        :param num_edge_type:
        :param mask_rate: % of atoms/motifs to be masked
        :param inter_mask_rate: % of atoms within motif to be masked
        :param mask_strat: node or element-wise masking
        :param mask_edge: If True, also mask the edges that connect to the
        masked atoms
        """
        self.num_atom_type = num_atom_type
        self.num_edge_type = num_edge_type
        self.mask_rate = mask_rate
        self.mask_edge = mask_edge
        
        self.num_chirality_tag = 3
        self.num_bond_direction = 3 

        self.offset = 0

        self.inter_mask_rate = inter_mask_rate
        self.mask_strat = mask_strat

    def __call__(self, data, smiles, masked_atom_indices=None):
        """
        :param data: pytorch geometric data object. Assume that the edge
        ordering is the default pytorch geometric ordering, where the two
        directions of a single edge occur in pairs.
        Eg. data.edge_index = tensor([[0, 1, 1, 2, 2, 3],
                                     [1, 0, 2, 1, 3, 2]])
        :param masked_atom_indices: If None, then randomly samples num_atoms
        * mask rate number of atom indices
        Otherwise a list of atom idx that sets the atoms to be masked (for
        debugging only)
        :return: None, Creates new attributes in original data object:
        data.mask_node_idx
        data.mask_node_label
        data.mask_edge_idx
        data.mask_edge_label
        """

        mol = Chem.MolFromSmiles(smiles)
        motifs = get_motifs(mol)
        grouping = torch.tensor([0] * len(data.x[:, 1]))

        num_atoms = data.x.size()[0]
        sample_size = int(num_atoms * self.mask_rate + 1)

        valid_motifs = []
        if len(motifs) != 1:
            for motif in motifs:
                for atom in mol.GetAtoms():
                    if atom.GetIdx() in motif:
                        if (inter_motif_proximity(motif, [atom], []) > 5):
                            break
                valid_motifs.append(motif)

        for i, x in enumerate(motifs):
            grouping[x] = i + self.offset

        self.offset += i
        
        masked_atom_indices = []

        # Select motifs according to 
        while len(masked_atom_indices) < sample_size:
            if len(valid_motifs) < 1:
                index_list = random.sample(range(num_atoms), sample_size)
                for index in index_list:
                    if index not in masked_atom_indices:
                        masked_atom_indices.append(index)
            else:
                candidate = valid_motifs[random.sample(range(0, len(valid_motifs)), 1)[0]]
                valid_motifs.remove(candidate)
                for atom_idx in candidate:
                    for i, edge in enumerate(data.edge_index[0]):
                        if atom_idx == edge:
                            for motif in valid_motifs:
                                if data.edge_index[1][i].item() in motif:
                                    valid_motifs.remove(motif)
                            
                if len(masked_atom_indices) + len(candidate) > sample_size + 0.1 * num_atoms:
                    continue
                for index in candidate:
                    masked_atom_indices.append(index)

        # random masking
        if masked_atom_indices == None:
            # sample x distinct atoms to be masked, based on mask rate. But
            # will sample at least 1 atom
            num_atoms = data.x.size()[0]
            sample_size = int(num_atoms * self.mask_rate + 1)
            masked_atom_indices = random.sample(range(num_atoms), sample_size)

        l = math.ceil(len(masked_atom_indices) * self.inter_mask_rate)

        masked_atom_indices_atom = random.sample(masked_atom_indices, l)
        masked_atom_indices_chi = random.sample(masked_atom_indices, l)

        # create mask node label by copying atom feature of mask atom
        # node-wise masking
        if (self.mask_strat == 'node'):
            mask_node_labels_list = []
            for atom_idx in masked_atom_indices_atom:
                mask_node_labels_list.append(data.x[atom_idx].view(1, -1))
            
            data.mask_node_label = torch.cat(mask_node_labels_list, dim=0)
            data.masked_atom_indices_atom = torch.tensor(masked_atom_indices_atom)
            data.masked_atom_indices_chi = torch.tensor(masked_atom_indices_atom)

            atom_type = F.one_hot(data.mask_node_label[:, 0], num_classes=self.num_atom_type).float()
            atom_chirality = F.one_hot(data.mask_node_label[:, 1], num_classes=self.num_chirality_tag).float()
            data.node_attr_label = atom_type
            data.node_attr_chi_label = atom_chirality

        # element-wise masking
        elif (self.mask_strat == 'element'):
            mask_atom_labels_list = []
            mask_chi_labels_list = []
            for atom_idx in masked_atom_indices_atom:
                mask_atom_labels_list.append(data.x[atom_idx].view(1, -1))
            for atom_idx in masked_atom_indices_chi:
                mask_chi_labels_list.append(data.x[atom_idx].view(1, -1))
            
            data.mask_atom_label = torch.cat(mask_atom_labels_list, dim=0)
            data.mask_chi_label = torch.cat(mask_chi_labels_list, dim=0)
            data.masked_atom_indices_atom = torch.tensor(masked_atom_indices_atom)
            data.masked_atom_indices_chi = torch.tensor(masked_atom_indices_chi)

            atom_type = F.one_hot(data.mask_atom_label[:, 0], num_classes=self.num_atom_type).float()
            atom_chirality = F.one_hot(data.mask_chi_label[:, 1], num_classes=self.num_chirality_tag).float()
            data.node_attr_label = atom_type
            data.node_attr_chi_label = atom_chirality

            for atom_idx in masked_atom_indices_atom:
                data.x[atom_idx] = torch.tensor([self.num_atom_type, data.x[atom_idx][1]])
            for atom_idx in masked_atom_indices_chi:
                data.x[atom_idx] = torch.tensor([data.x[atom_idx][0], 0])


        if self.mask_edge:
            # create mask edge labels by copying edge features of edges that are bonded to
            # mask atoms
            connected_edge_indices = []
            for bond_idx, (u, v) in enumerate(data.edge_index.cpu().numpy().T):
                for atom_idx in masked_atom_indices:
                    if atom_idx in set((u, v)) and \
                        bond_idx not in connected_edge_indices:
                        connected_edge_indices.append(bond_idx)

            if len(connected_edge_indices) > 0:
                # create mask edge labels by copying bond features of the bonds connected to
                # the mask atoms
                mask_edge_labels_list = []
                for bond_idx in connected_edge_indices[::2]: # because the
                    # edge ordering is such that two directions of a single
                    # edge occur in pairs, so to get the unique undirected
                    # edge indices, we take every 2nd edge index from list
                    mask_edge_labels_list.append(
                        data.edge_attr[bond_idx].view(1, -1))

                data.mask_edge_label = torch.cat(mask_edge_labels_list, dim=0)
                # modify the original bond features of the bonds connected to the mask atoms
                for bond_idx in connected_edge_indices:
                    data.edge_attr[bond_idx] = torch.tensor(
                        [self.num_edge_type, 0])

                data.connected_edge_indices = torch.tensor(
                    connected_edge_indices[::2])
            else:
                data.mask_edge_label = torch.empty((0, 2)).to(torch.int64)
                data.connected_edge_indices = torch.tensor(
                    connected_edge_indices).to(torch.int64)

            edge_type = F.one_hot(data.mask_edge_label[:, 0], num_classes=self.num_edge_type).float()
            bond_direction = F.one_hot(data.mask_edge_label[:, 1], num_classes=self.num_bond_direction).float()
            data.edge_attr_label = torch.cat((edge_type, bond_direction), dim=1)
            # data.edge_attr_label = edge_type

        return data

    def __repr__(self):
        return '{}(num_atom_type={}, num_edge_type={}, mask_rate={}, mask_edge={})'.format(
            self.__class__.__name__, self.num_atom_type, self.num_edge_type,
            self.mask_rate, self.mask_edge)

def inter_motif_proximity(target_motif, neighbors, checked):
    new_neighbors = []
    for atom in neighbors:
        for nei in atom.GetNeighbors():
            if nei.GetIdx() in checked:
                continue
            new_neighbors.append(nei)
            if nei.GetIdx() not in target_motif:
                return 1
        checked.append(atom.GetIdx())
    return inter_motif_proximity(target_motif, new_neighbors, checked) + 1