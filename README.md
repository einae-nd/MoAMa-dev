# MoAMa

A PyTorch implementation of "[Motif-aware Attribute Masking for Molecular Graph Pre-training]".

## Dependencies
+ pandas==1.5.2
+ python==3.10.4
+ rdkit==2022.9.3
+ torch==1.13.1
+ torch-cluster==1.6.0
+ torch-geometric==2.2.0
+ torch-scatter==2.1.0
+ torch-sparse==0.6.16

To install all dependencies:
```
pip install -r requirements.txt
```

## Usage

To download datasets

```
wget http://snap.stanford.edu/gnn-pretrain/data/chem_dataset.zip
unzip chem_dataset.zip -d transferLearning_MoleculeNet_PPI/chem/
```

To pre-train:
```
python pretrain.py
```

To fine-tune and evaluate:
```
python finetune.py --dataset <dataset_name>
```
