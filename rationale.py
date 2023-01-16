import torch
import torch.nn as nn
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set

class rationale_motif_pred(torch.nn.Module):

    def __init__(self, device, hidden_size , num_tasks, num_layer = 5, emb_dim = 300, gnn_type = 'gin', dropout_ratio = 0.5, gamma = 0.4, use_linear_predictor=False):
        '''
            num_tasks (int): number of labels to be predicted
        '''

        super(rationale_motif_pred, self).__init__()

        self.num_layer = num_layer
        self.drop_ratio = dropout_ratio
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks
        self.gamma  = gamma
        self.device = device
        self.hidden_size = hidden_size

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.pool = global_mean_pool
        self.pred_loss = nn.MSELoss(size_average=False)
        self.rationale_pred_linear = torch.nn.Linear(self.emb_dim, num_tasks)
        self.W = nn.Linear(hidden_size, hidden_size)
 
    def forward(self, graph_encoder, batched_data, node_rep, motif_node_rep):

        # Create mask for creating rationale motif list
        mask_list = []
        for motif_list in motif_node_rep:
            mask_list.append(self.rationale_pred_linear(motif_list))

        h_r = []
        h_aug = []
        motif_pred_list = []
        motif_true_list = []

        for i in range(len(motif_node_rep)):

            graph_motif_rep = mask_list[i] * motif_node_rep[i]
            index = torch.argmax(mask_list[i])
            motif_true = torch.reshape(graph_motif_rep[index], (1, self.emb_dim))
            motif_true_list.append(motif_true)
            motif_removed = torch.cat([graph_motif_rep[0:index], graph_motif_rep[index+1:]])

            # Rationale motif prediction
            graph_motif_rep = self.pool(graph_motif_rep, batch=None)
            motif_pred = nn.ReLU()(self.W(graph_motif_rep))
            motif_pred_list.append(motif_pred)

            # Augmented example
            graph_motif_pred_rep = self.pool(torch.cat([motif_removed, motif_pred]), batch=None)

            h_r.append(graph_motif_rep)
            h_aug.append(graph_motif_pred_rep)

        h_r = torch.cat(h_r)
        h_aug = torch.cat(h_aug)

        y_pred = torch.cat(motif_pred_list, dim=0)
        y_true = torch.cat(motif_true_list, dim=0)

        pred_loss = self.pred_loss(y_pred, y_true) / len(motif_node_rep)
        
        return h_r, h_aug, pred_loss
