import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax
from vq import DynamicResidualVectorQuant


class RelationAwareLayer(MessagePassing):
    def __init__(self, in_channels, out_channels, num_relations, heads=4, dropout=0.5, **kwargs):
        super().__init__(aggr='add', node_dim=0, **kwargs)
        self.num_relations = num_relations
        self.heads, self.out_channels = heads, out_channels

        # 为每种关系定义独立的 QKV 矩阵
        self.w_q = nn.ModuleList([nn.Linear(in_channels, heads * out_channels) for _ in range(num_relations)])
        self.w_k = nn.ModuleList([nn.Linear(in_channels, heads * out_channels) for _ in range(num_relations)])
        self.w_v = nn.ModuleList([nn.Linear(in_channels, heads * out_channels) for _ in range(num_relations)])

        self.scale = out_channels ** -0.5
        self.bias = nn.Parameter(torch.zeros(heads * out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        for q, k, v in zip(self.w_q, self.w_k, self.w_v):
            nn.init.xavier_uniform_(q.weight);
            nn.init.xavier_uniform_(k.weight);
            nn.init.xavier_uniform_(v.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x, edge_index, edge_type, return_attention=False):
        out = torch.zeros((x.size(0), self.heads * self.out_channels), device=x.device)
        att_data = []
        for r in range(self.num_relations):
            mask = (edge_type == r)
            if not mask.any(): continue
            rel_ei = edge_index[:, mask]
            q, k, v = self.w_q[r](x).view(-1, self.heads, self.out_channels), \
                self.w_k[r](x).view(-1, self.heads, self.out_channels), \
                self.w_v[r](x).view(-1, self.heads, self.out_channels)

            res = self.propagate(rel_ei, q=q, k=k, v=v, return_attention=return_attention)
            if return_attention:
                res, alpha = res
                att_data.append({'ei': rel_ei, 'alpha': alpha, 'type': r})
            out += res.view(-1, self.heads * self.out_channels)
        return (out + self.bias, att_data) if return_attention else out + self.bias

    def message(self, q_i, k_j, v_j, index, ptr, size_i, return_attention):
        score = (q_i * k_j).sum(dim=-1) * self.scale
        alpha = softmax(score, index, ptr, size_i)
        if return_attention: return v_j * alpha.unsqueeze(-1), alpha
        return v_j * F.dropout(alpha, p=0.5, training=self.training).unsqueeze(-1)

    def aggregate(self, inputs, index, ptr=None, dim_size=None, return_attention=False):
        if return_attention:
            out, alpha = inputs
            return super().aggregate(out, index, ptr, dim_size), alpha
        return super().aggregate(inputs, index, ptr, dim_size)


class GNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, local_layers=3, heads=4, num_relations=5,
                 num_node_types=10, **kwargs):
        super().__init__()
        self.lin_in = nn.Linear(in_channels, hidden_channels * heads)
        self.convs = nn.ModuleList(
            [RelationAwareLayer(hidden_channels * heads, hidden_channels, num_relations, heads) for _ in
             range(local_layers)])
        self.lins = nn.ModuleList(
            [nn.Linear(hidden_channels * heads, hidden_channels * heads) for _ in range(local_layers)])
        self.vqs = nn.ModuleList(
            [DynamicResidualVectorQuant(hidden_channels * heads, 64, num_node_types=num_node_types) for _ in
             range(local_layers)])
        self.classifier = nn.Sequential(nn.Linear(local_layers * hidden_channels * heads, hidden_channels * 2),
                                        nn.ReLU(), nn.Linear(hidden_channels * 2, out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        self.lin_in.reset_parameters()
        for c, l, v in zip(self.convs, self.lins, self.vqs):
            c.reset_parameters();
            l.reset_parameters()
            if hasattr(v, 'reset_parameters'): v.reset_parameters()
        for m in self.classifier:
            if isinstance(m, nn.Linear): torch.nn.init.xavier_uniform_(m.weight)

    def forward(self, x, edge_index, edge_type, node_type, visualize=False, vq_noise_scale=0.0):
        x = self.lin_in(F.dropout(x, p=0.2, training=self.training))
        projected, quantized_list, att_record, vq_idxs = x.clone(), [], None, []
        total_loss = 0
        for i, (conv, vq) in enumerate(zip(self.convs, self.vqs)):
            if visualize and i == len(self.convs) - 1:
                x_conv, att_record = conv(x, edge_index, edge_type, return_attention=True)
            else:
                x_conv = conv(x, edge_index, edge_type)
            x = F.relu(x_conv + self.lins[i](x))
            q, idx, loss, _ = vq(x, node_type, original_feat=projected, training=self.training,
                                 noise_scale=vq_noise_scale)
            quantized_list.append(q);
            total_loss += loss
            if visualize: vq_idxs.append(idx)
        out = self.classifier(torch.cat(quantized_list, dim=1))
        return (out, att_record, vq_idxs) if visualize else (out, None, None, total_loss)