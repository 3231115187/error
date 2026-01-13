import numpy as np
import scipy.sparse as sp
import torch
from torch_geometric.utils import from_scipy_sparse_matrix
import os


class MedicalNCDataset:
    def __init__(self, data_path):
        self.name = 'medical'
        from data_loader import data_loader
        dl = data_loader(r"D:\text3")

        # 1. 节点处理
        node_features_list = []
        node_type_list = []
        num_node_types = len(dl.nodes['count'])

        for i in range(num_node_types):
            count = dl.nodes['count'][i]
            feature_matrix = dl.nodes['attr'][i]
            if feature_matrix is None:
                feature_matrix = sp.eye(count)
            node_features_list.append(feature_matrix)
            node_type_list.append(torch.full((count,), i, dtype=torch.long))

        node_feat = sp.vstack(node_features_list).astype(np.float32)
        node_feat = torch.tensor(node_feat.todense(), dtype=torch.float)
        node_type = torch.cat(node_type_list, dim=0)

        # 2. 边处理：构建多关系边索引与类型
        edges = []
        edge_types = []
        num_relations = len(dl.links['data'])

        # 遍历每种医疗关系（如患者-药物、患者-手术）
        for r_id, adj in dl.links['data'].items():
            ei, _ = from_scipy_sparse_matrix(adj)
            edges.append(ei)
            # 为该关系下的边分配对应的类型ID，用于模型索引QKV矩阵
            edge_types.append(torch.full((ei.size(1),), r_id, dtype=torch.long))

        # 3. 标签与划分处理
        num_nodes = node_feat.shape[0]
        num_classes = dl.labels_train['num_classes']
        labels = np.zeros((num_nodes, num_classes), dtype=int)

        train_idx = np.nonzero(dl.labels_train['mask'])[0]
        test_idx = np.nonzero(dl.labels_test['mask'])[0]

        val_ratio = 0.2
        split = int(train_idx.shape[0] * (1 - val_ratio))
        real_train_idx = train_idx[:split]
        val_idx = train_idx[split:]

        labels[real_train_idx] = dl.labels_train['data'][real_train_idx]
        labels[val_idx] = dl.labels_train['data'][val_idx]
        labels[test_idx] = dl.labels_test['data'][test_idx]
        labels = torch.tensor(labels.argmax(axis=1), dtype=torch.long)

        self.graph = {
            'edge_index': torch.cat(edges, dim=1),
            'edge_type': torch.cat(edge_types, dim=0),
            'node_feat': node_feat,
            'node_type': node_type,
            'num_nodes': num_nodes,
            'num_relations': num_relations  # 记录关系总数供模型初始化
        }
        self.label = labels
        self.num_classes = num_classes
        self.train_idx = torch.tensor(real_train_idx, dtype=torch.long)
        self.valid_idx = torch.tensor(val_idx, dtype=torch.long)
        self.test_idx = torch.tensor(test_idx, dtype=torch.long)

    def get_idx_split(self, split_type='random', train_prop=0.5, valid_prop=0.25):
        return {'train': self.train_idx, 'valid': self.valid_idx, 'test': self.test_idx}

    def __getitem__(self, idx):
        return self.graph, self.label

    def __len__(self):
        return 1


def load_medical_dataset(data_dir, name):
    return MedicalNCDataset(data_dir)