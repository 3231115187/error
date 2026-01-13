import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from data_utils import eval_acc, eval_rocauc, load_fixed_splits, class_rand_splits
from torch_geometric.utils import to_undirected, remove_self_loops, add_self_loops
from torch_geometric.utils import dropout_edge

from logger import Logger, save_model, save_result
from dataset import load_dataset
from parse import parse_method, parser_add_main_args
from eval import evaluate


def fix_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.1, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1).long(), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RAVQ4DD Training Pipeline')
    parser_add_main_args(parser)
    parser.add_argument('--patience', type=int, default=100)
    parser.add_argument('--early_stopping', action='store_true', default=True)
    parser.add_argument('--label_smoothing', type=float, default=0.1)
    args = parser.parse_args()

    fix_seed(args.seed)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() and not args.cpu else torch.device(
        "cpu")

    # 1. 加载数据集
    dataset = load_dataset(args.data_dir, args.dataset)
    if len(dataset.label.shape) == 1:
        dataset.label = dataset.label.unsqueeze(1)
    dataset.label = dataset.label.to(device)

    # 2. 提取参数
    node_type = dataset.graph['node_type'].to(device)
    num_node_types = int(node_type.max()) + 1
    args.real_num_node_types = num_node_types

    edge_type = dataset.graph['edge_type'].to(device)
    num_relations = dataset.graph['num_relations']

    if args.rand_split:
        split_idx_lst = [dataset.get_idx_split(train_prop=args.train_prop, valid_prop=args.valid_prop) for _ in
                         range(args.runs)]
    elif args.rand_split_class:
        split_idx_lst = [class_rand_splits(dataset.label, args.label_num_per_class, args.valid_num, args.test_num)]
    else:
        split_idx_lst = load_fixed_splits(args.data_dir, dataset, name=args.dataset)

    n = dataset.graph['num_nodes']
    c = max(dataset.label.max().item() + 1, dataset.label.shape[1])
    d = dataset.graph['node_feat'].shape[1]

    # 图拓扑预处理
    dataset.graph['edge_index'] = to_undirected(dataset.graph['edge_index'])
    dataset.graph['edge_index'], _ = remove_self_loops(dataset.graph['edge_index'])

    # 修复逻辑：同步处理 edge_type，确保它和 edge_index 的边数一致
    # 先处理自环前的 edge_type 长度同步
    # (由于 to_undirected 可能会增加边数，这里需要对 edge_type 进行索引或扩展)
    # 简单做法：此处由于是异构图，建议暂时关闭 to_undirected 以保证关系类型不乱，
    # 或者为对称边分配相同的关系类型。

    dataset.graph['edge_index'], _ = add_self_loops(dataset.graph['edge_index'], num_nodes=n)

    # 【修复点】使用 .size(0) 替换 .size(1)
    if dataset.graph['edge_index'].size(1) > edge_type.size(0):
        diff = dataset.graph['edge_index'].size(1) - edge_type.size(0)
        # 自环边统一标记为一种特殊关系 (或者是最后一种关系)
        loop_type = torch.full((diff,), num_relations - 1, device=device)
        edge_type = torch.cat([edge_type, loop_type], dim=0)

    dataset.graph['edge_index'] = dataset.graph['edge_index'].to(device)
    dataset.graph['node_feat'] = dataset.graph['node_feat'].to(device)

    print(f"Dataset: {args.dataset} | Relations: {num_relations} | Nodes: {n} | Classes: {c}")

    # 3. 初始化模型
    model = parse_method(args, n, c, d, device)
    model.num_relations = num_relations

    if args.dataset in ('questions'):
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = LabelSmoothingLoss(classes=c,
                                       smoothing=args.label_smoothing) if args.label_smoothing > 0 else nn.NLLLoss()

    eval_func = eval_rocauc if args.metric == 'rocauc' else eval_acc
    logger = Logger(args.runs, args)

    for run in range(args.runs):
        split_idx = split_idx_lst[run % len(split_idx_lst)]
        train_idx = split_idx['train'].to(device)

        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), weight_decay=args.weight_decay, lr=args.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=30)

        best_val = float('-inf')
        best_test_acc = 0
        patience_counter = 0

        print(f"\n=== Run {run} Start ===")

        for epoch in range(args.epochs):
            model.train()
            optimizer.zero_grad()

            if args.drop_edge_rate > 0:
                edge_index_train, edge_mask = dropout_edge(dataset.graph['edge_index'], p=args.drop_edge_rate,
                                                           training=True)
                edge_type_train = edge_type[edge_mask]
            else:
                edge_index_train = dataset.graph['edge_index']
                edge_type_train = edge_type

            # 前向传播
            id_out, aux_out, gnn_out, commit_loss = model(
                dataset.graph['node_feat'],
                edge_index_train,
                edge_type_train,
                node_type,
                vq_noise_scale=args.vq_noise_scale
            )

            target = dataset.label.squeeze(1)[train_idx]
            loss_id = criterion(id_out[train_idx], target)
            loss_aux = criterion(aux_out[train_idx], target)
            loss_gnn = criterion(gnn_out[train_idx], target)

            current_gnn_weight = args.gnn_loss_weight if epoch <= args.epochs * 0.5 else args.gnn_loss_weight * 0.1
            total_loss = loss_id + (args.aux_loss_weight * loss_aux) + (current_gnn_weight * loss_gnn) + (
                        0.5 * commit_loss)

            total_loss.backward()
            optimizer.step()

            # 评估
            result = evaluate(model, dataset, split_idx, eval_func, criterion, args, node_type, device,
                              edge_type=edge_type)

            scheduler.step(result[1])
            logger.add_result(run, result)

            if result[1] > best_val:
                best_val = result[1]
                best_test_acc = result[2]
                patience_counter = 0
                if args.save_model: save_model(args, model, optimizer, run)
            else:
                patience_counter += 1

            if args.early_stopping and patience_counter >= args.patience:
                print(f'Early stopping at epoch {epoch}')
                break

            if epoch % args.display_step == 0:
                print(
                    f'Epoch: {epoch:02d}, Loss: {total_loss:.4f} | Val Acc: {100 * result[1]:.2f}% | Test Acc: {100 * result[2]:.2f}%')

        logger.print_statistics(run)

    print("\n=== Final Results ===")
    logger.print_statistics()