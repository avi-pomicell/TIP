import numpy as np
import torch


def negative_sampling(pos_edge_index, num_nodes, test_drugs=None):
    idx = (pos_edge_index[0] * num_nodes + pos_edge_index[1])
    idx = idx.to(torch.device('cpu'))

    perm = torch.tensor(np.random.choice(num_nodes ** 2, idx.size(0)))

    def get_mask_rest():
        if test_drugs is not None:
            mask_pos = np.isin(perm, idx)
            row, col = perm // num_nodes, perm % num_nodes
            mask_test = np.isin(row, test_drugs) | np.isin(col, test_drugs)
            mask = torch.from_numpy((mask_pos | mask_test).astype(np.uint8))
        else:
            mask = torch.from_numpy(np.isin(perm, idx).astype(np.uint8))
        rest = mask.nonzero().view(-1)
        return rest, mask

    rest, mask = get_mask_rest()

    while rest.numel() > 0:  # pragma: no cover
        tmp = torch.tensor(np.random.choice(num_nodes ** 2, rest.size(0)))
        perm[rest] = tmp
        rest, mask = get_mask_rest()

    row, col = perm / num_nodes, perm % num_nodes
    return torch.stack([row, col], dim=0).long().to(pos_edge_index.device)


def typed_negative_sampling(pos_edge_index, num_nodes, range_list, test_drugs=None):
    tmp = []
    for start, end in range_list:
        tmp.append(negative_sampling(pos_edge_index[:, start: end], num_nodes, test_drugs))
    return torch.cat(tmp, dim=1)
