import joblib
import pandas as pd
import torch
from scipy.sparse import dok_matrix

from data.utils import load_data_torch, process_prot_edge
from src.utils import *
import pickle
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors


def prepare_data(mono=False, drug_dim=512, sp_rate=0.9):
    if mono:
        raise NotImplementedError
    data = dict()
    drug_info, ppi, ddi = joblib.load('./data/tip_dataset_v1.joblib')
    drug_info['mols'] = drug_info['smiles'].apply(lambda s: Chem.MolFromSmiles(s))

    drug_info = drug_info[pd.notna(drug_info['mols'])]  # dropping 4 drugs that don't have valid smiles
    ddi = ddi[ddi.drug1.isin(drug_info.index) & ddi.drug2.isin(drug_info.index)]

    print('building drug features...')
    drug_info['fp'] = drug_info['mols'].apply(lambda m: rdMolDescriptors.GetMorganFingerprintAsBitVect(m, 2, nBits=drug_dim))
    data['d_feat'] = torch.Tensor([list(x) for x in drug_info['fp'].values])
    drug_num = len(drug_info)

    prots = set(ppi['source_entity_id']) | set(ppi['target_entity_id'])
    for idx, row in drug_info.iterrows():
        prots.update(row['targets'] or [])
    prots = sorted(list(prots))
    protein_num = len(prots)

    # prot features - one hot sparse tensor
    ind = torch.LongTensor([range(protein_num), range(protein_num)])
    val = torch.FloatTensor([1] * protein_num)
    protein_feat = torch.sparse.FloatTensor(ind, val,
                                            torch.Size([protein_num, protein_num]))
    data['p_feat'] = protein_feat
    data['n_drug'] = data['d_feat'].shape[0]
    data['n_prot'] = data['p_feat'].shape[0]
    data['n_drug_feat'] = data['d_feat'].shape[1]
    data['d_norm'] = torch.ones(data['n_drug_feat'])

    # internal id to pomicell id
    piid2ppid = {i:ppid for i,ppid in enumerate(prots)}
    diid2dpid = {i:dpid for i,dpid in enumerate(drug_info.index)}
    dpid2diid = {v:k for k,v in diid2dpid.items()}
    ppid2piid = {v:k for k,v in piid2ppid.items()}

    dd_et_list = sorted(list(set(ddi.Y)))
    dd_adj_list = []
    # sum_adj = sp.csr_matrix((drug_num, drug_num))
    for i in dd_et_list:
        ddi_subset = ddi[ddi.Y == i][['drug1', 'drug2']]
        dct = {}
        for idx, (d1, d2) in ddi_subset.iterrows():
            row = dpid2diid[int(d1)]
            col = dpid2diid[int(d2)]
            dct[(row, col)] = 1
            dct[(col, row)] = 1

        smat = dok_matrix((drug_num, drug_num))
        smat._update(dct)
        adj = smat.tocsr()
        # adj = sp.load_npz(''.join([path, 'sym_adj/drug-sparse-adj/type_', str(i), '.npz']))

        dd_adj_list.append(sp.triu(adj).tocsr())
        # sum_adj += adj
    data['dd_adj_list'] = dd_adj_list

    num = [0]
    edge_index_list = []
    edge_type_list = []

    n_et = len(dd_et_list)
    print(n_et, ' polypharmacy side effects')
    data['n_dd_et'] = n_et

    for i in range(n_et):
        # pos samples
        adj = dd_adj_list[i].tocoo()
        edge_index_list.append(torch.tensor([adj.row, adj.col], dtype=torch.long))
        edge_type_list.append(torch.tensor([i] * adj.nnz, dtype=torch.long))
        num.append(num[-1] + adj.nnz)

    data['dd_edge_index'] = edge_index_list
    data['dd_edge_type'] = edge_type_list
    data['dd_edge_type_num'] = num
    data['dd_y_pos'] = torch.ones(num[-1])
    data['dd_y_neg'] = torch.zeros(num[-1])

    ppi_subset = ppi[ppi['confidence']>=0.89][['source_entity_id', 'target_entity_id']]
    dct = {}
    for idx, (p1, p2) in ppi_subset.iterrows():
        row = ppid2piid[p1]
        col = ppid2piid[p2]
        dct[(row, col)] = 1
        dct[(col, row)] = 1

    smat = dok_matrix((protein_num, protein_num))
    smat._update(dct)
    data['pp_adj'] = smat.tocoo()

    dct = {}
    for d, (p_list,) in drug_info[['targets']].iterrows():
        for p in p_list or []:
            row = dpid2diid[d]
            col = ppid2piid[p]
            dct[(row, col)] = 1

    smat = dok_matrix((drug_num, protein_num))
    smat._update(dct)
    data['dp_adj'] = smat.tocoo()

    data['dd_train_idx'], data['dd_train_et'], data['dd_train_range'], data['dd_test_idx'], data['dd_test_et'], \
            data['dd_test_range'], data['test_drugs'] = split(data['dd_edge_index'], drug_info, dpid2diid, stratified=True)
    data['pp_train_indices'], data['pp_test_indices'] = process_prot_edge(data['pp_adj'])
    # ###################################
    # dp_edge_index and range index
    # ###################################
    data['dp_edge_index'] = np.array([data['dp_adj'].col - 1, data['dp_adj'].row - 1])
    count_drug = np.zeros(data['n_drug'], dtype=np.int32)
    for i in data['dp_edge_index'][1, :]:
        count_drug[i] += 1
    range_list = []
    start = 0
    end = 0
    for i in count_drug:
        end += i
        range_list.append((start, end))
        start = end
    data['dp_edge_index'] = torch.from_numpy(data['dp_edge_index'] + np.array([[0], [data['n_prot']]]))
    data['dp_range_list'] = torch.Tensor(range_list)

    return data



def split(raw_edge_list, drug_info, dpid2diid, stratified=True):
    train_list = []
    test_list = []
    train_label_list = []
    test_label_list = []

    if stratified:
        train_drugs = [np.int64(dpid2diid[x]) for x in drug_info[drug_info.split == 'train'].index]
        test_drugs = [np.int64(dpid2diid[x]) for x in drug_info[drug_info.split == 'test'].index]

        for i, idx in enumerate(raw_edge_list):

            test_mask = np.isin(idx[0,:], test_drugs) | np.isin(idx[1,:], test_drugs)
            train_mask = np.isin(idx[0,:], train_drugs) & np.isin(idx[1,:], train_drugs)
            train_set = train_mask.nonzero()[0]
            test_set = test_mask.nonzero()[0]

            print(f'train_set: {train_set.size}, test set: {test_set.size}, '
                  f'train drugs: {len(train_drugs)}, test drugs: {len(test_drugs)}')

            edge_count = idx.shape[1]
            assert edge_count * 0.85 <= train_set.size + test_set.size <= edge_count, \
                "expecting most edges to be either in train set or in test set"

            train_list.append(idx[:, train_set])
            test_list.append(idx[:, test_set])

            train_label_list.append(torch.ones(2 * train_set.size, dtype=torch.long) * i)
            test_label_list.append(torch.ones(2 * test_set.size, dtype=torch.long) * i)

    else:
        for i, idx in enumerate(raw_edge_list):
            train_mask = np.random.binomial(1, 0.9, idx.shape[1])
            test_mask = 1 - train_mask
            train_set = train_mask.nonzero()[0]
            test_set = test_mask.nonzero()[0]

            train_list.append(idx[:, train_set])
            test_list.append(idx[:, test_set])

            train_label_list.append(torch.ones(2 * train_set.size, dtype=torch.long) * i)
            test_label_list.append(torch.ones(2 * test_set.size, dtype=torch.long) * i)

    train_list = [to_bidirection(idx) for idx in train_list]
    test_list = [to_bidirection(idx) for idx in test_list]

    train_range = get_range_list(train_list)
    test_range = get_range_list(test_list)

    train_edge_idx = torch.cat(train_list, dim=1)
    test_edge_idx = torch.cat(test_list, dim=1)

    train_et = torch.cat(train_label_list)
    test_et = torch.cat(test_label_list)

    if stratified:
        return train_edge_idx, train_et, train_range, test_edge_idx, test_et, test_range, test_drugs
    else:
        return train_edge_idx, train_et, train_range, test_edge_idx, test_et, test_range

if __name__ == '__main__':
    data = prepare_data()

    out_file = './data/data_dict.pkl'
    with open(out_file, 'wb') as f:
        pickle.dump(data, f)

    print("Data has been prepared and is ready to use --> ./data/data_dict.pkl")
