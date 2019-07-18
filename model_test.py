from data.utils import load_data_torch
from model.ddip import HGCN
from torch_geometric.data import Data
from torch_geometric.nn.models.autoencoder import negative_sampling
import torch as t
from torch.nn.functional import binary_cross_entropy
import numpy as np
from sklearn import metrics
import pickle
import time
from tempfile import TemporaryFile



# ##############################################
# load data

et_list = [20, 34, 38, 41, 42, 46, 55, 57, 89, 92, 99, 103, 105, 110, 125, 126, 129, 139, 147, 149, 152, 155, 157, 163, 171, 174, 180, 190, 191, 198, 210, 216, 230, 240, 246, 251, 256, 260, 262, 264, 268, 273, 300, 306, 308, 309, 321, 322, 324, 327, 336, 353, 354, 358, 359, 372, 373, 379, 382, 386, 388, 389, 390, 395, 397, 411, 412, 415, 422, 425, 427, 428, 430, 432, 433, 435, 439, 447, 450, 451, 452, 453, 454, 455, 457, 459, 461, 462, 464, 466, 468, 470, 471, 473, 481, 483, 484, 485, 490, 499, 502, 507, 511, 515, 517, 520, 525, 528, 529, 531, 535, 540, 542, 552, 553, 559, 561, 563, 566, 568, 574, 579, 580, 581, 584, 586, 589, 591, 592, 594, 602, 605, 607, 618, 620, 622, 627, 629, 634, 635, 636, 637, 639, 644, 645, 646, 651, 656, 657, 658, 662, 663, 664, 665, 666, 668, 669, 671, 672, 673, 674, 680, 682, 684, 685, 687, 691, 694, 695, 696, 697, 699, 700, 705, 706, 710, 711, 713, 714, 715, 717, 718, 720, 722, 725, 726, 727, 729, 730, 731, 735, 737, 739, 740, 742, 744, 748, 749, 752, 754, 757, 759, 760, 761, 762, 763, 767, 768, 769, 770, 771, 772, 773, 776, 781, 782, 784, 787, 788, 793, 794, 795, 797, 799, 801, 802, 803, 804, 805, 806, 809, 812, 813, 814, 815, 816, 818, 820, 821, 822, 825, 826, 827, 830, 833, 834, 835, 839, 840, 841, 842, 843, 844, 845, 846, 847, 848, 850, 851, 852, 853, 855, 856, 858, 859, 860, 863, 865, 866, 867, 869, 875, 876, 878, 879, 881, 884, 885, 887, 889, 890, 891, 892, 893, 894, 895, 896, 897, 898, 901, 903, 904, 905, 906, 907, 909, 910, 911, 912, 913, 914, 915, 916, 917, 918, 919, 921, 922, 923, 924, 926, 930, 931, 933, 935, 939, 940, 941, 942, 943, 944, 946, 947, 948, 949, 950, 951, 952, 953, 954, 955, 956, 958, 960, 961, 964, 965, 966, 967, 968, 969, 975, 976, 977, 978, 981, 982, 983, 984, 985, 989, 990, 992, 993, 994, 995, 996, 997, 999, 1002, 1004, 1007, 1008, 1009, 1010, 1011, 1013, 1014, 1016, 1018, 1019, 1024, 1025, 1026, 1027, 1029, 1031, 1032, 1033, 1034, 1036, 1037, 1039, 1048, 1049, 1050, 1051, 1054, 1055, 1060, 1062, 1067, 1069, 1073, 1076, 1082, 1085, 1087, 1088, 1090, 1091, 1093, 1095, 1101, 1102, 1104, 1106, 1107, 1108, 1112, 1116, 1118, 1123, 1126, 1128, 1133, 1137, 1138, 1145, 1148, 1152, 1153, 1171, 1181, 1205]
feed_dict = load_data_torch("./data/", et_list, mono=True)


[n_drug, n_feat_d] = feed_dict['d_feat'].shape
n_et_dd = len(et_list)


data = Data.from_dict(feed_dict)

print('constructing model, opt and sent data to device')
device = t.device('cuda' if t.cuda.is_available() else 'cpu')
model = HGCN(n_feat_d, n_et_dd, 50).to(device)
optimizer = t.optim.Adam(model.parameters(), lr=0.001)

n_edge = data.dd_edge_index.shape[1]
train_mask = np.random.binomial(1, 0.95, n_edge)
test_mask = 1 - train_mask
train_set = train_mask.nonzero()[0]
test_set = test_mask.nonzero()[0]

data.train_idx = data.dd_edge_index[:, train_set]
data.test_idx = data.dd_edge_index[:, test_set]

data.train_type = data.train_type[:, train_set]
data.test_type = data.test_type[:, test_set]

data = data.to(device)


# train
def train():
    model.train()

    optimizer.zero_grad()
    pos_pred, neg_pred = model(data.d_feat, data.train_idx, data.dd_edge_type, data.dd_edge_type_num)

    pos_loss = binary_cross_entropy(pos_pred, t.ones(train_set.shape[0]).cuda())
    neg_loss = binary_cross_entropy(neg_pred, t.zeros(train_set.shape[0]).cuda())
    (pos_loss + neg_loss).backward()

    optimizer.step()

    return pos_pred, neg_pred, pos_loss, neg_loss


# acc
def test():
    model.eval()

    roc_sc = metrics.roc_auc_score(targ.tolist(), pred.tolist())
    aupr_sc = metrics.average_precision_score(targ.tolist(), pred.tolist())
    # apk_sc = apk(targ.tolist(), pred.tolist(), k=50)


    return roc_sc, aupr_sc


EPOCH_NUM = 100
train_loss = np.zeros(EPOCH_NUM)
# y_targ = t.cat((data.dd_y_pos, data.dd_y_neg), 0)


print('model training ...')
for epoch in range(EPOCH_NUM):
    start = time.time()
    p_pred, n_pred, p_loss, n_loss = train()


    # acc pos
    p_pred = p_pred > 0.5
    pos_acc = p_pred.sum().tolist() / data.dd_edge_type_num[-1]
    # acc neg
    n_pred = n_pred <= 0.5
    neg_acc = n_pred.sum().tolist() / data.dd_edge_type_num[-1]

    log = 'Epoch: {:03d}/{:.2f}, pos_loss: {:.4f}, neg_loss: {:.4f}, total_loss: {:.4f}, pos_acc: {:.4f}, neg_acc: {:.4f}'
    train_loss[epoch] = p_loss+n_loss
    print(log.format(epoch, time.time() - start, p_loss, n_loss, train_loss[epoch], pos_acc, neg_acc))


outfile = TemporaryFile()
np.save(outfile, train_loss)

# outfile.seek(0) # Only needed here to simulate closing & reopening file
# np.load(outfile)
