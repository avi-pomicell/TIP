import random

import joblib
import pandas as pd

from prepare import prepare_data
from src.utils import *
from src.layers import *

# choose TIP model: 'cat' - TIP-cat
#					'add' - TIP-add
MOD = 'cat'
MAX_EPOCH = 100
MONO_DRUG_SIDES = False

if torch.cuda.is_available():
    print('cuda available')
else:
    print('no cuda')
    
data_dict = prepare_data(MONO_DRUG_SIDES)
joblib.dump(data_dict, 'data_dict-v2.joblib')

data_dict = joblib.load('data_dict-v2.joblib')
# set training device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# initial model
if MOD == 'cat':
    settings = Setting(sp_rate=0.9, lr=0.001, prot_drug_dim=64, n_embed=192, n_hid1=256, n_hid2=128, num_base=24)
    model = TIP(settings, device, data_dict=data_dict)
else:
    settings = Setting(sp_rate=0.9, lr=0.01, prot_drug_dim=64, n_embed=64, n_hid1=32, n_hid2=16, num_base=32)
    model = TIP(settings, device, mod='add', data_dict=data_dict)
    

print(settings.__dict__)


# initial optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=settings.lr)

# train TIP model
for e in range(MAX_EPOCH):
    model.train()
    optimizer.zero_grad()
    loss = model()
    print(f'epoch {e} loss: {loss.item()}')  # epoch 0 loss: 1.3866474628448486, epoch 99 loss: 0.7132794857025146
    loss.backward()
    optimizer.step()
    if e % 4 == 0:
        model.test()

# evaluate on test set
model.test()  # On test set: auprc:0.8949   auroc:0.9185   ap@50:0.8959

# save trained model
torch.save(model, f'./tip-{model.mod}-v2.pt')
print(settings.__dict__)


model = torch.load(f'./tip-{model.mod}-v2.pt')

q_pairs = [
    ('Cannabidiol', 'Cannabigerol'),
    ('Cannabidiol', 'Myrcene'),
    ('Cannabidiol', 'Beta-caryophyllene'),
    ('Cannabidiol', 'Alpha Bisabolol'),
    ('Cannabidiol', 'Linalool'),
    ('Cannabidiol', 'D-LIMONENE'),
    ('Cannabidiol', 'Cannabigerol', 'Myrcene'),
    ('Cannabidiol', 'Cannabigerol', 'Beta-caryophyllene'),
    ('Cannabidiol', 'Cannabigerol', 'Alpha Bisabolol'),
    ('Cannabidiol', 'Cannabigerol', 'Linalool'),
    ('Cannabidiol', 'Cannabigerol', 'D-LIMONENE'),
    # should have side effect:
    ('Cannabidiol', 'KOUMIDINE'),
    ('Cannabidiol', 'chlorothiazide'),
    ('Cannabidiol', 'Methadone'),
    ('Cannabidiol', 'Mirtazapine'),
    ('Cannabidiol', 'Zolmitriptan'),
    ('Benzocaine', 'Acetazolamide'),
    ('Benzocaine', 'Acyclovir'),
    ('clemastine', 'minaprine'), # "#Drug1 may increase the anticholinergic activities of #Drug2."
    ('oxazepam', 'dihydroergotamine'), # "The metabolism of #Drug2 can be decreased when combined with #Drug1."
]
for i in range(15):
    random.seed(i)
    d1, d2 = random.sample(data_dict['name2diid'].keys(), 2)
    q_pairs.append((d1,d2))
predictions = dict()


def predict_pair(d1, d2):
    q_list = []
    q_label_list = []
    for i, idx in enumerate(data_dict['dd_edge_index']):
        q_list.append(torch.tensor([d1, d2], dtype=torch.long))
        q_label_list.append(torch.ones(1, dtype=torch.long) * i)
    q_edge_idx = torch.stack(q_list, dim=1)
    q_et = torch.cat(q_label_list)
    pred = model.decoder(model.embeddings, q_edge_idx, q_et).detach().cpu().numpy()
    return pred


for pair in q_pairs:
    pair_iids = [data_dict['name2diid'][p.lower()] for p in pair]
    pair_lbl = ' + '.join(pair)
    if len(pair_iids) == 2:
        d1, d2 = pair_iids
        predictions[pair_lbl] = predict_pair(d1, d2)
    else:
        d1, d2, d3 = pair_iids
        predictions[pair_lbl] = np.max([predict_pair(d1,d2), predict_pair(d2,d3), predict_pair(d1,d3)], axis=0)
predictions = pd.DataFrame.from_dict(predictions, orient='index', columns=data_dict['side_effect_name'].values())
predictions = predictions.reindex(predictions[:11].mean().sort_values(ascending=False).index, axis=1)
predictions.T.to_csv('tip-results-v2.csv')
top_predictions = {}
for pair, row in predictions.iterrows():
    row = row.sort_values(ascending=False)[:5]
    pair_tops = ', '.join([f'{se} ({val * 100:.01f}%)' for se, val in row.items()])
    top_predictions[pair] = pair_tops
    print(pair, ':')
    print(pair_tops)

# print(top_predictions)

""" ppi conf = 0.89
On test set: auprc:0.7983   auroc:0.8172   ap@50:0.8169    
epoch 61 loss: 0.3599148988723755
...
epoch 70 loss: 0.3245539367198944
On test set: auprc:0.7827   auroc:0.7985   ap@50:0.8010    
epoch 71 loss: 0.3203185498714447
"""

""" ppi conf = 0.85
epoch 60 loss: 0.36797210574150085
On test set: auprc:0.8047   auroc:0.8231   ap@50:0.8228    
epoch 61 loss: 0.3633195757865906
epoch 62 loss: 0.359530508518219
...
epoch 69 loss: 0.3319576680660248
On test set: auprc:0.8015   auroc:0.8201   ap@50:0.8191 
"""

""" ppi conf = 0.80
ppi count: 101906
...
epoch 60 loss: 0.3655155301094055
On test set: auprc:0.8137   auroc:0.8220   ap@50:0.8271    
epoch 61 loss: 0.36313632130622864
epoch 62 loss: 0.35680705308914185
...
epoch 69 loss: 0.3328017592430115
On test set: auprc:0.8019   auroc:0.8146   ap@50:0.8170  
"""

""" on pomicell dataset (v2):
epoch 99 loss: 0.16707593202590942
On test set: auprc:0.9068   auroc:0.9201   ap@50:0.9085    
{'sp_rate': 0.9, 'lr': 0.001, 'prot_drug_dim': 64, 'n_embed': 192, 'n_hid1': 256, 'n_hid2': 128, 'num_base': 24}
"""
