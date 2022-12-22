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


# set training device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# initial model
if MOD == 'cat':
    settings = Setting(sp_rate=0.9, lr=0.01, prot_drug_dim=16, n_embed=48, n_hid1=32, n_hid2=16, num_base=32)
    model = TIP(settings, device, data_dict=data_dict)
else:
    settings = Setting(sp_rate=0.9, lr=0.01, prot_drug_dim=64, n_embed=64, n_hid1=32, n_hid2=16, num_base=32)
    model = TIP(settings, device, mod='add', data_dict=data_dict)

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

# evaluate on test set
model.test()  # On test set: auprc:0.8949   auroc:0.9185   ap@50:0.8959

# save trained model
torch.save(model, f'./tip-{model.mod}-example.pt')
