import pandas as pd
import torch
import torch.utils.data as td
import numpy as np
from sklearn.preprocessing import minmax_scale
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

from optimizer_plugin import OptimizerPlugin
from prox_point_plugin import ProxPointPlugin


class LinReg(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = torch.nn.Parameter(torch.empty((1,), dtype=torch.float64), requires_grad=True)
        self.beta = torch.nn.Parameter(torch.empty((3,), dtype=torch.float64), requires_grad=True)

        torch.nn.init.normal_(self.alpha)
        torch.nn.init.normal_(self.beta)

    def forward(self, p):
        return self.alpha + torch.dot(self.beta, p)


boston = pd.read_csv('boston.csv')
print(boston.head())

X = boston[['RM', 'LSTAT', 'PTRATIO']].to_numpy()
X = minmax_scale(X)
b = boston['MEDV'].to_numpy()
b = minmax_scale(b)

train_set = td.TensorDataset(torch.tensor(X), torch.tensor(b))

X_pad = np.pad(X, pad_width=[(0, 0), (0, 1)], constant_values=1)
opt_w, opt_loss, residual, svd = np.linalg.lstsq(X_pad, b)
opt_loss /= len(train_set)
print('Optimal solution = ' + str(opt_w))
print('Optimal loss = ' + str(opt_loss))

epochs = range(0, 100)
attempts = range(0, 20)
lrs = np.geomspace(0.001, 100, num=60)
losses = pd.DataFrame(columns=['lr', 'epoch', 'attempt', 'loss'])

# choose either OptimizerPlugin to run the experiment using an optimizer, or the ProxPointPlugin to run
# the experiment using the stochastic proximal-point algorithmn.
# plugin = OptimizerPlugin(make_optimizer=lambda model, lr: torch.optim.Adagrad(model.parameters()),
#                          make_scheduler=lambda optimizer, lr: torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[lambda i: lr]))
plugin = ProxPointPlugin()

for lr in lrs:
    for attempt in attempts:
        model = LinReg()

        plugin.construct(model, lr)

        with tqdm(epochs, desc=f'lr = {lr}, attempt = {attempt}', unit='epoch') as tqdm_epochs:
            for epoch in tqdm_epochs:
                train_loss = 0
                for x, y in td.DataLoader(train_set, shuffle=True, batch_size=1):
                    xx = x.squeeze(0)

                    model.zero_grad()

                    pred = model.forward(xx)
                    loss = (pred - y) ** 2
                    train_loss += loss.item()
                    loss.backward()

                    plugin.sample_step(xx, y, pred)

                train_loss /= len(train_set)
                losses = losses.append(pd.DataFrame.from_dict(
                    {'loss': [train_loss - opt_loss.item()],
                     'epoch': [epoch],
                     'lr': [lr],
                     'attempt': attempt}), sort=True)

                plugin.epoch_step()


sns.set()

plt.figure()
best_loss_df = losses[['lr', 'attempt', 'loss']].groupby(['lr', 'attempt'], as_index=False).min()
best_loss_df = best_loss_df[['lr', 'loss']]
print(best_loss_df.head())
ax = sns.lineplot(x='lr', y='loss', data=best_loss_df, err_style='band')
ax.set_yscale('log')
ax.set_xscale('log')

plt.show()
