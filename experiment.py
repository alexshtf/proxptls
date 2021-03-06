import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.utils.data as td
import torch.optim as opt
from sklearn.preprocessing import minmax_scale
from torch.optim.lr_scheduler import LambdaLR
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


# solves the least-squares problem using a direct method
def compute_opt_loss(P, b):
    # pad with a column of ones - the coefficients of the alpha parameter
    Q = np.pad(P, pad_width=[(0, 0), (0, 1)], constant_values=1)
    opt_w, opt_loss, residual, svd = np.linalg.lstsq(Q, b)
    opt_loss /= P.shape[0]
    return opt_loss


def run_experiment(name, plugin, train_set, epochs, attempts, lrs, opt_loss):
    losses = pd.DataFrame(columns=['lr', 'epoch', 'attempt', 'loss'])

    total_epochs = len(lrs) * len(attempts) * len(epochs)
    with tqdm(total=total_epochs, unit='epochs', ncols=140) as pbar:
        for lr in lrs:
            for attempt in attempts:
                pbar.set_description(desc=f'plugin = {name}, lr = {lr}, attempt = {attempt}')

                model = LinReg()
                plugin.construct(model, lr)
                for epoch in epochs:
                    train_loss = 0
                    for x, y in td.DataLoader(train_set, shuffle=True, batch_size=1):
                        xx = x.squeeze(0)

                        model.zero_grad()

                        pred = model.forward(xx)
                        loss = (pred - y) ** 2
                        train_loss += loss.item()
                        loss.backward()

                        plugin.process_sample(xx, y, pred)

                    train_loss /= len(train_set)
                    losses = losses.append(pd.DataFrame.from_dict(
                        {'loss': [train_loss - opt_loss.item()],
                         'epoch': [epoch],
                         'lr': [lr],
                         'attempt': attempt}), sort=True)

                    plugin.end_epoch()
                    pbar.update()

    best_loss_df = losses[['lr', 'attempt', 'loss']].groupby(['lr', 'attempt'], as_index=False).min()
    return best_loss_df[['lr', 'loss']]


boston = pd.read_csv('boston.csv')
X = boston[['RM', 'LSTAT', 'PTRATIO']].to_numpy()
X = minmax_scale(X)
b = boston['MEDV'].to_numpy()
b = minmax_scale(b)

optimal_loss = compute_opt_loss(X, b)
train_set = td.TensorDataset(torch.tensor(X), torch.tensor(b))

epochs = range(0, 100)
attempts = range(0, 20)
lrs = np.geomspace(0.001, 100, num=60)
experiments = [
    ('Proximal point', ProxPointPlugin()),
    ('Adagrad', OptimizerPlugin(make_optimizer=lambda model, lr: opt.Adagrad(model.parameters()),
                                make_scheduler=lambda optimizer, lr: LambdaLR(optimizer,
                                                                              lr_lambda=[lambda i: lr]))),
    ('Adam', OptimizerPlugin(make_optimizer=lambda model, lr: opt.Adam(model.parameters()),
                             make_scheduler=lambda optimizer, lr: LambdaLR(optimizer,
                                                                           lr_lambda=[lambda i: lr]))),
    ('SGD', OptimizerPlugin(make_optimizer=lambda model, lr: opt.SGD(model.parameters(), lr=lr),
                            make_scheduler=lambda optimizer, lr: LambdaLR(optimizer,
                                                                          lr_lambda=[lambda i: lr / math.sqrt(1 + i)])))
]

experiment_results = [run_experiment(name, plugin, train_set, epochs, attempts, lrs, optimal_loss).assign(name=name)
                      for name, plugin in experiments]
result_df = pd.concat(experiment_results)
result_df = result_df[result_df['loss'] < 10000]
result_df.head()

sns.set()
ax = sns.lineplot(x='lr', y='loss', hue='name', data=result_df, err_style='band')
ax.set_yscale('log')
ax.set_xscale('log')
plt.show()
