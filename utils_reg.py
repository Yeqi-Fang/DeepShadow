import torch
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer, lr):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def violin_plot(df, save_dir, loss_fn, test_mae):
        fig, ax = plt.subplots(figsize=(7, 7))
        palette = sns.color_palette('mako_r', n_colors=13)
        sns.violinplot(x=df.Real.astype(int), y=df.Pred, inner=None, ax=ax, palette=palette)
        newax = fig.add_axes(ax.get_position(), frameon=False)
        x = np.arange(64, 75)
        y = x
        # with sns.set_theme(style="darkgrid"):
        newax.plot(x, y, '-', color='#dcbe87', markersize=6, lw=2)
        newax.plot(x, y, 'o', color='#FFC75F', markersize=6, lw=2)
        newax.grid(False)
        # ax.set_xlim(60, 77)
        ax.set_ylim(df.Pred.min() - 0.6, df.Pred.max() + 0.6)
        newax.set_ylim(df.Pred.min() - 0.6, df.Pred.max() + 0.6)
        ax.set_xlabel('Real size of black hole (px)')
        ax.set_ylabel('Predicted distribution of the size (px)')
        plt.savefig(f'{save_dir}/violin.png', dpi=600)
        plt.savefig(f'{save_dir}/violin.pdf', dpi=600)
        df.to_csv(f'{save_dir}/{loss_fn}-{test_mae:.3f}.csv')



class CNN(nn.Module):
    def __init__(self, base, out_features=2, DROPOUT_RATE=0.2):
        super(CNN, self).__init__()
        self.base = base
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=DROPOUT_RATE)
        self.fc2 = nn.Linear(in_features=256, out_features=32, bias=True)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(p=DROPOUT_RATE)
        self.fc3 = nn.Linear(in_features=32, out_features=out_features, bias=True)

    def forward(self, x):
        out = self.base(x)
        out = self.relu2(out)
        out = self.dropout2(out)
        out = self.fc2(out)
        out = self.relu3(out)
        out = self.dropout3(out)
        out = self.fc3(out)
        return out