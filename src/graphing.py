import matplotlib
matplotlib.use('Agg')
import numpy as np   
from uuid import uuid4
from collections import OrderedDict
import os
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [16, 12]
from torch import from_numpy
from data import get_data
from model import get_model

def graph(result):
    data = result.loc[0]['input']['data']
    if data == 'SampleData1d':
        g = Grapher1d(result)
    elif data == 'SampleData2d':
        g = Grapher2d(result)
    return {"local_graphs_root": g.graph()}

class Grapher:

    def __init__(self, result):
        self.result = result
        self.data = get_data(result.loc[0]['input'])
        self.model = get_model(result.loc[0]['input'])
        self.root = f'../Results/{uuid4().hex}'
        if not os.path.exists(self.root):
            os.makedirs(self.root)
        self.i = 0
        self.axs = []
            
    def make_title(self, selection=None):
        res = self.result.loc[0]['input']
        title = f"Name: {res['experiment_name']}"
        data_title = [f"{k}:{v}" for k, v in res['data_args'].items()]
        model_title = f"Architecture: ({res['model_args']['dimsin']}, {res['model_args']['hidden_nodes']}, {res['model_args']['dimsout']})"
        title += f"\nData: {res['data']}({', '.join(data_title)})"
        title += f"\nArchitecture: {model_title}"
        title += f"\nLoss: {res['model_args']['loss_function']}"
        title += f"\nOptimizer: {res['model_args']['optimizer_function']}(LR={res['model_args']['lr']}, WD={res['model_args']['wd']})"
        title += f"\nScoring Heuristic: {res['model_args']['scoring_heuristic']}"
        title += f"\nRound: {self.i}"
        if selection is not None:
            title += f", Selection: {selection}"
        return title

    def get_interpolations(self):
        for i, o in self.result.loc[0]['output'].iterrows():
            sd = OrderedDict({k: from_numpy(v) for k, v in o['state_dicts'].items()})
            self.model.load_state_dict(sd)
            x = self.data.x[o['labeled']]
            predy = self.model.predict(x).numpy()
            finex = self.data.finex
            finepredy = self.model.predict(finex).numpy()
            yield o, (x, predy), (finex, finepredy)

    def graph(self):
        return ""

    def save(self, fig, ax):
        path = os.path.join(self.root, str(self.i).zfill(5) + ".png")
        fig.savefig(path, bbox_inches='tight')
        # with open(self.root + "/animation.mp4", 'w') as fp:
        #     fp.write("Hello\n")
        self.i += 1
        return path

    def square(self, axs):
        for axidx in range(len(axs)):
            ax = axs[axidx]
            ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
        return

    def animate(self):
        cmd = f"./cmds/animate.sh {self.root} animation.mp4"
        os.system(cmd)
        return 

class Grapher1d(Grapher):

    def __init__(self, result):
        super().__init__(result)

    def graph(self):
        for o, (x, predy), (finex, finepredy) in self.get_interpolations():
            fig, ax = plt.subplots(ncols=3)
            self._ticks(ax[1])
            self._ticks(ax[2])
            ax[0].set_title("Interpolation", fontsize=16)
            ax[1].set_title("Scoring", fontsize=16)
            ax[2].set_title("Loss", fontsize=16)
            cbar = ax[0].scatter(x[:,0], predy, c=predy, cmap='bwr')
            fig.colorbar(cbar, ax=ax[0], fraction=0.046, pad=0.04)
            ax[0].plot(finex[:,0], finepredy, c='grey')
            ax[0].tick_params(which='both', length=0)
            ax[0].set_ylim(-1.025, 1.025)
            selection = o['selection_idx']
            if o['round_results'] is not None:
                xticks = o['round_results']['idx']
                ax[0].axvline(x=self.data.x[selection], color='grey', alpha=0.5, linestyle='--')
                c = ['grey'] * len(o['round_results']['score'])
                c[np.argmax(o['round_results']['score'])] = 'g'
                ax[1].bar(xticks, o['round_results']['score'], color=c)
                ax[2].set_title(f"Loss, max={np.round(np.max(o['round_results']['loss'].values), 3)}", fontsize=16)
                ax[2].bar(xticks, o['round_results']['loss'], color=c)
            self.square(ax)
            title_selection = None if selection == -1 else selection
            fig.suptitle(self.make_title(title_selection), y=.85, fontsize=18)
            self.save(fig, ax)
            plt.close()
            del fig
        self.animate()
        return self.root

    def _ticks(self, ax):
        ax.set_xlim((-1.,1.))
        ax.set_xticks(np.arange(self.data.N))
        xtls = [''] * self.data.N
        ax.set_xticklabels(xtls)
        ax.set_yticklabels([''] * len(ax.get_yticklabels()))
        ax.tick_params(axis='both', which='both', length=0)

class Grapher2d(Grapher):

    def __init__(self, result):
        super().__init__(result)

    def graph(self):
        for o, (x, predy), (finex, finepredy) in self.get_interpolations():
            fig, ax = plt.subplots(ncols=3)
            self._ticks(ax[0])
            self._ticks(ax[1])
            self._ticks(ax[2])
            ax[0].set_xlim((-1.025,1.025))
            ax[0].set_ylim((-1.025,1.025))
            ax[0].set_title("Interpolation", fontsize=16)
            ax[1].set_title("Scoring", fontsize=16)
            ax[2].set_title("Loss", fontsize=16)
            cbar = ax[0].scatter(finex[:,0], finex[:,1], c=finepredy, cmap='bwr')
            fig.colorbar(cbar, ax=ax[0], fraction=0.046, pad=0.04)
            ax[0].plot(x[:,0], x[:,1], c="g", marker="s", fillstyle='none', mew=2, linestyle='none')
            selection = o['selection_idx']
            if o['round_results'] is not None:
                # ax[0].plot(self.data.x[selection,0], self.data.x[selection,1], c="g", marker="s", fillstyle='none', mew=2)
                print("LOSS, SCORE")
                self._mark_selection(ax[1], selection)
                self._mark_selection(ax[2], selection)
                self._graph_loss_or_score(o, fig, ax[1], 'score')
                self._graph_loss_or_score(o, fig, ax[2], 'loss')
                ax[2].set_title(f"Loss, max={np.round(np.max(o['round_results']['loss'].values), 3)}", fontsize=16)
            self.square(ax)
            title_selection = None if selection == -1 else self.data.x[selection,:]
            fig.suptitle(self.make_title(title_selection), y=.85, fontsize=18)
            self.save(fig, ax)
            plt.close()
            del fig
        self.animate()
        return self.root

    def _mark_selection(self, ax, selection):
        ax.plot(self.data.x[selection,0], self.data.x[selection,1], c="g", marker="s", ms=12, fillstyle='none', mew=2)

    def _graph_loss_or_score(self, o, fig, ax, key):
        ax.set_xlim((-1.025,1.025))
        ax.set_ylim((-1.025,1.025))
        xticks = o['round_results']['idx'].values
        currx = self.data.x[o['round_results']['idx'].values]
        c = o['round_results'][key].values
        cmap = 'Wistia'
        ax.scatter(self.data.x[:,0], self.data.x[:,1], c='grey', s=100, marker='x')
        cbar = ax.scatter(currx[:,0], currx[:,1], c=c, cmap=cmap, s=300, marker='s', alpha=0.85)
        if key == 'loss':
            fig.colorbar(cbar, ax=ax, fraction=0.046, pad=0.04)

    def _ticks(self, ax):
        xt = np.linspace(-1, 1, self.data.w)
        yt = np.linspace(-1, 1, self.data.h)
        ax.set_xticks(xt)
        ax.set_xticklabels([''] * len(xt))
        ax.set_yticks(yt)
        ax.set_yticklabels([''] * len(yt))
        ax.tick_params(axis='both', which='both', length=0)
