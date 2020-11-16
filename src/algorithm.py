import numpy as np
import pandas as pd 
import os
import time
from collections import OrderedDict
from distributed import worker_client
from data import get_data
from model import get_model

def run_algorithm(yi):
    algo = Algorithm(yi)
    return algo.run()

class Algorithm:

    def __init__(self, yaml_input):
        self.yi = yaml_input
        self.scoring_heuristic = self.yi['model_args']['scoring_heuristic']
        self.data = get_data(self.yi)
        self.current_state_dict = None
        self.log_ = {k: [] for k in ['round', 'labeled', 'selection_idx', 'loss', 'state_dicts', 'round_results']}
        self.rd = 0

    def run(self):
        self.train_report_labeled()
        N_ROUNDS = self.data.N - np.sum(self.data.labeled_mask)
        while self.data.has_unlabeled:
            print("[START ROUND]", self.rd, " / ", N_ROUNDS)
            round_results = self.explore_unlabeled_points()
            idxs = np.unique(round_results['idx'])
            idxmins = [round_results[round_results['idx'] == idx]['score'].idxmin() for idx in idxs]
            round_results = round_results.loc[idxmins]
            maximin_selection = round_results.loc[round_results['score'].idxmax()]
            self.data.mark_labeled(maximin_selection['idx'])
            learn = self.train_report_labeled(maximin_selection, round_results[['idx', 'is_true_y', 'loss', 'score']])
            print("[END ROUND]", self.rd, " / ", N_ROUNDS, f"LOSS={learn['loss']}")
            print("==================================")
        print("[RUN DONE]")
        return self.output

    def analyze(self):
        max_loss = np.max(self.log_['loss'])
        avg_loss = np.mean(self.log_['loss'])
        return {"max_loss": max_loss, "avg_loss": avg_loss}

    def explore_unlabeled_points(self):
        futures = []
        with worker_client() as wc:
            for point in self.data.iterator(self.scoring_heuristic):
                futures.append(wc.submit(self.explore_unlabeled_point, point))
            futures = wc.gather(futures)
        futures = pd.DataFrame(futures)
        return futures

    def explore_unlabeled_point(self, point):
        model = get_model(self.yi)
        sd = None if self.scoring_heuristic != 'gradient_heuristic' else self.current_state_dict
        learn = model.learn(point['x'], point['y'], sd)
        point.update(learn)
        del model
        return point

    def train_report_labeled(self, selected_point={}, round_results=None):
        model = get_model(self.yi)
        x, y = self.data.labeled
        learn = model.learn(x, y)
        self.current_state_dict = learn['state_dict']
        self.log_['round'].append(self.rd)
        self.log_['labeled'].append(self.data.labeled_mask.copy())
        self.log_['selection_idx'].append(selected_point.get('idx', -1))
        self.log_['loss'].append(learn['loss'])
        self.log_['state_dicts'].append(OrderedDict({k: v.numpy() for k, v in learn['state_dict'].items()}))
        self.log_['round_results'].append(round_results)
        self.rd += 1
        return learn

    @property
    def output(self):
        out = {}
        out['input'] = [self.yi]
        out['output'] = [pd.DataFrame(self.log_)]
        out['timestamp'] = [int(time.time()*1000)]
        out['analytics'] = [self.analyze()]
        out = pd.DataFrame(out)
        return out