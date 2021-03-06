import torch
import torch.nn as nn
import torch.optim as optim 
import numpy as np

# entry to get model given dictionary input
def get_model(kw):
        return Model(**kw['model_args'])

# https://raw.githubusercontent.com/Bjarten/early-stopping-pytorch/master/pytorchtools.py
class EarlyStopper:

    def __init__(self, patience, delta):
        self.patience = int(patience)
        self.delta = float(delta)
        self.prev_val = None
        self.counter = 0

    def __call__(self, val):
        if self.prev_val is None:
            self.prev_val = val
            return False
        else:
            improvement = self.prev_val - val
            self.counter += 1 if improvement < self.delta else 0
        if self.counter >= self.patience:
            return True
        self.prev_val = val
        return False

class Model(nn.Module):

    def __init__(self, dimsin, hidden_nodes, dimsout, loss_function, \
                        optimizer_function, lr, wd, epochs, scoring_heuristic, \
                        early_stop_patience, early_stop_delta):
        super().__init__()
        self.dims_in = int(dimsin)
        self.hidden_nodes = int(hidden_nodes)
        self.dims_out = int(dimsout)
        self.loss_func = loss_function
        self.optim_func = optimizer_function
        self.optim_params = {'lr': float(lr), 'weight_decay': float(wd)}
        self.epochs = epochs
        self.scoring_heuristic = scoring_heuristic
        self.early_stop_patience = early_stop_patience
        self.early_stop_delta = early_stop_delta
        self.l1 = nn.Linear(self.dims_in, self.hidden_nodes)
        self.l2 = nn.Linear(self.hidden_nodes, self.dims_out)
        self.current_loss = None
        self.current_state_dict = None
        self.dev = torch.device('cpu:0')

    # learn an (x,y) dataset and return performance and parameters in dict
    def learn(self, x, y, sd=None, allow_earlystop=True):
        x = torch.from_numpy(x.astype(np.float32)).to(self.dev)
        y = torch.from_numpy(y.astype(np.float32)).to(self.dev)
        loss_func = getattr(nn, self.loss_func)()
        optimizer = getattr(optim, self.optim_func)
        optimizer = optimizer(self.parameters(), **self.optim_params)
        epochs = self.epochs
        stopper = EarlyStopper(self.early_stop_patience, self.early_stop_delta)
        if sd is not None and self.scoring_heuristic == 'gradient_heuristic':
            self.load_state_dict(sd)
            epochs = 1
        for i in range(epochs):
            pred = self.forward(x)
            loss = loss_func(pred, y)
            self.current_loss = loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if epochs != 1 and stopper(self.current_loss) and allow_earlystop:
                print(f"Stopped @ {i} / {epochs}")
                break
        self.current_state_dict = self.state_dict()
        out = {}
        out['loss'] = loss.item()
        out['score'] = self.score
        out['state_dict'] = self.current_state_dict
        out['trained_for'] = i
        return out

    # forward pass on x
    def forward(self, x):
        relu = self.l1(x).clamp(min=0)
        return self.l2(relu)

    # forward pass with no backprop on x
    def predict(self, x):
        x = torch.from_numpy(x.astype(np.float32)).to(self.dev)
        with torch.no_grad():
            return self.forward(x)

    # calculate score of model 
    @property
    def score(self):
        if self.scoring_heuristic == 'norm_heuristic':
            return self.norm_heuristic()
        elif self.scoring_heuristic == 'gradient_heuristic':
            return self.gradient_heuristic()

    # calculate sum of norms of weights
    def norm_heuristic(self):
        norm = 0
        l1 = self.l1.weight.detach().numpy()
        l2 = self.l2.weight.detach().numpy()
        norm += np.linalg.norm(l1)
        norm += np.linalg.norm(l2)
        return norm

    # calculate norm of gradient after one forward pass
    def gradient_heuristic(self):
        loss = self.current_loss
        wi_grads = np.array([])
        for w in [self.l1, self.l2]:
            w = w.weight.grad.numpy().flatten()
            wi_grads = np.append(wi_grads, w)
        wi_grads *= loss
        norm = np.linalg.norm(wi_grads)
        return norm

