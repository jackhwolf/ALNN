import numpy as np 

# entry to get data given dictionary input
def get_data(kw):
    data = None
    t = kw['data']
    if t == 'SampleData1d':
        n_t, t_l = int(kw['data_args']['n_threshold']), int(kw['data_args']['threshold_length'])
        data = SampleData1d(n_t, t_l)
    elif t == 'SampleData2d':
        h, w, dbt = int(kw['data_args']['h']), int(kw['data_args']['w']), kw['data_args']['decision_boundary_t']
        data = SampleData2d(h, w, dbt)
    return data

class Data:

    def __init__(self, N, D):
        self.N = N
        self.D = D
        self.x = np.zeros((N, D))
        self.finex = None
        self.y = np.zeros(N)
        self.labeled_mask = np.array([False] * N)
        self.possible_labels = np.unique(self.y)

    # yield all (xi, yi) combinations
    def iterator(self, scoring_heuristic):
        for idx in np.where(~self.labeled_mask)[0]:
            x, true_y = self.x[[idx],:], self.y[idx]
            for possible_label in self.possible_labels:
                out = {}
                if scoring_heuristic == "norm_heuristic":
                    lx, ly = self.pop_on(x, possible_label)
                elif scoring_heuristic == "gradient_heuristic":
                    lx, ly = x.reshape(1,-1), possible_label.reshape(1,-1)
                out['idx'] = idx
                out['x'] = lx
                out['y'] = ly 
                out['true_y'] = true_y
                out['is_true_y'] = possible_label == true_y[0]
                yield out

    # return a copy of the dataset with new x,y popped on
    def pop_on(self, x, y):
        lx, ly = self.labeled
        lx = np.append(lx, x, axis=0)
        ly = np.append(ly, y.reshape(-1,1), axis=0)
        return lx, ly

    # are there unlabeled points remaining
    @property
    def has_unlabeled(self):
        return len(np.where(~self.labeled_mask)[0]) != 0

    # set the contents of the dataset
    def set_data(self, x, finex, y):
        if len(x.shape) == 1:
            x = x.reshape(-1, 1)
        self.x = x
        if len(finex.shape) == 1:
            finex = finex.reshape(-1, 1)
        self.finex = finex
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        self.y = y
        self.possible_labels = np.unique(self.y)
        return

    # get labeled x, y
    @property
    def labeled(self):
        return self.x[self.labeled_mask].copy(), self.y[self.labeled_mask].copy()

    # get unlabeled x, y
    @property
    def unlabeled(self):
        return self.x[~self.labeled_mask].copy(), self.y[~self.labeled_mask].copy()

    # mark an index as labeled
    def mark_labeled(self, idx):
        self.labeled_mask[idx] = True
        return

class SampleData1d(Data):

    def __init__(self, n_threshold, threshold_length):
        self.n_threshold = n_threshold
        self.threshold_length = threshold_length
        super().__init__(self.n_threshold * self.threshold_length, 1)
        x = np.linspace(-1, 1, self.N).astype(np.float32)
        finex = np.linspace(-1, 1, 100).astype(np.float32)
        y = np.arange(self.N).astype(np.float32)
        back = 0
        label = -1
        for i in range(threshold_length, self.N + threshold_length, threshold_length):
            y[back:i] = label
            back = i
            label *= -1
        self.set_data(x, finex, y)
        self.initial_label()

    # set the intial labeled points
    def initial_label(self):
        a = np.where(self.x == np.min(self.x))[0][0]
        b = np.where(self.x == np.max(self.x))[0][0]
        self.mark_labeled(a)
        self.mark_labeled(b)

class SampleData2d(Data):

    def __init__(self, h, w, decision_boundary_t):
        self.h = h
        self.w = w
        self.decision_boundary_t = decision_boundary_t
        super().__init__(self.h * self.w, 2)
        x, y = self.build_data(self.w, self.h)
        finex, _ = self.build_data(100, 100)
        self.set_data(x, finex, y)
        self.initial_label()

    # build the 2d x, y
    def build_data(self, w, h):
        x = np.zeros((w * h, 2))
        y = np.zeros(w * h)
        boundary_func = self.get_decision_boundary_func()
        w_across = np.linspace(-1., 1., w).astype(np.float32)
        h_across = np.linspace(-1., 1., h).astype(np.float32)
        idx = 0
        for wa in w_across:
            for ha in h_across:
                x[idx,0] = wa 
                x[idx,1] = ha
                y[idx] = boundary_func(x[idx,:])
                idx += 1
        y = (y * 2) - 1
        return x, y
                
    # set the intial labeled points
    def initial_label(self):
        xa = np.min(self.x[:,0])
        xb = np.max(self.x[:,0])
        ymid = np.sort(self.x[:,1])[self.N//2]
        a = np.where((self.x == (xa, ymid)).all(axis=1))
        b = np.where((self.x == (xb, ymid)).all(axis=1))
        self.mark_labeled(a)
        self.mark_labeled(b)

    # determine how to calculate y(x)
    def get_decision_boundary_func(self):
        dbm = {}
        dbm['diagonal'] = self.diagonal
        dbm['diagonal_slope'] = self.diagonal_slope
        return dbm[self.decision_boundary_t]

    # diagonal decision boundary 
    def diagonal(self, x):
        return x[0] < x[1]

    # diagonal decision boundary with some slope
    def diagonal_slope(self, x, slope=1.25):
        return x[0] < (x[1]*slope)
