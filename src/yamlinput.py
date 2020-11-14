import yaml
from itertools import product

def get_yamlinputs(fname):
    yi = yamlinput(fname)
    return list(yi.input_iterator())

class yamlinput:

    def __init__(self, fname):
        self.fname = fname
        self.yaml_content = None
        self.read()

    def input_iterator(self):
        iterkeys = [foo for foo in self.yaml_content if isinstance(self.yaml_content[foo], dict)]
        iters = {}
        for ik in iterkeys:
            iters[ik] = list(self._input_iterator(ik))
        iters = product(*iters.values())
        iters = [dict(zip(iterkeys, i)) for i in iters]
        for it in iters:
            out = self.yaml_content.copy()
            out.update(it)
            yield out

    def _input_iterator(self, parent=None):
        if parent is None:
            data = self.yaml_content
        else:
            data = self.yaml_content[parent]
        for k in list(data):
            if not isinstance(data[k], list):
                data[k] = [data[k]]
        combos = product(*data.values())
        keys = data.keys()
        for c in combos:
            out = dict(zip(keys, c))
            yield out

    def read(self):
        with open(self.fname) as file:
            self.yaml_content = yaml.load(file, yaml.FullLoader)
        return

      