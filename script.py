from src.yamlinput import get_yamlinputs
from src.algorithm import run_algorithm
from src.resultsmanager import add_result

def script(fname):
    yamls = get_yamlinputs(fname)
    futures = []
    for yi in yamls:
        futures.append(run_algorithm(yi))
    for future in futures:
        add_result(future)
    return futures

if __name__ == '__main__':
    import sys
    # from distributed import Client
    fname = sys.argv[1]
    script(fname)

