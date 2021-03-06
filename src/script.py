if __name__ == '__main__':
    import sys
    import os
    import time     
    from distributed import Client
    
    fname = sys.argv[1]
    dask_addr = sys.argv[2]

    # get dask client and sleep so it has time to set up
    cli = Client(dask_addr)
    time.sleep(10)

    # upload files so workers can import
    cli.upload_file("data.py")
    cli.upload_file("model.py")
    cli.upload_file("graphing.py")
    cli.upload_file("resultsmanager.py")
    cli.upload_file("algorithm.py")
    cli.upload_file("yamlinput.py")
    
    # run imports after uploading 
    from algorithm import run_algorithm
    from yamlinput import get_yamlinputs
    from resultsmanager import add_result

    # for each input in the file, distribute a call to 
    # the helper run_algorithm, gather results, and save
    yamls = get_yamlinputs(fname)
    futures = []
    for yi in yamls:
        futures.append(cli.submit(run_algorithm, yi))
    futures = cli.gather(futures)
    print("******************")
    for future in futures:
        add_result(future)
    print(futures)

