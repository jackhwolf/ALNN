#!/bin/bash

open_scheduler_window() {
    tmux rename-window -t 0 'scheduler'
    tmux send-keys -t 'scheduler' 'source venv/bin/activate' C-m 'dask-scheduler' C-m 
}

open_worker_window() {
    tmux new-window -t $SESSION:$1 -n 'worker'$2
    tmux send-keys -t 'worker'$2 'source venv/bin/activate' C-m 'export OMP_NUM_THREADS='$3 C-m
    tmux send-keys -t 'worker'$2 'dask-worker '$4 C-m
}

open_script_window() {
    tmux new-window -t $SESSION:$1 -n 'script'
    tmux send-keys -t 'script' 'source venv/bin/activate' C-m 
    tmux send-keys -t 'script' 'python3 script.py '$2' '$3 C-m
}


addr=$(python3 -c '''
import json
from socket import gethostname
dask_addr_map = {
    "SkippyElvis": "192.168.1.10:8786",
    "opt-a003.discovery.wisc.edu": "144.92.142.183:8786",
    "opt-a004.discovery.wisc.edu": "144.92.142.184:8786"
}
print(dask_addr_map.get(gethostname()))
''')

N_FRAMES=$1
SCRIPT_FRAME=$N_FRAMES
let N_FRAMES-=1
export fname=$2
echo $fname
if [[ $addr = "None" ]];
    then
        echo "add "$(hostname)" dask address to entry.sh"
        exit 1
fi

echo $addr

SESSION="alnn-tmux"
SESSIONEXISTS=$(tmux list-sessions | grep $SESSION)

if [[ $SESSIONEXISTS = "" ]]
then
    tmux new-session -d -s $SESSION
    open_scheduler_window
    for i in $(seq 1 $N_FRAMES); do
        open_worker_window $i $i 4 $addr
    done
    open_script_window $SCRIPT_FRAME $fname $addr

fi

# Attach Session, on the Main window
tmux attach-session -t $SESSION
