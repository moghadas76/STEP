#!/bin/bash

while true; do

    python step/run.py --cfg='step/TSFormer_METR-LA.py' --gpus="1, 2"
    # Check if the script exited successfully
    if [ $? -eq 0 ]; then
        echo "Script exited successfully."
        break
    else
        echo "Script exited with an error. Restarting..."
    fi
done
