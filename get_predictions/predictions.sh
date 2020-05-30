#!/bin/bash
python run_server.py $2 & python samples_pred.py $1
kill $(lsof -t -i:9001)
