#!/bin/bash
python run_server.py & python samples_pred.py $1
kill $(lsof -t -i:9001)
