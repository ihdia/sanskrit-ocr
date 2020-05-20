#!/bin/bash
mkdir ./model/modelss
while true; do
  cp -n ./checkpoints/model.ckpt-* ./model/modelss/
  cp ./checkpoints/checkpoint* ./model/modelss/
  sleep 30
done
