#!/bin/bash
mkdir ./model/CRNN/modelss
while true; do
  cp -n ./model/CRNN/model/shadownet/shadownet* ./model/CRNN/modelss/
  cp ./model/CRNN/model/shadownet/checkpoint* ./model/CRNN/modelss/
  sleep 30
done
