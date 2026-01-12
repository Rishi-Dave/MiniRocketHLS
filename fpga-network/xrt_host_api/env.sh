#!/bin/bash

sudo apt-get update
sudo apt-get install -y python3 python3-pip libidn12 cmake libjsoncpp-dev
pip3 install numpy scikit-learn sktime

sudo ln -s /usr/lib/x86_64-linux-gnu/libidn.so.12 /usr/lib/x86_64-linux-gnu/libidn.so.11