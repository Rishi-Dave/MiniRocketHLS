#!/bin/bash


sudo apt update
sudo apt install -y build-essential cmake wget unzip python3-pip dpdk dpdk-dev

pip install numpy aeon scikit-learn posix_ipc

sudo sysctl -w vm.nr_hugepages=1024