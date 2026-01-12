#!/bin/bash

# sudo apt-get update
# sudo apt-get install -y python3 python3-pip
# pip3 install numpy scikit-learn sktime

mkdir /proj/octfpga-PG0/pyuvaraj/rocket
cp -r ./benchmark.intf0.xilinx_u280_gen3x16_xdma_1_202211_1/vnx_benchmark_if0.xclbin /proj/octfpga-PG0/pyuvaraj/rocket
cp -r xrt_host_api /proj/octfpga-PG0/pyuvaraj/rocket