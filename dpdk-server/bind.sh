#!/bin/bash


sudo modprobe vfio-pci
echo 1 | sudo tee /sys/module/vfio/parameters/enable_unsafe_noiommu_mode
sudo dpdk-devbind.py --bind=vfio-pci 0000:86:00.0
sudo dpdk-devbind.py --status