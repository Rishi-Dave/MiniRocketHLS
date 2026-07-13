#
# Copyright 2019-2021 Xilinx, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# makefile-generator v1.0.3
#

SHELL := /bin/bash

############################## Help Section ##############################
.PHONY: help

help::
	$(ECHO) "Makefile Usage:"
	$(ECHO) "  make all TARGET=<sw_emu/hw_emu/hw> PLATFORM=<FPGA platform>"
	$(ECHO) ""
	$(ECHO) "  make build TARGET=<sw_emu/hw_emu/hw> PLATFORM=<FPGA platform>"
	$(ECHO) "      Command to build xclbin application."
	$(ECHO) ""
	$(ECHO) "  make xo"
	$(ECHO) "      Build all three .xo files (load, store, hydra_inference)."
	$(ECHO) ""
	$(ECHO) "  make clean"
	$(ECHO) "      Command to remove generated non-hardware files."
	$(ECHO) ""
	$(ECHO) "  make cleanall"
	$(ECHO) "      Command to remove all generated files."
	$(ECHO) ""

############################## Setting up Project Variables ##############################
MK_PATH := $(abspath $(lastword $(MAKEFILE_LIST)))
COMMON_REPO ?= $(shell bash -c 'export MK_PATH=$(MK_PATH); echo $${MK_PATH%/*}')
PWD = $(shell readlink -f .)
XF_PROJ_ROOT = $(COMMON_REPO)

TARGET := hw
HOST_ARCH := x86
SYSROOT :=

include ./utils.mk

PLATFORM ?= xilinx_u280_gen3x16_xdma_1_202211_1

XSA :=
ifneq ($(PLATFORM), )
XSA := $(call device2xsa, $(PLATFORM))
endif
TEMP_DIR := ./_x.$(TARGET).$(XSA)
BUILD_DIR := ./build_dir.$(TARGET).$(XSA)
IP_REPO := ./build/iprepo

VPP := v++
VPP_FLAGS += -t $(TARGET) --platform $(PLATFORM) --save-temps --config config.cfg
ifneq ($(TARGET), hw)
	VPP_FLAGS += -g
endif

############################## Declaring Binary Containers ##############################
BINARY_CONTAINERS += $(BUILD_DIR)/krnl.xclbin
BINARY_CONTAINER_krnl_OBJS += $(IP_REPO)/hydra_inference.xo $(IP_REPO)/load.xo $(IP_REPO)/store.xo

############################## Setting Rules for Binary Containers (Building Kernels) ##############################

# load.xo: synthesize via existing make.tcl
$(IP_REPO)/load.xo: $(PWD)/load/src/load.cpp $(PWD)/load/include/load.hpp
	cd $(PWD)/build/load && vitis_hls -f make.tcl -tclargs synthesis
	cd $(PWD)/build/load && vitis_hls -f make.tcl -tclargs ip
	cd $(PWD)/build/load && vitis_hls -f make.tcl -tclargs installip

# store.xo: synthesize via existing make.tcl
$(IP_REPO)/store.xo: $(PWD)/store/src/store.cpp $(PWD)/store/include/store.hpp
	cd $(PWD)/build/store && vitis_hls -f make.tcl -tclargs synthesis
	cd $(PWD)/build/store && vitis_hls -f make.tcl -tclargs ip
	cd $(PWD)/build/store && vitis_hls -f make.tcl -tclargs installip

# hydra_inference.xo: already synthesized — just verify it exists
$(IP_REPO)/hydra_inference.xo:
	@echo "hydra_inference.xo should already exist in $(IP_REPO)"
	@test -f $(IP_REPO)/hydra_inference.xo || (echo "ERROR: hydra_inference.xo missing from $(IP_REPO)" && exit 1)

.PHONY: xo
xo: $(IP_REPO)/hydra_inference.xo $(IP_REPO)/load.xo $(IP_REPO)/store.xo

$(BUILD_DIR)/krnl.xclbin: $(BINARY_CONTAINER_krnl_OBJS)
	mkdir -p $(BUILD_DIR)
	$(VPP) -l $(VPP_FLAGS) --temp_dir $(TEMP_DIR) -o'$(BUILD_DIR)/krnl.link.xclbin' $(+)
	$(VPP) -p $(BUILD_DIR)/krnl.link.xclbin -t $(TARGET) --platform $(PLATFORM) --package.out_dir $(BUILD_DIR) -o $(BUILD_DIR)/krnl.xclbin

.PHONY: all build xclbin
all: build
build: $(BINARY_CONTAINERS)
xclbin: build

############################## Cleaning Rules ##############################
clean:
	-rm -rf profile_* TempConfig system_estimate.xtxt *.rpt *.csv
	-rm -rf *.log *.jou *.wcfg *.wdb

cleanall: clean
	-rm -rf build_dir* _x*
	-rm -rf package.*
	-rm -rf *.xclbin
