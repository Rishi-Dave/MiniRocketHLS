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
	$(ECHO) "  make build PLATFORM=<FPGA platform>"
	$(ECHO) "      Command to generate the design for specified Target and Shell."
	$(ECHO) ""
	$(ECHO) "  make clean "
	$(ECHO) "      Command to remove the generated non-hardware files."
	$(ECHO) ""
	$(ECHO) "  make cleanall"
	$(ECHO) "      Command to remove all the generated files."
	$(ECHO) ""


############################## Setting up Project Variables ##############################
# Points to top directory of Git repository
MK_PATH := $(abspath $(lastword $(MAKEFILE_LIST)))
COMMON_REPO ?= $(shell bash -c 'export MK_PATH=$(MK_PATH); echo $${MK_PATH}')
PWD = $(shell readlink -f .)
XF_PROJ_ROOT = $(shell readlink -f $(COMMON_REPO))

TARGET := hw

IPREPOPATH ?= ./iprepo
export IPREPOPATH
include ./utils.mk
include ./opencl.mk


XSA := 
ifneq ($(PLATFORM), )
XSA := $(call device2xsa, $(PLATFORM))
endif

TEMP_DIR := _x.$(XSA)
BUILD_DIR := ./build_dir.$(TARGET).$(XSA)
NETLAYERDIR := ./NetLayers/
CMACDIR := ./cmac_krnl/

MINIROCKET_XO_REPO := ./build/iprepo
NETLAYER_XO_REPO := $(NETLAYERDIR)$(TEMP_DIR)
NETLAYERHLS = 100G-fpga-network-stack-core
CMAC_XO_REPO := $(CMACDIR)$(TEMP_DIR)

HLS_IP_FOLDER  = $(shell readlink -f ./$(NETLAYERDIR)$(NETLAYERHLS)/synthesis_results_HBM)
LIST_REPOS := --user_ip_repo_paths $(HLS_IP_FOLDER) --user_ip_repo_paths $(IPREPOPATH)

HOST_DIR := ./host

VPP := v++
CXXFLAGS += $(opencl_CXXFLAGS) -Wall -O0 -g -std=c++1y
LDFLAGS += $(opencl_LDFLAGS)

INCLUDES = ./src

MINIROCKET_KRNL := minirocket_inference
NETLAYER_KRNL := networklayer
CMAC_KRNL:=cmac_krnl
include config_cmac.mk
XOs += $(MINIROCKET_XO_REPO)/$(MINIROCKET_KRNL).xo $(NETLAYER_XO_REPO)/$(NETLAYER_KRNL).xo $(CMAC_XO_REPO)/${CMAC_KRNL}.xo

############################## Setting up Host Variables ##############################

HOST_SRCS += $(HOST_DIR)/src/minirocket_host.cpp $(HOST_DIR)/src/minirocket_loader.cpp
CXXFLAGS += -fmessage-length=0 -I$(INCLUDES) -I${HOST_DIR}/include
LDFLAGS += -lrt -lstdc++ 

############################## Setting up Kernel Variables ##############################
VPP_FLAGS += -t $(TARGET) --platform $(PLATFORM) --save-temps -I$(INCLUDES) --config config.cfg
ifneq ($(TARGET), hw)
	VPP_FLAGS += -g
endif



############################## Declaring Binary Containers ##############################
XLCBIN += $(BUILD_DIR)/krnl.xclbin
EXECUTABLE = ./minirocket_host

############################## Setting Targets ##############################
CP = cp -rf

.PHONY: all clean cleanall 
all: check-platform check-device check-vitis $(EXECUTABLE) $(XLCBIN) 

.PHONY: host
host: $(EXECUTABLE)

.PHONY: build
build: check-vitis check-device $(XLCBIN)

.PHONY: xclbin
xclbin: build

############################## Setting Rules for Binary Containers (Building Kernels) ##############################

$(XLCBIN): $(XOs)
	mkdir -p $(BUILD_DIR)
ifeq ($(HOST_ARCH), x86)
	$(VPP) $(VPP_FLAGS) -l $(VPP_LDFLAGS) $(LIST_REPOS) --temp_dir $(TEMP_DIR) -o'$(BUILD_DIR)/krnl.link.xclbin' $(+)
	$(VPP) -p $(BUILD_DIR)/krnl.link.xclbin -t $(TARGET) --platform $(PLATFORM) --package.out_dir $(PACKAGE_OUT) -o $(BUILD_DIR)/krnl.xclbin
else
	$(VPP) $(VPP_FLAGS) -l $(VPP_LDFLAGS) $(LIST_REPOS) --temp_dir $(TEMP_DIR) -o'$(BUILD_DIR)/krnl.xclbin' $(+)
endif

############################## Cleaning Rules ##############################
# Cleaning stuff
clean:
	-$(RMDIR) $(EXECUTABLE) $(XCLBIN)/{*sw_emu*,*hw_emu*} 
	-$(RMDIR) profile_* TempConfig system_estimate.xtxt *.rpt *.csv 
	-$(RMDIR) src/*.ll *v++* .Xil emconfig.json dltmp* xmltmp* *.log *.jou *.wcfg *.wdb

cleanall: clean
	-$(RMDIR) build_dir* sd_card*
	-$(RMDIR) package.*
	-$(RMDIR) _x* *xclbin.run_summary qemu-memory-_* emulation _vimage pl* start_simulation.sh *.xclbin packaged_kernel_* tmp_kernel_*
