#
# MiniRocket Modular - Three-kernel FPGA pipeline
# K1: Feature Extraction, K2: Scaler, K3: Classifier
# Connected via AXI-Stream
#
# Usage:
#   make host                              - Build host only
#   make build TARGET=hw_emu PLATFORM=...  - Build xclbin
#   make all TARGET=hw_emu PLATFORM=...    - Build everything
#   make run TARGET=hw_emu PLATFORM=...    - Build and run emulation
#   make clean / make cleanall
#

SHELL := /bin/bash

# Defaults
TARGET ?= hw_emu
HOST_ARCH := x86
PLATFORM ?= xilinx_u280_gen3x16_xdma_1_202211_1

# Platform short name (strip path and .xpfm extension)
# Use $(notdir) instead of $(shell basename) to avoid capturing shell init output
XSA := $(strip $(patsubst %.xpfm,%,$(notdir $(PLATFORM))))

# Directory layout
BUILD_DIR  := build_dir.$(TARGET).$(XSA)
TEMP_DIR   := _x.$(TARGET).$(XSA)
IP_REPO    := build/iprepo
HOST_DIR   := host
PACKAGE_OUT := package.$(TARGET)

# XRT / OpenCL settings
ifndef XILINX_XRT
$(error XILINX_XRT is not set. Source /opt/xilinx/xrt/setup.sh first)
endif
XRT_INC    := $(XILINX_XRT)/include
XRT_LIB    := $(XILINX_XRT)/lib
VIVADO_INC := $(XILINX_VIVADO)/include

# Compiler settings
CXX       := g++
CXXFLAGS  := -I$(XRT_INC) -I$(VIVADO_INC) -I./include -I$(HOST_DIR)/include \
             -Wall -O0 -g -std=c++1y -fmessage-length=0
LDFLAGS   := -L$(XRT_LIB) -lOpenCL -pthread -lrt -lstdc++

# Vitis compiler
VPP       := v++
VPP_FLAGS := -t $(TARGET) --platform $(PLATFORM) --save-temps -I./include
ifneq ($(TARGET),hw)
VPP_FLAGS += -g
endif
# config.cfg (nk, sc, sp directives) only applies during linking
VPP_LINK_FLAGS := $(VPP_FLAGS) --config config.cfg

# Host
EXECUTABLE := modular_host
HOST_SRCS  := $(HOST_DIR)/src/modular_host.cpp $(HOST_DIR)/src/minirocket_loader.cpp
CMD_ARGS    = $(BUILD_DIR)/krnl.xclbin minirocket_model.json minirocket_model_test_data.json

# Kernel .xo objects
XO_FE   := $(IP_REPO)/feature_extraction.xo
XO_SC   := $(IP_REPO)/scaler.xo
XO_CL   := $(IP_REPO)/classifier.xo
XCLBIN  := $(BUILD_DIR)/krnl.xclbin

# ===================== Targets =====================

.PHONY: all host build run emconfig clean cleanall help

help:
	@echo "Makefile Usage:"
	@echo "  make all TARGET=<sw_emu|hw_emu|hw> PLATFORM=<platform>"
	@echo "  make host          - Build host only"
	@echo "  make build         - Build xclbin only"
	@echo "  make run           - Build all and run"
	@echo "  make clean/cleanall"

all: host build emconfig

build: $(XCLBIN)

# ===================== Host =====================

host: $(EXECUTABLE)

$(EXECUTABLE): $(HOST_SRCS)
	$(CXX) -o $@ $^ $(CXXFLAGS) $(LDFLAGS)

# ===================== Kernel Compilation =====================

$(XO_FE): feature_extraction/src/feature_extraction.cpp
	mkdir -p $(IP_REPO)
	$(VPP) $(VPP_FLAGS) -c -k feature_extraction \
		-I./include -I./feature_extraction/src \
		--temp_dir $(TEMP_DIR)/feature_extraction -o $@ $<

$(XO_SC): scaler/src/scaler.cpp
	mkdir -p $(IP_REPO)
	$(VPP) $(VPP_FLAGS) -c -k scaler \
		-I./include -I./scaler/src \
		--temp_dir $(TEMP_DIR)/scaler -o $@ $<

$(XO_CL): classifier/src/classifier.cpp
	mkdir -p $(IP_REPO)
	$(VPP) $(VPP_FLAGS) -c -k classifier \
		-I./include -I./classifier/src \
		--temp_dir $(TEMP_DIR)/classifier -o $@ $<

# ===================== Kernel Linking =====================

$(XCLBIN): $(XO_FE) $(XO_SC) $(XO_CL)
	mkdir -p $(BUILD_DIR)
	$(VPP) $(VPP_LINK_FLAGS) -l --temp_dir $(TEMP_DIR) \
		-o $(BUILD_DIR)/krnl.link.xclbin $^
	$(VPP) -p $(BUILD_DIR)/krnl.link.xclbin \
		-t $(TARGET) --platform $(PLATFORM) \
		--package.out_dir $(PACKAGE_OUT) \
		-o $(XCLBIN)

# ===================== Emulation Config =====================

emconfig: $(TEMP_DIR)/emconfig.json

$(TEMP_DIR)/emconfig.json:
	mkdir -p $(TEMP_DIR)
	emconfigutil --platform $(PLATFORM) --od $(TEMP_DIR)

# ===================== Run =====================

run: all
ifeq ($(TARGET),$(filter $(TARGET),sw_emu hw_emu))
	cp -f $(TEMP_DIR)/emconfig.json .
	XCL_EMULATION_MODE=$(TARGET) ./$(EXECUTABLE) $(CMD_ARGS)
else
	./$(EXECUTABLE) $(CMD_ARGS)
endif

# ===================== Clean =====================

clean:
	-rm -f $(EXECUTABLE)
	-rm -rf profile_* TempConfig *.rpt *.csv *.log *.jou
	-rm -rf .Xil emconfig.json *v++* dltmp* xmltmp* *.wcfg *.wdb

cleanall: clean
	-rm -rf build_dir* $(IP_REPO) _x* package.* *.xclbin *.xclbin.run_summary
	-rm -rf emulation _vimage start_simulation.sh qemu-memory-_*
