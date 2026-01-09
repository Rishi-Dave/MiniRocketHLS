# HYDRA FPGA Makefile
# Adapted from MiniRocket/MultiRocket Vitis build system

TARGET := hw
HOST_ARCH := x86
SYSROOT :=

include ./utils.mk

# Platform selection
PLATFORM := $(DEVICE)
ifeq ($(PLATFORM),)
$(error PLATFORM not set. Please set DEVICE or PLATFORM variable)
endif

# Extract XSA name from platform path
XSA := $(strip $(patsubst %.xpfm, %, $(shell basename $(PLATFORM) 2>/dev/null)))

TEMP_DIR := ./_x.$(TARGET).$(XSA)
BUILD_DIR := ./build_dir.$(TARGET).$(XSA)

# IP repository
IP_REPO := ./build/iprepo

# v++ flags
VPP_FLAGS := --platform $(PLATFORM) -t $(TARGET) --save-temps

# Kernel XO file
BINARY_CONTAINER_hydra_OBJS += $(IP_REPO)/hydra_inference.xo
BINARY_CONTAINERS += $(BUILD_DIR)/krnl.xclbin

# Host executable
HOST_SRCS += host/src/hydra_host.cpp
HOST_SRCS += host/src/hydra_loader.cpp

CXXFLAGS += -I./host/include -I$(XILINX_XRT)/include -I$(XILINX_VIVADO)/include
LDFLAGS += -L$(XILINX_XRT)/lib -pthread -lOpenCL -lrt -lstdc++

# Arguments for host application
CMD_ARGS = $(BUILD_DIR)/krnl.xclbin models/hydra_model.json models/hydra_test.json

include ./opencl.mk

# Build targets

.PHONY: all clean cleanall help test run ip

all: check-devices check-vitis check-xrt host xclbin

# Check environment
.PHONY: check-platform check-device check-vitis check-xrt

check-platform:
ifndef PLATFORM
	$(error PLATFORM not set. Please set the PLATFORM properly)
endif

check-device:
ifndef DEVICE
	$(error DEVICE not set. Please set the DEVICE properly)
endif

check-vitis:
ifndef XILINX_VITIS
	$(error XILINX_VITIS not set. Please source Vitis settings64.sh)
endif

check-xrt:
ifndef XILINX_XRT
	$(error XILINX_XRT not set. Please source XRT setup script)
endif

# IP synthesis using CMake/Vitis HLS
ip: check-vitis $(BINARY_CONTAINER_hydra_OBJS)

$(IP_REPO)/hydra_inference.xo: hydra/src/hydra.cpp hydra/src/hydra_pooling.cpp hydra/include/hydra.hpp
	mkdir -p build
	cd build && cmake .. && make installip.hydra_inference

# Host application
host: check-xrt $(HOST_SRCS)
	mkdir -p $(BUILD_DIR)
	g++ $(CXXFLAGS) $(HOST_SRCS) $(LDFLAGS) -o $(BUILD_DIR)/hydra_host

# XClbin build
build: check-vitis check-xrt $(BINARY_CONTAINERS)

$(BUILD_DIR)/krnl.xclbin: $(BINARY_CONTAINER_hydra_OBJS)
	mkdir -p $(BUILD_DIR)
	v++ -l $(VPP_FLAGS) --temp_dir $(TEMP_DIR) --config ./config.cfg \
		-o $@ $(BINARY_CONTAINER_hydra_OBJS)

xclbin: build

# Run on hardware/emulation
run: all
	cd $(BUILD_DIR) && ./hydra_host $(CMD_ARGS)

test: run

# Clean targets
clean:
	-$(RMDIR) $(TEMP_DIR)
	-$(RMDIR) $(BUILD_DIR)
	-$(RMDIR) _x
	-$(RMDIR) .Xil
	-$(RMDIR) *.log *.jou

cleanall: clean
	-$(RMDIR) build
	-$(RMDIR) *.xclbin
	-$(RMDIR) *.ltx
	-$(RMDIR) *.info

# Help
help:
	@echo "Makefile Usage:"
	@echo ""
	@echo "  make all TARGET=<hw/hw_emu/sw_emu>"
	@echo "      Build complete system (IP + host + xclbin)"
	@echo ""
	@echo "  make ip"
	@echo "      Synthesize HLS IP (hydra_inference.xo)"
	@echo ""
	@echo "  make host"
	@echo "      Compile host application"
	@echo ""
	@echo "  make build TARGET=<hw/hw_emu/sw_emu>"
	@echo "      Link kernel and generate xclbin"
	@echo ""
	@echo "  make run TARGET=<hw/hw_emu/sw_emu>"
	@echo "      Run host application"
	@echo ""
	@echo "  make clean"
	@echo "      Remove temporary files"
	@echo ""
	@echo "  make cleanall"
	@echo "      Remove all generated files"
