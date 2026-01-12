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
HOST_SRCS += host/src/hydra_loader_v2.cpp
HOST_SRCS += host/src/npy_loader.cpp
EXECUTABLE = ./host/hydra_host

CXXFLAGS += -I./host/include -I$(XILINX_XRT)/include -I$(XILINX_VIVADO)/include
LDFLAGS += -L$(XILINX_XRT)/lib -pthread -lOpenCL -lrt -lstdc++

# Arguments for host application
XCLBIN_FILE = build_dir.hw.xilinx_u280_gen3x16_xdma_1_202211_1/krnl.xclbin
MODEL_FILE = models/hydra_insectsound_model.json
TEST_FILE_FULL = models/hydra_insectsound_test.json
TEST_FILE_QUICK = models/hydra_insectsound_test_1000.json

CMD_ARGS = $(XCLBIN_FILE) $(MODEL_FILE) $(TEST_FILE_FULL)

include ./opencl.mk

# Build targets

.PHONY: all clean cleanall help test run run-quick ip

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
	mkdir -p host
	g++ $(CXXFLAGS) $(HOST_SRCS) $(LDFLAGS) -o $(EXECUTABLE)

# XClbin build
build: check-vitis check-xrt $(BINARY_CONTAINERS)

$(BUILD_DIR)/krnl.xclbin: $(BINARY_CONTAINER_hydra_OBJS)
	mkdir -p $(BUILD_DIR)
	v++ -l $(VPP_FLAGS) --temp_dir $(TEMP_DIR) --config ./config.cfg \
		-o $@ $(BINARY_CONTAINER_hydra_OBJS)

xclbin: build

# Run on hardware/emulation
run:
	@echo "========================================="
	@echo "HYDRA FPGA Full Test - InsectSound"
	@echo "========================================="
	@echo "Test samples: 25,000"
	@echo "Expected time: ~2 minutes"
	@echo "========================================="
	@echo ""
	$(EXECUTABLE) $(XCLBIN_FILE) $(MODEL_FILE) $(TEST_FILE_FULL)

run-quick:
	@echo "========================================="
	@echo "HYDRA FPGA Quick Test - InsectSound"
	@echo "========================================="
	@echo "Test samples: 1,000"
	@echo "Expected time: ~5 seconds"
	@echo "========================================="
	@echo ""
	$(EXECUTABLE) $(XCLBIN_FILE) $(MODEL_FILE) $(TEST_FILE_QUICK)

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
	@echo "  make run"
	@echo "      Run full HYDRA FPGA test (25,000 samples, ~2 min)"
	@echo ""
	@echo "  make run-quick"
	@echo "      Run quick HYDRA FPGA test (1,000 samples, ~5 sec)"
	@echo ""
	@echo "  make clean"
	@echo "      Remove temporary files"
	@echo ""
	@echo "  make cleanall"
	@echo "      Remove all generated files"
