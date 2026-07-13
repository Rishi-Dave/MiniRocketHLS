# Minimal Makefile for hydra_axis HLS variant.
#
# Targets:
#   make csim     — run C simulation (smoke test, ~30 s)
#   make synth    — run csynth_design only (no co-sim, no IP export)
#   make ip       — produce hydra_axis_inference.xo for v++ linking
#   make clean    — remove HLS project
#
# For the full xclbin (CMAC + NetLayer + drop module + this kernel), drop
# hydra_axis_inference.xo into a v++ link command using
# config_axis.cfg from this directory; see fpga-network/Makefile for the
# v++ invocation pattern that the OCT build job mimics.

VITIS_HLS ?= vitis_hls
PROJECT   := hydra_axis_inference

.PHONY: csim synth ip cosim clean

csim:
	$(VITIS_HLS) -f make.tcl -tclargs csim

synth:
	$(VITIS_HLS) -f make.tcl -tclargs synthesis

ip:
	$(VITIS_HLS) -f make.tcl -tclargs ip

cosim:
	$(VITIS_HLS) -f make.tcl -tclargs cosim

clean:
	rm -rf $(PROJECT) vitis_hls.log *.xo
