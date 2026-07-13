# Minimal Makefile for multirocket_axis HLS variant.
# Mirrors hydra_axis/Makefile. See that file for usage examples.

VITIS_HLS ?= vitis_hls
PROJECT   := multirocket_axis_inference

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
