VIVADO := $(XILINX_VIVADO)/bin/vivado
# CMAC KERNEL (SYS VERILOG)
$(CMAC_XO_REPO)/${CMAC_KRNL}.xo: cmac_krnl/cmac_krnl.xml cmac_krnl/package_cmac_krnl.tcl scripts/gen_xo.tcl cmac_krnl/src/hdl/*.sv
	mkdir -p $(CMAC_XO_REPO)
	$(VIVADO) -mode batch -source scripts/gen_xo.tcl -tclargs $(CMAC_XO_REPO)/${CMAC_KRNL}.xo ${CMAC_KRNL} $(TARGET) $(PLATFORM) $(XSA) cmac_krnl/cmac_krnl.xml cmac_krnl/package_cmac_krnl.tcl