// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2023 Advanced Micro Devices, Inc. All Rights Reserved.
// --------------------------------------------------------------------------------
// Tool Version: Vivado v.2023.2 (lin64) Build 4029153 Fri Oct 13 20:13:54 MDT 2023
// Date        : Thu Nov 20 11:33:51 2025
// Host        : wolverine running 64-bit Ubuntu 22.04.5 LTS
// Command     : write_verilog -force -mode synth_stub -rename_top decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix -prefix
//               decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_ ulp_krnl_top_1_0_stub.v
// Design      : ulp_krnl_top_1_0
// Purpose     : Stub declaration of top-level module interface
// Device      : xcu280-fsvh2892-2L-e
// --------------------------------------------------------------------------------

// This empty module with port declaration file causes synthesis tools to infer a black box for IP.
// The synthesis directives are for Synopsys Synplify support to prevent IO buffer insertion.
// Please paste the declaration into a Verilog source file or add the file as an additional source.
(* X_CORE_INFO = "krnl_top,Vivado 2023.2" *)
module decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix(s_axi_control_AWADDR, 
  s_axi_control_AWVALID, s_axi_control_AWREADY, s_axi_control_WDATA, s_axi_control_WSTRB, 
  s_axi_control_WVALID, s_axi_control_WREADY, s_axi_control_BRESP, s_axi_control_BVALID, 
  s_axi_control_BREADY, s_axi_control_ARADDR, s_axi_control_ARVALID, 
  s_axi_control_ARREADY, s_axi_control_RDATA, s_axi_control_RRESP, s_axi_control_RVALID, 
  s_axi_control_RREADY, ap_clk, ap_rst_n, interrupt, m_axi_gmem0_AWID, m_axi_gmem0_AWADDR, 
  m_axi_gmem0_AWLEN, m_axi_gmem0_AWSIZE, m_axi_gmem0_AWBURST, m_axi_gmem0_AWLOCK, 
  m_axi_gmem0_AWREGION, m_axi_gmem0_AWCACHE, m_axi_gmem0_AWPROT, m_axi_gmem0_AWQOS, 
  m_axi_gmem0_AWVALID, m_axi_gmem0_AWREADY, m_axi_gmem0_WID, m_axi_gmem0_WDATA, 
  m_axi_gmem0_WSTRB, m_axi_gmem0_WLAST, m_axi_gmem0_WVALID, m_axi_gmem0_WREADY, 
  m_axi_gmem0_BID, m_axi_gmem0_BRESP, m_axi_gmem0_BVALID, m_axi_gmem0_BREADY, 
  m_axi_gmem0_ARID, m_axi_gmem0_ARADDR, m_axi_gmem0_ARLEN, m_axi_gmem0_ARSIZE, 
  m_axi_gmem0_ARBURST, m_axi_gmem0_ARLOCK, m_axi_gmem0_ARREGION, m_axi_gmem0_ARCACHE, 
  m_axi_gmem0_ARPROT, m_axi_gmem0_ARQOS, m_axi_gmem0_ARVALID, m_axi_gmem0_ARREADY, 
  m_axi_gmem0_RID, m_axi_gmem0_RDATA, m_axi_gmem0_RRESP, m_axi_gmem0_RLAST, 
  m_axi_gmem0_RVALID, m_axi_gmem0_RREADY, m_axi_gmem1_AWID, m_axi_gmem1_AWADDR, 
  m_axi_gmem1_AWLEN, m_axi_gmem1_AWSIZE, m_axi_gmem1_AWBURST, m_axi_gmem1_AWLOCK, 
  m_axi_gmem1_AWREGION, m_axi_gmem1_AWCACHE, m_axi_gmem1_AWPROT, m_axi_gmem1_AWQOS, 
  m_axi_gmem1_AWVALID, m_axi_gmem1_AWREADY, m_axi_gmem1_WID, m_axi_gmem1_WDATA, 
  m_axi_gmem1_WSTRB, m_axi_gmem1_WLAST, m_axi_gmem1_WVALID, m_axi_gmem1_WREADY, 
  m_axi_gmem1_BID, m_axi_gmem1_BRESP, m_axi_gmem1_BVALID, m_axi_gmem1_BREADY, 
  m_axi_gmem1_ARID, m_axi_gmem1_ARADDR, m_axi_gmem1_ARLEN, m_axi_gmem1_ARSIZE, 
  m_axi_gmem1_ARBURST, m_axi_gmem1_ARLOCK, m_axi_gmem1_ARREGION, m_axi_gmem1_ARCACHE, 
  m_axi_gmem1_ARPROT, m_axi_gmem1_ARQOS, m_axi_gmem1_ARVALID, m_axi_gmem1_ARREADY, 
  m_axi_gmem1_RID, m_axi_gmem1_RDATA, m_axi_gmem1_RRESP, m_axi_gmem1_RLAST, 
  m_axi_gmem1_RVALID, m_axi_gmem1_RREADY, m_axi_gmem2_AWID, m_axi_gmem2_AWADDR, 
  m_axi_gmem2_AWLEN, m_axi_gmem2_AWSIZE, m_axi_gmem2_AWBURST, m_axi_gmem2_AWLOCK, 
  m_axi_gmem2_AWREGION, m_axi_gmem2_AWCACHE, m_axi_gmem2_AWPROT, m_axi_gmem2_AWQOS, 
  m_axi_gmem2_AWVALID, m_axi_gmem2_AWREADY, m_axi_gmem2_WID, m_axi_gmem2_WDATA, 
  m_axi_gmem2_WSTRB, m_axi_gmem2_WLAST, m_axi_gmem2_WVALID, m_axi_gmem2_WREADY, 
  m_axi_gmem2_BID, m_axi_gmem2_BRESP, m_axi_gmem2_BVALID, m_axi_gmem2_BREADY, 
  m_axi_gmem2_ARID, m_axi_gmem2_ARADDR, m_axi_gmem2_ARLEN, m_axi_gmem2_ARSIZE, 
  m_axi_gmem2_ARBURST, m_axi_gmem2_ARLOCK, m_axi_gmem2_ARREGION, m_axi_gmem2_ARCACHE, 
  m_axi_gmem2_ARPROT, m_axi_gmem2_ARQOS, m_axi_gmem2_ARVALID, m_axi_gmem2_ARREADY, 
  m_axi_gmem2_RID, m_axi_gmem2_RDATA, m_axi_gmem2_RRESP, m_axi_gmem2_RLAST, 
  m_axi_gmem2_RVALID, m_axi_gmem2_RREADY, m_axi_gmem3_AWID, m_axi_gmem3_AWADDR, 
  m_axi_gmem3_AWLEN, m_axi_gmem3_AWSIZE, m_axi_gmem3_AWBURST, m_axi_gmem3_AWLOCK, 
  m_axi_gmem3_AWREGION, m_axi_gmem3_AWCACHE, m_axi_gmem3_AWPROT, m_axi_gmem3_AWQOS, 
  m_axi_gmem3_AWVALID, m_axi_gmem3_AWREADY, m_axi_gmem3_WID, m_axi_gmem3_WDATA, 
  m_axi_gmem3_WSTRB, m_axi_gmem3_WLAST, m_axi_gmem3_WVALID, m_axi_gmem3_WREADY, 
  m_axi_gmem3_BID, m_axi_gmem3_BRESP, m_axi_gmem3_BVALID, m_axi_gmem3_BREADY, 
  m_axi_gmem3_ARID, m_axi_gmem3_ARADDR, m_axi_gmem3_ARLEN, m_axi_gmem3_ARSIZE, 
  m_axi_gmem3_ARBURST, m_axi_gmem3_ARLOCK, m_axi_gmem3_ARREGION, m_axi_gmem3_ARCACHE, 
  m_axi_gmem3_ARPROT, m_axi_gmem3_ARQOS, m_axi_gmem3_ARVALID, m_axi_gmem3_ARREADY, 
  m_axi_gmem3_RID, m_axi_gmem3_RDATA, m_axi_gmem3_RRESP, m_axi_gmem3_RLAST, 
  m_axi_gmem3_RVALID, m_axi_gmem3_RREADY, m_axi_gmem4_AWID, m_axi_gmem4_AWADDR, 
  m_axi_gmem4_AWLEN, m_axi_gmem4_AWSIZE, m_axi_gmem4_AWBURST, m_axi_gmem4_AWLOCK, 
  m_axi_gmem4_AWREGION, m_axi_gmem4_AWCACHE, m_axi_gmem4_AWPROT, m_axi_gmem4_AWQOS, 
  m_axi_gmem4_AWVALID, m_axi_gmem4_AWREADY, m_axi_gmem4_WID, m_axi_gmem4_WDATA, 
  m_axi_gmem4_WSTRB, m_axi_gmem4_WLAST, m_axi_gmem4_WVALID, m_axi_gmem4_WREADY, 
  m_axi_gmem4_BID, m_axi_gmem4_BRESP, m_axi_gmem4_BVALID, m_axi_gmem4_BREADY, 
  m_axi_gmem4_ARID, m_axi_gmem4_ARADDR, m_axi_gmem4_ARLEN, m_axi_gmem4_ARSIZE, 
  m_axi_gmem4_ARBURST, m_axi_gmem4_ARLOCK, m_axi_gmem4_ARREGION, m_axi_gmem4_ARCACHE, 
  m_axi_gmem4_ARPROT, m_axi_gmem4_ARQOS, m_axi_gmem4_ARVALID, m_axi_gmem4_ARREADY, 
  m_axi_gmem4_RID, m_axi_gmem4_RDATA, m_axi_gmem4_RRESP, m_axi_gmem4_RLAST, 
  m_axi_gmem4_RVALID, m_axi_gmem4_RREADY, m_axi_gmem5_AWID, m_axi_gmem5_AWADDR, 
  m_axi_gmem5_AWLEN, m_axi_gmem5_AWSIZE, m_axi_gmem5_AWBURST, m_axi_gmem5_AWLOCK, 
  m_axi_gmem5_AWREGION, m_axi_gmem5_AWCACHE, m_axi_gmem5_AWPROT, m_axi_gmem5_AWQOS, 
  m_axi_gmem5_AWVALID, m_axi_gmem5_AWREADY, m_axi_gmem5_WID, m_axi_gmem5_WDATA, 
  m_axi_gmem5_WSTRB, m_axi_gmem5_WLAST, m_axi_gmem5_WVALID, m_axi_gmem5_WREADY, 
  m_axi_gmem5_BID, m_axi_gmem5_BRESP, m_axi_gmem5_BVALID, m_axi_gmem5_BREADY, 
  m_axi_gmem5_ARID, m_axi_gmem5_ARADDR, m_axi_gmem5_ARLEN, m_axi_gmem5_ARSIZE, 
  m_axi_gmem5_ARBURST, m_axi_gmem5_ARLOCK, m_axi_gmem5_ARREGION, m_axi_gmem5_ARCACHE, 
  m_axi_gmem5_ARPROT, m_axi_gmem5_ARQOS, m_axi_gmem5_ARVALID, m_axi_gmem5_ARREADY, 
  m_axi_gmem5_RID, m_axi_gmem5_RDATA, m_axi_gmem5_RRESP, m_axi_gmem5_RLAST, 
  m_axi_gmem5_RVALID, m_axi_gmem5_RREADY, m_axi_gmem6_AWID, m_axi_gmem6_AWADDR, 
  m_axi_gmem6_AWLEN, m_axi_gmem6_AWSIZE, m_axi_gmem6_AWBURST, m_axi_gmem6_AWLOCK, 
  m_axi_gmem6_AWREGION, m_axi_gmem6_AWCACHE, m_axi_gmem6_AWPROT, m_axi_gmem6_AWQOS, 
  m_axi_gmem6_AWVALID, m_axi_gmem6_AWREADY, m_axi_gmem6_WID, m_axi_gmem6_WDATA, 
  m_axi_gmem6_WSTRB, m_axi_gmem6_WLAST, m_axi_gmem6_WVALID, m_axi_gmem6_WREADY, 
  m_axi_gmem6_BID, m_axi_gmem6_BRESP, m_axi_gmem6_BVALID, m_axi_gmem6_BREADY, 
  m_axi_gmem6_ARID, m_axi_gmem6_ARADDR, m_axi_gmem6_ARLEN, m_axi_gmem6_ARSIZE, 
  m_axi_gmem6_ARBURST, m_axi_gmem6_ARLOCK, m_axi_gmem6_ARREGION, m_axi_gmem6_ARCACHE, 
  m_axi_gmem6_ARPROT, m_axi_gmem6_ARQOS, m_axi_gmem6_ARVALID, m_axi_gmem6_ARREADY, 
  m_axi_gmem6_RID, m_axi_gmem6_RDATA, m_axi_gmem6_RRESP, m_axi_gmem6_RLAST, 
  m_axi_gmem6_RVALID, m_axi_gmem6_RREADY, m_axi_gmem7_AWID, m_axi_gmem7_AWADDR, 
  m_axi_gmem7_AWLEN, m_axi_gmem7_AWSIZE, m_axi_gmem7_AWBURST, m_axi_gmem7_AWLOCK, 
  m_axi_gmem7_AWREGION, m_axi_gmem7_AWCACHE, m_axi_gmem7_AWPROT, m_axi_gmem7_AWQOS, 
  m_axi_gmem7_AWVALID, m_axi_gmem7_AWREADY, m_axi_gmem7_WID, m_axi_gmem7_WDATA, 
  m_axi_gmem7_WSTRB, m_axi_gmem7_WLAST, m_axi_gmem7_WVALID, m_axi_gmem7_WREADY, 
  m_axi_gmem7_BID, m_axi_gmem7_BRESP, m_axi_gmem7_BVALID, m_axi_gmem7_BREADY, 
  m_axi_gmem7_ARID, m_axi_gmem7_ARADDR, m_axi_gmem7_ARLEN, m_axi_gmem7_ARSIZE, 
  m_axi_gmem7_ARBURST, m_axi_gmem7_ARLOCK, m_axi_gmem7_ARREGION, m_axi_gmem7_ARCACHE, 
  m_axi_gmem7_ARPROT, m_axi_gmem7_ARQOS, m_axi_gmem7_ARVALID, m_axi_gmem7_ARREADY, 
  m_axi_gmem7_RID, m_axi_gmem7_RDATA, m_axi_gmem7_RRESP, m_axi_gmem7_RLAST, 
  m_axi_gmem7_RVALID, m_axi_gmem7_RREADY, m_axi_gmem8_AWID, m_axi_gmem8_AWADDR, 
  m_axi_gmem8_AWLEN, m_axi_gmem8_AWSIZE, m_axi_gmem8_AWBURST, m_axi_gmem8_AWLOCK, 
  m_axi_gmem8_AWREGION, m_axi_gmem8_AWCACHE, m_axi_gmem8_AWPROT, m_axi_gmem8_AWQOS, 
  m_axi_gmem8_AWVALID, m_axi_gmem8_AWREADY, m_axi_gmem8_WID, m_axi_gmem8_WDATA, 
  m_axi_gmem8_WSTRB, m_axi_gmem8_WLAST, m_axi_gmem8_WVALID, m_axi_gmem8_WREADY, 
  m_axi_gmem8_BID, m_axi_gmem8_BRESP, m_axi_gmem8_BVALID, m_axi_gmem8_BREADY, 
  m_axi_gmem8_ARID, m_axi_gmem8_ARADDR, m_axi_gmem8_ARLEN, m_axi_gmem8_ARSIZE, 
  m_axi_gmem8_ARBURST, m_axi_gmem8_ARLOCK, m_axi_gmem8_ARREGION, m_axi_gmem8_ARCACHE, 
  m_axi_gmem8_ARPROT, m_axi_gmem8_ARQOS, m_axi_gmem8_ARVALID, m_axi_gmem8_ARREADY, 
  m_axi_gmem8_RID, m_axi_gmem8_RDATA, m_axi_gmem8_RRESP, m_axi_gmem8_RLAST, 
  m_axi_gmem8_RVALID, m_axi_gmem8_RREADY)
/* synthesis syn_black_box black_box_pad_pin="s_axi_control_AWADDR[7:0],s_axi_control_AWVALID,s_axi_control_AWREADY,s_axi_control_WDATA[31:0],s_axi_control_WSTRB[3:0],s_axi_control_WVALID,s_axi_control_WREADY,s_axi_control_BRESP[1:0],s_axi_control_BVALID,s_axi_control_BREADY,s_axi_control_ARADDR[7:0],s_axi_control_ARVALID,s_axi_control_ARREADY,s_axi_control_RDATA[31:0],s_axi_control_RRESP[1:0],s_axi_control_RVALID,s_axi_control_RREADY,ap_rst_n,interrupt,m_axi_gmem0_AWID[0:0],m_axi_gmem0_AWADDR[63:0],m_axi_gmem0_AWLEN[7:0],m_axi_gmem0_AWSIZE[2:0],m_axi_gmem0_AWBURST[1:0],m_axi_gmem0_AWLOCK[1:0],m_axi_gmem0_AWREGION[3:0],m_axi_gmem0_AWCACHE[3:0],m_axi_gmem0_AWPROT[2:0],m_axi_gmem0_AWQOS[3:0],m_axi_gmem0_AWVALID,m_axi_gmem0_AWREADY,m_axi_gmem0_WID[0:0],m_axi_gmem0_WDATA[31:0],m_axi_gmem0_WSTRB[3:0],m_axi_gmem0_WLAST,m_axi_gmem0_WVALID,m_axi_gmem0_WREADY,m_axi_gmem0_BID[0:0],m_axi_gmem0_BRESP[1:0],m_axi_gmem0_BVALID,m_axi_gmem0_BREADY,m_axi_gmem0_ARID[0:0],m_axi_gmem0_ARADDR[63:0],m_axi_gmem0_ARLEN[7:0],m_axi_gmem0_ARSIZE[2:0],m_axi_gmem0_ARBURST[1:0],m_axi_gmem0_ARLOCK[1:0],m_axi_gmem0_ARREGION[3:0],m_axi_gmem0_ARCACHE[3:0],m_axi_gmem0_ARPROT[2:0],m_axi_gmem0_ARQOS[3:0],m_axi_gmem0_ARVALID,m_axi_gmem0_ARREADY,m_axi_gmem0_RID[0:0],m_axi_gmem0_RDATA[31:0],m_axi_gmem0_RRESP[1:0],m_axi_gmem0_RLAST,m_axi_gmem0_RVALID,m_axi_gmem0_RREADY,m_axi_gmem1_AWID[0:0],m_axi_gmem1_AWADDR[63:0],m_axi_gmem1_AWLEN[7:0],m_axi_gmem1_AWSIZE[2:0],m_axi_gmem1_AWBURST[1:0],m_axi_gmem1_AWLOCK[1:0],m_axi_gmem1_AWREGION[3:0],m_axi_gmem1_AWCACHE[3:0],m_axi_gmem1_AWPROT[2:0],m_axi_gmem1_AWQOS[3:0],m_axi_gmem1_AWVALID,m_axi_gmem1_AWREADY,m_axi_gmem1_WID[0:0],m_axi_gmem1_WDATA[31:0],m_axi_gmem1_WSTRB[3:0],m_axi_gmem1_WLAST,m_axi_gmem1_WVALID,m_axi_gmem1_WREADY,m_axi_gmem1_BID[0:0],m_axi_gmem1_BRESP[1:0],m_axi_gmem1_BVALID,m_axi_gmem1_BREADY,m_axi_gmem1_ARID[0:0],m_axi_gmem1_ARADDR[63:0],m_axi_gmem1_ARLEN[7:0],m_axi_gmem1_ARSIZE[2:0],m_axi_gmem1_ARBURST[1:0],m_axi_gmem1_ARLOCK[1:0],m_axi_gmem1_ARREGION[3:0],m_axi_gmem1_ARCACHE[3:0],m_axi_gmem1_ARPROT[2:0],m_axi_gmem1_ARQOS[3:0],m_axi_gmem1_ARVALID,m_axi_gmem1_ARREADY,m_axi_gmem1_RID[0:0],m_axi_gmem1_RDATA[31:0],m_axi_gmem1_RRESP[1:0],m_axi_gmem1_RLAST,m_axi_gmem1_RVALID,m_axi_gmem1_RREADY,m_axi_gmem2_AWID[0:0],m_axi_gmem2_AWADDR[63:0],m_axi_gmem2_AWLEN[7:0],m_axi_gmem2_AWSIZE[2:0],m_axi_gmem2_AWBURST[1:0],m_axi_gmem2_AWLOCK[1:0],m_axi_gmem2_AWREGION[3:0],m_axi_gmem2_AWCACHE[3:0],m_axi_gmem2_AWPROT[2:0],m_axi_gmem2_AWQOS[3:0],m_axi_gmem2_AWVALID,m_axi_gmem2_AWREADY,m_axi_gmem2_WID[0:0],m_axi_gmem2_WDATA[31:0],m_axi_gmem2_WSTRB[3:0],m_axi_gmem2_WLAST,m_axi_gmem2_WVALID,m_axi_gmem2_WREADY,m_axi_gmem2_BID[0:0],m_axi_gmem2_BRESP[1:0],m_axi_gmem2_BVALID,m_axi_gmem2_BREADY,m_axi_gmem2_ARID[0:0],m_axi_gmem2_ARADDR[63:0],m_axi_gmem2_ARLEN[7:0],m_axi_gmem2_ARSIZE[2:0],m_axi_gmem2_ARBURST[1:0],m_axi_gmem2_ARLOCK[1:0],m_axi_gmem2_ARREGION[3:0],m_axi_gmem2_ARCACHE[3:0],m_axi_gmem2_ARPROT[2:0],m_axi_gmem2_ARQOS[3:0],m_axi_gmem2_ARVALID,m_axi_gmem2_ARREADY,m_axi_gmem2_RID[0:0],m_axi_gmem2_RDATA[31:0],m_axi_gmem2_RRESP[1:0],m_axi_gmem2_RLAST,m_axi_gmem2_RVALID,m_axi_gmem2_RREADY,m_axi_gmem3_AWID[0:0],m_axi_gmem3_AWADDR[63:0],m_axi_gmem3_AWLEN[7:0],m_axi_gmem3_AWSIZE[2:0],m_axi_gmem3_AWBURST[1:0],m_axi_gmem3_AWLOCK[1:0],m_axi_gmem3_AWREGION[3:0],m_axi_gmem3_AWCACHE[3:0],m_axi_gmem3_AWPROT[2:0],m_axi_gmem3_AWQOS[3:0],m_axi_gmem3_AWVALID,m_axi_gmem3_AWREADY,m_axi_gmem3_WID[0:0],m_axi_gmem3_WDATA[31:0],m_axi_gmem3_WSTRB[3:0],m_axi_gmem3_WLAST,m_axi_gmem3_WVALID,m_axi_gmem3_WREADY,m_axi_gmem3_BID[0:0],m_axi_gmem3_BRESP[1:0],m_axi_gmem3_BVALID,m_axi_gmem3_BREADY,m_axi_gmem3_ARID[0:0],m_axi_gmem3_ARADDR[63:0],m_axi_gmem3_ARLEN[7:0],m_axi_gmem3_ARSIZE[2:0],m_axi_gmem3_ARBURST[1:0],m_axi_gmem3_ARLOCK[1:0],m_axi_gmem3_ARREGION[3:0],m_axi_gmem3_ARCACHE[3:0],m_axi_gmem3_ARPROT[2:0],m_axi_gmem3_ARQOS[3:0],m_axi_gmem3_ARVALID,m_axi_gmem3_ARREADY,m_axi_gmem3_RID[0:0],m_axi_gmem3_RDATA[31:0],m_axi_gmem3_RRESP[1:0],m_axi_gmem3_RLAST,m_axi_gmem3_RVALID,m_axi_gmem3_RREADY,m_axi_gmem4_AWID[0:0],m_axi_gmem4_AWADDR[63:0],m_axi_gmem4_AWLEN[7:0],m_axi_gmem4_AWSIZE[2:0],m_axi_gmem4_AWBURST[1:0],m_axi_gmem4_AWLOCK[1:0],m_axi_gmem4_AWREGION[3:0],m_axi_gmem4_AWCACHE[3:0],m_axi_gmem4_AWPROT[2:0],m_axi_gmem4_AWQOS[3:0],m_axi_gmem4_AWVALID,m_axi_gmem4_AWREADY,m_axi_gmem4_WID[0:0],m_axi_gmem4_WDATA[31:0],m_axi_gmem4_WSTRB[3:0],m_axi_gmem4_WLAST,m_axi_gmem4_WVALID,m_axi_gmem4_WREADY,m_axi_gmem4_BID[0:0],m_axi_gmem4_BRESP[1:0],m_axi_gmem4_BVALID,m_axi_gmem4_BREADY,m_axi_gmem4_ARID[0:0],m_axi_gmem4_ARADDR[63:0],m_axi_gmem4_ARLEN[7:0],m_axi_gmem4_ARSIZE[2:0],m_axi_gmem4_ARBURST[1:0],m_axi_gmem4_ARLOCK[1:0],m_axi_gmem4_ARREGION[3:0],m_axi_gmem4_ARCACHE[3:0],m_axi_gmem4_ARPROT[2:0],m_axi_gmem4_ARQOS[3:0],m_axi_gmem4_ARVALID,m_axi_gmem4_ARREADY,m_axi_gmem4_RID[0:0],m_axi_gmem4_RDATA[31:0],m_axi_gmem4_RRESP[1:0],m_axi_gmem4_RLAST,m_axi_gmem4_RVALID,m_axi_gmem4_RREADY,m_axi_gmem5_AWID[0:0],m_axi_gmem5_AWADDR[63:0],m_axi_gmem5_AWLEN[7:0],m_axi_gmem5_AWSIZE[2:0],m_axi_gmem5_AWBURST[1:0],m_axi_gmem5_AWLOCK[1:0],m_axi_gmem5_AWREGION[3:0],m_axi_gmem5_AWCACHE[3:0],m_axi_gmem5_AWPROT[2:0],m_axi_gmem5_AWQOS[3:0],m_axi_gmem5_AWVALID,m_axi_gmem5_AWREADY,m_axi_gmem5_WID[0:0],m_axi_gmem5_WDATA[31:0],m_axi_gmem5_WSTRB[3:0],m_axi_gmem5_WLAST,m_axi_gmem5_WVALID,m_axi_gmem5_WREADY,m_axi_gmem5_BID[0:0],m_axi_gmem5_BRESP[1:0],m_axi_gmem5_BVALID,m_axi_gmem5_BREADY,m_axi_gmem5_ARID[0:0],m_axi_gmem5_ARADDR[63:0],m_axi_gmem5_ARLEN[7:0],m_axi_gmem5_ARSIZE[2:0],m_axi_gmem5_ARBURST[1:0],m_axi_gmem5_ARLOCK[1:0],m_axi_gmem5_ARREGION[3:0],m_axi_gmem5_ARCACHE[3:0],m_axi_gmem5_ARPROT[2:0],m_axi_gmem5_ARQOS[3:0],m_axi_gmem5_ARVALID,m_axi_gmem5_ARREADY,m_axi_gmem5_RID[0:0],m_axi_gmem5_RDATA[31:0],m_axi_gmem5_RRESP[1:0],m_axi_gmem5_RLAST,m_axi_gmem5_RVALID,m_axi_gmem5_RREADY,m_axi_gmem6_AWID[0:0],m_axi_gmem6_AWADDR[63:0],m_axi_gmem6_AWLEN[7:0],m_axi_gmem6_AWSIZE[2:0],m_axi_gmem6_AWBURST[1:0],m_axi_gmem6_AWLOCK[1:0],m_axi_gmem6_AWREGION[3:0],m_axi_gmem6_AWCACHE[3:0],m_axi_gmem6_AWPROT[2:0],m_axi_gmem6_AWQOS[3:0],m_axi_gmem6_AWVALID,m_axi_gmem6_AWREADY,m_axi_gmem6_WID[0:0],m_axi_gmem6_WDATA[31:0],m_axi_gmem6_WSTRB[3:0],m_axi_gmem6_WLAST,m_axi_gmem6_WVALID,m_axi_gmem6_WREADY,m_axi_gmem6_BID[0:0],m_axi_gmem6_BRESP[1:0],m_axi_gmem6_BVALID,m_axi_gmem6_BREADY,m_axi_gmem6_ARID[0:0],m_axi_gmem6_ARADDR[63:0],m_axi_gmem6_ARLEN[7:0],m_axi_gmem6_ARSIZE[2:0],m_axi_gmem6_ARBURST[1:0],m_axi_gmem6_ARLOCK[1:0],m_axi_gmem6_ARREGION[3:0],m_axi_gmem6_ARCACHE[3:0],m_axi_gmem6_ARPROT[2:0],m_axi_gmem6_ARQOS[3:0],m_axi_gmem6_ARVALID,m_axi_gmem6_ARREADY,m_axi_gmem6_RID[0:0],m_axi_gmem6_RDATA[31:0],m_axi_gmem6_RRESP[1:0],m_axi_gmem6_RLAST,m_axi_gmem6_RVALID,m_axi_gmem6_RREADY,m_axi_gmem7_AWID[0:0],m_axi_gmem7_AWADDR[63:0],m_axi_gmem7_AWLEN[7:0],m_axi_gmem7_AWSIZE[2:0],m_axi_gmem7_AWBURST[1:0],m_axi_gmem7_AWLOCK[1:0],m_axi_gmem7_AWREGION[3:0],m_axi_gmem7_AWCACHE[3:0],m_axi_gmem7_AWPROT[2:0],m_axi_gmem7_AWQOS[3:0],m_axi_gmem7_AWVALID,m_axi_gmem7_AWREADY,m_axi_gmem7_WID[0:0],m_axi_gmem7_WDATA[31:0],m_axi_gmem7_WSTRB[3:0],m_axi_gmem7_WLAST,m_axi_gmem7_WVALID,m_axi_gmem7_WREADY,m_axi_gmem7_BID[0:0],m_axi_gmem7_BRESP[1:0],m_axi_gmem7_BVALID,m_axi_gmem7_BREADY,m_axi_gmem7_ARID[0:0],m_axi_gmem7_ARADDR[63:0],m_axi_gmem7_ARLEN[7:0],m_axi_gmem7_ARSIZE[2:0],m_axi_gmem7_ARBURST[1:0],m_axi_gmem7_ARLOCK[1:0],m_axi_gmem7_ARREGION[3:0],m_axi_gmem7_ARCACHE[3:0],m_axi_gmem7_ARPROT[2:0],m_axi_gmem7_ARQOS[3:0],m_axi_gmem7_ARVALID,m_axi_gmem7_ARREADY,m_axi_gmem7_RID[0:0],m_axi_gmem7_RDATA[31:0],m_axi_gmem7_RRESP[1:0],m_axi_gmem7_RLAST,m_axi_gmem7_RVALID,m_axi_gmem7_RREADY,m_axi_gmem8_AWID[0:0],m_axi_gmem8_AWADDR[63:0],m_axi_gmem8_AWLEN[7:0],m_axi_gmem8_AWSIZE[2:0],m_axi_gmem8_AWBURST[1:0],m_axi_gmem8_AWLOCK[1:0],m_axi_gmem8_AWREGION[3:0],m_axi_gmem8_AWCACHE[3:0],m_axi_gmem8_AWPROT[2:0],m_axi_gmem8_AWQOS[3:0],m_axi_gmem8_AWVALID,m_axi_gmem8_AWREADY,m_axi_gmem8_WID[0:0],m_axi_gmem8_WDATA[31:0],m_axi_gmem8_WSTRB[3:0],m_axi_gmem8_WLAST,m_axi_gmem8_WVALID,m_axi_gmem8_WREADY,m_axi_gmem8_BID[0:0],m_axi_gmem8_BRESP[1:0],m_axi_gmem8_BVALID,m_axi_gmem8_BREADY,m_axi_gmem8_ARID[0:0],m_axi_gmem8_ARADDR[63:0],m_axi_gmem8_ARLEN[7:0],m_axi_gmem8_ARSIZE[2:0],m_axi_gmem8_ARBURST[1:0],m_axi_gmem8_ARLOCK[1:0],m_axi_gmem8_ARREGION[3:0],m_axi_gmem8_ARCACHE[3:0],m_axi_gmem8_ARPROT[2:0],m_axi_gmem8_ARQOS[3:0],m_axi_gmem8_ARVALID,m_axi_gmem8_ARREADY,m_axi_gmem8_RID[0:0],m_axi_gmem8_RDATA[31:0],m_axi_gmem8_RRESP[1:0],m_axi_gmem8_RLAST,m_axi_gmem8_RVALID,m_axi_gmem8_RREADY" */
/* synthesis syn_force_seq_prim="ap_clk" */;
  input [7:0]s_axi_control_AWADDR;
  input s_axi_control_AWVALID;
  output s_axi_control_AWREADY;
  input [31:0]s_axi_control_WDATA;
  input [3:0]s_axi_control_WSTRB;
  input s_axi_control_WVALID;
  output s_axi_control_WREADY;
  output [1:0]s_axi_control_BRESP;
  output s_axi_control_BVALID;
  input s_axi_control_BREADY;
  input [7:0]s_axi_control_ARADDR;
  input s_axi_control_ARVALID;
  output s_axi_control_ARREADY;
  output [31:0]s_axi_control_RDATA;
  output [1:0]s_axi_control_RRESP;
  output s_axi_control_RVALID;
  input s_axi_control_RREADY;
  input ap_clk /* synthesis syn_isclock = 1 */;
  input ap_rst_n;
  output interrupt;
  output [0:0]m_axi_gmem0_AWID;
  output [63:0]m_axi_gmem0_AWADDR;
  output [7:0]m_axi_gmem0_AWLEN;
  output [2:0]m_axi_gmem0_AWSIZE;
  output [1:0]m_axi_gmem0_AWBURST;
  output [1:0]m_axi_gmem0_AWLOCK;
  output [3:0]m_axi_gmem0_AWREGION;
  output [3:0]m_axi_gmem0_AWCACHE;
  output [2:0]m_axi_gmem0_AWPROT;
  output [3:0]m_axi_gmem0_AWQOS;
  output m_axi_gmem0_AWVALID;
  input m_axi_gmem0_AWREADY;
  output [0:0]m_axi_gmem0_WID;
  output [31:0]m_axi_gmem0_WDATA;
  output [3:0]m_axi_gmem0_WSTRB;
  output m_axi_gmem0_WLAST;
  output m_axi_gmem0_WVALID;
  input m_axi_gmem0_WREADY;
  input [0:0]m_axi_gmem0_BID;
  input [1:0]m_axi_gmem0_BRESP;
  input m_axi_gmem0_BVALID;
  output m_axi_gmem0_BREADY;
  output [0:0]m_axi_gmem0_ARID;
  output [63:0]m_axi_gmem0_ARADDR;
  output [7:0]m_axi_gmem0_ARLEN;
  output [2:0]m_axi_gmem0_ARSIZE;
  output [1:0]m_axi_gmem0_ARBURST;
  output [1:0]m_axi_gmem0_ARLOCK;
  output [3:0]m_axi_gmem0_ARREGION;
  output [3:0]m_axi_gmem0_ARCACHE;
  output [2:0]m_axi_gmem0_ARPROT;
  output [3:0]m_axi_gmem0_ARQOS;
  output m_axi_gmem0_ARVALID;
  input m_axi_gmem0_ARREADY;
  input [0:0]m_axi_gmem0_RID;
  input [31:0]m_axi_gmem0_RDATA;
  input [1:0]m_axi_gmem0_RRESP;
  input m_axi_gmem0_RLAST;
  input m_axi_gmem0_RVALID;
  output m_axi_gmem0_RREADY;
  output [0:0]m_axi_gmem1_AWID;
  output [63:0]m_axi_gmem1_AWADDR;
  output [7:0]m_axi_gmem1_AWLEN;
  output [2:0]m_axi_gmem1_AWSIZE;
  output [1:0]m_axi_gmem1_AWBURST;
  output [1:0]m_axi_gmem1_AWLOCK;
  output [3:0]m_axi_gmem1_AWREGION;
  output [3:0]m_axi_gmem1_AWCACHE;
  output [2:0]m_axi_gmem1_AWPROT;
  output [3:0]m_axi_gmem1_AWQOS;
  output m_axi_gmem1_AWVALID;
  input m_axi_gmem1_AWREADY;
  output [0:0]m_axi_gmem1_WID;
  output [31:0]m_axi_gmem1_WDATA;
  output [3:0]m_axi_gmem1_WSTRB;
  output m_axi_gmem1_WLAST;
  output m_axi_gmem1_WVALID;
  input m_axi_gmem1_WREADY;
  input [0:0]m_axi_gmem1_BID;
  input [1:0]m_axi_gmem1_BRESP;
  input m_axi_gmem1_BVALID;
  output m_axi_gmem1_BREADY;
  output [0:0]m_axi_gmem1_ARID;
  output [63:0]m_axi_gmem1_ARADDR;
  output [7:0]m_axi_gmem1_ARLEN;
  output [2:0]m_axi_gmem1_ARSIZE;
  output [1:0]m_axi_gmem1_ARBURST;
  output [1:0]m_axi_gmem1_ARLOCK;
  output [3:0]m_axi_gmem1_ARREGION;
  output [3:0]m_axi_gmem1_ARCACHE;
  output [2:0]m_axi_gmem1_ARPROT;
  output [3:0]m_axi_gmem1_ARQOS;
  output m_axi_gmem1_ARVALID;
  input m_axi_gmem1_ARREADY;
  input [0:0]m_axi_gmem1_RID;
  input [31:0]m_axi_gmem1_RDATA;
  input [1:0]m_axi_gmem1_RRESP;
  input m_axi_gmem1_RLAST;
  input m_axi_gmem1_RVALID;
  output m_axi_gmem1_RREADY;
  output [0:0]m_axi_gmem2_AWID;
  output [63:0]m_axi_gmem2_AWADDR;
  output [7:0]m_axi_gmem2_AWLEN;
  output [2:0]m_axi_gmem2_AWSIZE;
  output [1:0]m_axi_gmem2_AWBURST;
  output [1:0]m_axi_gmem2_AWLOCK;
  output [3:0]m_axi_gmem2_AWREGION;
  output [3:0]m_axi_gmem2_AWCACHE;
  output [2:0]m_axi_gmem2_AWPROT;
  output [3:0]m_axi_gmem2_AWQOS;
  output m_axi_gmem2_AWVALID;
  input m_axi_gmem2_AWREADY;
  output [0:0]m_axi_gmem2_WID;
  output [31:0]m_axi_gmem2_WDATA;
  output [3:0]m_axi_gmem2_WSTRB;
  output m_axi_gmem2_WLAST;
  output m_axi_gmem2_WVALID;
  input m_axi_gmem2_WREADY;
  input [0:0]m_axi_gmem2_BID;
  input [1:0]m_axi_gmem2_BRESP;
  input m_axi_gmem2_BVALID;
  output m_axi_gmem2_BREADY;
  output [0:0]m_axi_gmem2_ARID;
  output [63:0]m_axi_gmem2_ARADDR;
  output [7:0]m_axi_gmem2_ARLEN;
  output [2:0]m_axi_gmem2_ARSIZE;
  output [1:0]m_axi_gmem2_ARBURST;
  output [1:0]m_axi_gmem2_ARLOCK;
  output [3:0]m_axi_gmem2_ARREGION;
  output [3:0]m_axi_gmem2_ARCACHE;
  output [2:0]m_axi_gmem2_ARPROT;
  output [3:0]m_axi_gmem2_ARQOS;
  output m_axi_gmem2_ARVALID;
  input m_axi_gmem2_ARREADY;
  input [0:0]m_axi_gmem2_RID;
  input [31:0]m_axi_gmem2_RDATA;
  input [1:0]m_axi_gmem2_RRESP;
  input m_axi_gmem2_RLAST;
  input m_axi_gmem2_RVALID;
  output m_axi_gmem2_RREADY;
  output [0:0]m_axi_gmem3_AWID;
  output [63:0]m_axi_gmem3_AWADDR;
  output [7:0]m_axi_gmem3_AWLEN;
  output [2:0]m_axi_gmem3_AWSIZE;
  output [1:0]m_axi_gmem3_AWBURST;
  output [1:0]m_axi_gmem3_AWLOCK;
  output [3:0]m_axi_gmem3_AWREGION;
  output [3:0]m_axi_gmem3_AWCACHE;
  output [2:0]m_axi_gmem3_AWPROT;
  output [3:0]m_axi_gmem3_AWQOS;
  output m_axi_gmem3_AWVALID;
  input m_axi_gmem3_AWREADY;
  output [0:0]m_axi_gmem3_WID;
  output [31:0]m_axi_gmem3_WDATA;
  output [3:0]m_axi_gmem3_WSTRB;
  output m_axi_gmem3_WLAST;
  output m_axi_gmem3_WVALID;
  input m_axi_gmem3_WREADY;
  input [0:0]m_axi_gmem3_BID;
  input [1:0]m_axi_gmem3_BRESP;
  input m_axi_gmem3_BVALID;
  output m_axi_gmem3_BREADY;
  output [0:0]m_axi_gmem3_ARID;
  output [63:0]m_axi_gmem3_ARADDR;
  output [7:0]m_axi_gmem3_ARLEN;
  output [2:0]m_axi_gmem3_ARSIZE;
  output [1:0]m_axi_gmem3_ARBURST;
  output [1:0]m_axi_gmem3_ARLOCK;
  output [3:0]m_axi_gmem3_ARREGION;
  output [3:0]m_axi_gmem3_ARCACHE;
  output [2:0]m_axi_gmem3_ARPROT;
  output [3:0]m_axi_gmem3_ARQOS;
  output m_axi_gmem3_ARVALID;
  input m_axi_gmem3_ARREADY;
  input [0:0]m_axi_gmem3_RID;
  input [31:0]m_axi_gmem3_RDATA;
  input [1:0]m_axi_gmem3_RRESP;
  input m_axi_gmem3_RLAST;
  input m_axi_gmem3_RVALID;
  output m_axi_gmem3_RREADY;
  output [0:0]m_axi_gmem4_AWID;
  output [63:0]m_axi_gmem4_AWADDR;
  output [7:0]m_axi_gmem4_AWLEN;
  output [2:0]m_axi_gmem4_AWSIZE;
  output [1:0]m_axi_gmem4_AWBURST;
  output [1:0]m_axi_gmem4_AWLOCK;
  output [3:0]m_axi_gmem4_AWREGION;
  output [3:0]m_axi_gmem4_AWCACHE;
  output [2:0]m_axi_gmem4_AWPROT;
  output [3:0]m_axi_gmem4_AWQOS;
  output m_axi_gmem4_AWVALID;
  input m_axi_gmem4_AWREADY;
  output [0:0]m_axi_gmem4_WID;
  output [31:0]m_axi_gmem4_WDATA;
  output [3:0]m_axi_gmem4_WSTRB;
  output m_axi_gmem4_WLAST;
  output m_axi_gmem4_WVALID;
  input m_axi_gmem4_WREADY;
  input [0:0]m_axi_gmem4_BID;
  input [1:0]m_axi_gmem4_BRESP;
  input m_axi_gmem4_BVALID;
  output m_axi_gmem4_BREADY;
  output [0:0]m_axi_gmem4_ARID;
  output [63:0]m_axi_gmem4_ARADDR;
  output [7:0]m_axi_gmem4_ARLEN;
  output [2:0]m_axi_gmem4_ARSIZE;
  output [1:0]m_axi_gmem4_ARBURST;
  output [1:0]m_axi_gmem4_ARLOCK;
  output [3:0]m_axi_gmem4_ARREGION;
  output [3:0]m_axi_gmem4_ARCACHE;
  output [2:0]m_axi_gmem4_ARPROT;
  output [3:0]m_axi_gmem4_ARQOS;
  output m_axi_gmem4_ARVALID;
  input m_axi_gmem4_ARREADY;
  input [0:0]m_axi_gmem4_RID;
  input [31:0]m_axi_gmem4_RDATA;
  input [1:0]m_axi_gmem4_RRESP;
  input m_axi_gmem4_RLAST;
  input m_axi_gmem4_RVALID;
  output m_axi_gmem4_RREADY;
  output [0:0]m_axi_gmem5_AWID;
  output [63:0]m_axi_gmem5_AWADDR;
  output [7:0]m_axi_gmem5_AWLEN;
  output [2:0]m_axi_gmem5_AWSIZE;
  output [1:0]m_axi_gmem5_AWBURST;
  output [1:0]m_axi_gmem5_AWLOCK;
  output [3:0]m_axi_gmem5_AWREGION;
  output [3:0]m_axi_gmem5_AWCACHE;
  output [2:0]m_axi_gmem5_AWPROT;
  output [3:0]m_axi_gmem5_AWQOS;
  output m_axi_gmem5_AWVALID;
  input m_axi_gmem5_AWREADY;
  output [0:0]m_axi_gmem5_WID;
  output [31:0]m_axi_gmem5_WDATA;
  output [3:0]m_axi_gmem5_WSTRB;
  output m_axi_gmem5_WLAST;
  output m_axi_gmem5_WVALID;
  input m_axi_gmem5_WREADY;
  input [0:0]m_axi_gmem5_BID;
  input [1:0]m_axi_gmem5_BRESP;
  input m_axi_gmem5_BVALID;
  output m_axi_gmem5_BREADY;
  output [0:0]m_axi_gmem5_ARID;
  output [63:0]m_axi_gmem5_ARADDR;
  output [7:0]m_axi_gmem5_ARLEN;
  output [2:0]m_axi_gmem5_ARSIZE;
  output [1:0]m_axi_gmem5_ARBURST;
  output [1:0]m_axi_gmem5_ARLOCK;
  output [3:0]m_axi_gmem5_ARREGION;
  output [3:0]m_axi_gmem5_ARCACHE;
  output [2:0]m_axi_gmem5_ARPROT;
  output [3:0]m_axi_gmem5_ARQOS;
  output m_axi_gmem5_ARVALID;
  input m_axi_gmem5_ARREADY;
  input [0:0]m_axi_gmem5_RID;
  input [31:0]m_axi_gmem5_RDATA;
  input [1:0]m_axi_gmem5_RRESP;
  input m_axi_gmem5_RLAST;
  input m_axi_gmem5_RVALID;
  output m_axi_gmem5_RREADY;
  output [0:0]m_axi_gmem6_AWID;
  output [63:0]m_axi_gmem6_AWADDR;
  output [7:0]m_axi_gmem6_AWLEN;
  output [2:0]m_axi_gmem6_AWSIZE;
  output [1:0]m_axi_gmem6_AWBURST;
  output [1:0]m_axi_gmem6_AWLOCK;
  output [3:0]m_axi_gmem6_AWREGION;
  output [3:0]m_axi_gmem6_AWCACHE;
  output [2:0]m_axi_gmem6_AWPROT;
  output [3:0]m_axi_gmem6_AWQOS;
  output m_axi_gmem6_AWVALID;
  input m_axi_gmem6_AWREADY;
  output [0:0]m_axi_gmem6_WID;
  output [31:0]m_axi_gmem6_WDATA;
  output [3:0]m_axi_gmem6_WSTRB;
  output m_axi_gmem6_WLAST;
  output m_axi_gmem6_WVALID;
  input m_axi_gmem6_WREADY;
  input [0:0]m_axi_gmem6_BID;
  input [1:0]m_axi_gmem6_BRESP;
  input m_axi_gmem6_BVALID;
  output m_axi_gmem6_BREADY;
  output [0:0]m_axi_gmem6_ARID;
  output [63:0]m_axi_gmem6_ARADDR;
  output [7:0]m_axi_gmem6_ARLEN;
  output [2:0]m_axi_gmem6_ARSIZE;
  output [1:0]m_axi_gmem6_ARBURST;
  output [1:0]m_axi_gmem6_ARLOCK;
  output [3:0]m_axi_gmem6_ARREGION;
  output [3:0]m_axi_gmem6_ARCACHE;
  output [2:0]m_axi_gmem6_ARPROT;
  output [3:0]m_axi_gmem6_ARQOS;
  output m_axi_gmem6_ARVALID;
  input m_axi_gmem6_ARREADY;
  input [0:0]m_axi_gmem6_RID;
  input [31:0]m_axi_gmem6_RDATA;
  input [1:0]m_axi_gmem6_RRESP;
  input m_axi_gmem6_RLAST;
  input m_axi_gmem6_RVALID;
  output m_axi_gmem6_RREADY;
  output [0:0]m_axi_gmem7_AWID;
  output [63:0]m_axi_gmem7_AWADDR;
  output [7:0]m_axi_gmem7_AWLEN;
  output [2:0]m_axi_gmem7_AWSIZE;
  output [1:0]m_axi_gmem7_AWBURST;
  output [1:0]m_axi_gmem7_AWLOCK;
  output [3:0]m_axi_gmem7_AWREGION;
  output [3:0]m_axi_gmem7_AWCACHE;
  output [2:0]m_axi_gmem7_AWPROT;
  output [3:0]m_axi_gmem7_AWQOS;
  output m_axi_gmem7_AWVALID;
  input m_axi_gmem7_AWREADY;
  output [0:0]m_axi_gmem7_WID;
  output [31:0]m_axi_gmem7_WDATA;
  output [3:0]m_axi_gmem7_WSTRB;
  output m_axi_gmem7_WLAST;
  output m_axi_gmem7_WVALID;
  input m_axi_gmem7_WREADY;
  input [0:0]m_axi_gmem7_BID;
  input [1:0]m_axi_gmem7_BRESP;
  input m_axi_gmem7_BVALID;
  output m_axi_gmem7_BREADY;
  output [0:0]m_axi_gmem7_ARID;
  output [63:0]m_axi_gmem7_ARADDR;
  output [7:0]m_axi_gmem7_ARLEN;
  output [2:0]m_axi_gmem7_ARSIZE;
  output [1:0]m_axi_gmem7_ARBURST;
  output [1:0]m_axi_gmem7_ARLOCK;
  output [3:0]m_axi_gmem7_ARREGION;
  output [3:0]m_axi_gmem7_ARCACHE;
  output [2:0]m_axi_gmem7_ARPROT;
  output [3:0]m_axi_gmem7_ARQOS;
  output m_axi_gmem7_ARVALID;
  input m_axi_gmem7_ARREADY;
  input [0:0]m_axi_gmem7_RID;
  input [31:0]m_axi_gmem7_RDATA;
  input [1:0]m_axi_gmem7_RRESP;
  input m_axi_gmem7_RLAST;
  input m_axi_gmem7_RVALID;
  output m_axi_gmem7_RREADY;
  output [0:0]m_axi_gmem8_AWID;
  output [63:0]m_axi_gmem8_AWADDR;
  output [7:0]m_axi_gmem8_AWLEN;
  output [2:0]m_axi_gmem8_AWSIZE;
  output [1:0]m_axi_gmem8_AWBURST;
  output [1:0]m_axi_gmem8_AWLOCK;
  output [3:0]m_axi_gmem8_AWREGION;
  output [3:0]m_axi_gmem8_AWCACHE;
  output [2:0]m_axi_gmem8_AWPROT;
  output [3:0]m_axi_gmem8_AWQOS;
  output m_axi_gmem8_AWVALID;
  input m_axi_gmem8_AWREADY;
  output [0:0]m_axi_gmem8_WID;
  output [31:0]m_axi_gmem8_WDATA;
  output [3:0]m_axi_gmem8_WSTRB;
  output m_axi_gmem8_WLAST;
  output m_axi_gmem8_WVALID;
  input m_axi_gmem8_WREADY;
  input [0:0]m_axi_gmem8_BID;
  input [1:0]m_axi_gmem8_BRESP;
  input m_axi_gmem8_BVALID;
  output m_axi_gmem8_BREADY;
  output [0:0]m_axi_gmem8_ARID;
  output [63:0]m_axi_gmem8_ARADDR;
  output [7:0]m_axi_gmem8_ARLEN;
  output [2:0]m_axi_gmem8_ARSIZE;
  output [1:0]m_axi_gmem8_ARBURST;
  output [1:0]m_axi_gmem8_ARLOCK;
  output [3:0]m_axi_gmem8_ARREGION;
  output [3:0]m_axi_gmem8_ARCACHE;
  output [2:0]m_axi_gmem8_ARPROT;
  output [3:0]m_axi_gmem8_ARQOS;
  output m_axi_gmem8_ARVALID;
  input m_axi_gmem8_ARREADY;
  input [0:0]m_axi_gmem8_RID;
  input [31:0]m_axi_gmem8_RDATA;
  input [1:0]m_axi_gmem8_RRESP;
  input m_axi_gmem8_RLAST;
  input m_axi_gmem8_RVALID;
  output m_axi_gmem8_RREADY;
endmodule
