// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2023 Advanced Micro Devices, Inc. All Rights Reserved.
// --------------------------------------------------------------------------------
// Tool Version: Vivado v.2023.2 (lin64) Build 4029153 Fri Oct 13 20:13:54 MDT 2023
// Date        : Thu Nov 20 11:44:28 2025
// Host        : wolverine running 64-bit Ubuntu 22.04.5 LTS
// Command     : write_verilog -force -mode funcsim -rename_top decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix -prefix
//               decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_ bd_58f6_lut_buffer_0_sim_netlist.v
// Design      : bd_58f6_lut_buffer_0
// Purpose     : This verilog netlist is a functional simulation representation of the design and should not be modified
//               or synthesized. This netlist cannot be used for SDF annotated simulation.
// Device      : xcu280-fsvh2892-2L-e
// --------------------------------------------------------------------------------
`timescale 1 ps / 1 ps

(* CHECK_LICENSE_TYPE = "bd_58f6_lut_buffer_0,lut_buffer_v2_0_0_lut_buffer,{}" *) (* DowngradeIPIdentifiedWarnings = "yes" *) (* X_CORE_INFO = "lut_buffer_v2_0_0_lut_buffer,Vivado 2023.2" *) 
(* NotValidForBitStream *)
module decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix
   (tdi_i,
    tms_i,
    tck_i,
    drck_i,
    sel_i,
    shift_i,
    update_i,
    capture_i,
    runtest_i,
    reset_i,
    bscanid_en_i,
    tdo_o,
    tdi_o,
    tms_o,
    tck_o,
    drck_o,
    sel_o,
    shift_o,
    update_o,
    capture_o,
    runtest_o,
    reset_o,
    bscanid_en_o,
    tdo_i);
  (* X_INTERFACE_INFO = "xilinx.com:interface:bscan:1.0 s_bscan TDI" *) input tdi_i;
  (* X_INTERFACE_INFO = "xilinx.com:interface:bscan:1.0 s_bscan TMS" *) input tms_i;
  (* X_INTERFACE_INFO = "xilinx.com:interface:bscan:1.0 s_bscan TCK" *) input tck_i;
  (* X_INTERFACE_INFO = "xilinx.com:interface:bscan:1.0 s_bscan DRCK" *) input drck_i;
  (* X_INTERFACE_INFO = "xilinx.com:interface:bscan:1.0 s_bscan SEL" *) input sel_i;
  (* X_INTERFACE_INFO = "xilinx.com:interface:bscan:1.0 s_bscan SHIFT" *) input shift_i;
  (* X_INTERFACE_INFO = "xilinx.com:interface:bscan:1.0 s_bscan UPDATE" *) input update_i;
  (* X_INTERFACE_INFO = "xilinx.com:interface:bscan:1.0 s_bscan CAPTURE" *) input capture_i;
  (* X_INTERFACE_INFO = "xilinx.com:interface:bscan:1.0 s_bscan RUNTEST" *) input runtest_i;
  (* X_INTERFACE_INFO = "xilinx.com:interface:bscan:1.0 s_bscan RESET" *) input reset_i;
  (* X_INTERFACE_INFO = "xilinx.com:interface:bscan:1.0 s_bscan BSCANID_EN" *) input bscanid_en_i;
  (* X_INTERFACE_INFO = "xilinx.com:interface:bscan:1.0 s_bscan TDO" *) output tdo_o;
  (* X_INTERFACE_INFO = "xilinx.com:interface:bscan:1.0 m_bscan TDI" *) output tdi_o;
  (* X_INTERFACE_INFO = "xilinx.com:interface:bscan:1.0 m_bscan TMS" *) output tms_o;
  (* X_INTERFACE_INFO = "xilinx.com:interface:bscan:1.0 m_bscan TCK" *) output tck_o;
  (* X_INTERFACE_INFO = "xilinx.com:interface:bscan:1.0 m_bscan DRCK" *) output drck_o;
  (* X_INTERFACE_INFO = "xilinx.com:interface:bscan:1.0 m_bscan SEL" *) output sel_o;
  (* X_INTERFACE_INFO = "xilinx.com:interface:bscan:1.0 m_bscan SHIFT" *) output shift_o;
  (* X_INTERFACE_INFO = "xilinx.com:interface:bscan:1.0 m_bscan UPDATE" *) output update_o;
  (* X_INTERFACE_INFO = "xilinx.com:interface:bscan:1.0 m_bscan CAPTURE" *) output capture_o;
  (* X_INTERFACE_INFO = "xilinx.com:interface:bscan:1.0 m_bscan RUNTEST" *) output runtest_o;
  (* X_INTERFACE_INFO = "xilinx.com:interface:bscan:1.0 m_bscan RESET" *) output reset_o;
  (* X_INTERFACE_INFO = "xilinx.com:interface:bscan:1.0 m_bscan BSCANID_EN" *) output bscanid_en_o;
  (* X_INTERFACE_INFO = "xilinx.com:interface:bscan:1.0 m_bscan TDO" *) input tdo_i;

  wire bscanid_en_i;
  wire bscanid_en_o;
  wire capture_i;
  wire capture_o;
  wire drck_i;
  wire drck_o;
  wire reset_i;
  wire reset_o;
  wire runtest_i;
  wire runtest_o;
  wire sel_i;
  wire sel_o;
  wire shift_i;
  wire shift_o;
  wire tck_i;
  wire tck_o;
  wire tdi_i;
  wire tdi_o;
  wire tdo_i;
  wire tdo_o;
  wire tms_i;
  wire tms_o;
  wire update_i;
  wire update_o;
  wire [31:0]NLW_inst_bscanid_o_UNCONNECTED;

  (* C_EN_BSCANID_VEC = "0" *) 
  (* DONT_TOUCH *) 
  (* KEEP_HIERARCHY = "soft" *) 
  (* is_du_within_envelope = "true" *) 
  decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_lut_buffer_v2_0_0_lut_buffer inst
       (.bscanid_en_i(bscanid_en_i),
        .bscanid_en_o(bscanid_en_o),
        .bscanid_i({1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0}),
        .bscanid_o(NLW_inst_bscanid_o_UNCONNECTED[31:0]),
        .capture_i(capture_i),
        .capture_o(capture_o),
        .drck_i(drck_i),
        .drck_o(drck_o),
        .reset_i(reset_i),
        .reset_o(reset_o),
        .runtest_i(runtest_i),
        .runtest_o(runtest_o),
        .sel_i(sel_i),
        .sel_o(sel_o),
        .shift_i(shift_i),
        .shift_o(shift_o),
        .tck_i(tck_i),
        .tck_o(tck_o),
        .tdi_i(tdi_i),
        .tdi_o(tdi_o),
        .tdo_i(tdo_i),
        .tdo_o(tdo_o),
        .tms_i(tms_i),
        .tms_o(tms_o),
        .update_i(update_i),
        .update_o(update_o));
endmodule
`pragma protect begin_protected
`pragma protect version = 1
`pragma protect encrypt_agent = "XILINX"
`pragma protect encrypt_agent_info = "Xilinx Encryption Tool 2023.2"
`pragma protect key_keyowner="Synopsys", key_keyname="SNPS-VCS-RSA-2", key_method="rsa"
`pragma protect encoding = (enctype="BASE64", line_length=76, bytes=128)
`pragma protect key_block
Y0xh9134RsBng5etaGXwBVUEA8J2bWMgUTQDTIAitarHeeOiZsCqvqnqSMBLJzSAT8Clo2xSEdDb
f2T0bmuallNuAd4vpVjfpqZxOdLmUV+3X8aXUTr4KkIVjWQBC+K45w+OfnfdElcaUW1WQbp7AvaT
05bjWBH/BdJKBk0Kz8k=

`pragma protect key_keyowner="Aldec", key_keyname="ALDEC15_001", key_method="rsa"
`pragma protect encoding = (enctype="BASE64", line_length=76, bytes=256)
`pragma protect key_block
PE7RpcxN5xLwFXFfGeLCerEjPkCaTxfWD0b+oprDoEGqctY+F7Wc4+NqdSUAG4JbLWR/Pc8Mvten
+K5WbuDTljggJrkieJAK3rBOK8BdubtGJNC4uQ5v2trZYixfg4cWld5Z1MOB9aGfI0nF082l9Fc1
oNJFrkGcenyozvDKcCrtwvJaYRweulCV8/ynKznNpn+AvYhnoluR06IxzZXoj14b7IZt4g/2m455
clbUPyf1qLHbJGSK/Rvl0/W7cB5xxs3pM9/5p3UZ1MPFflZOAeCDwlOgzpXGAzfCkl9cSVqnIFFm
Q3cDkAfbRmP4jbDuwO2EbXgpiNcvc0br8gVcbA==

`pragma protect key_keyowner="Mentor Graphics Corporation", key_keyname="MGC-VELOCE-RSA", key_method="rsa"
`pragma protect encoding = (enctype="BASE64", line_length=76, bytes=128)
`pragma protect key_block
V73a/OJRlITsAnnyOJ7wtYI9yf/7gLrmQAWi1efFadTpN18wAOW5wcGb51JuXb/TOad9XvQaZxoY
I1ZZckK9R3kp0xHb3eRHqTEs38gIdB9DieJsPfcgrAgAh7N3AeXDhRXODyfLCVtmoF6cv0lmGI3F
6gKImN8nvzJnJHdYS0w=

`pragma protect key_keyowner="Mentor Graphics Corporation", key_keyname="MGC-VERIF-SIM-RSA-2", key_method="rsa"
`pragma protect encoding = (enctype="BASE64", line_length=76, bytes=256)
`pragma protect key_block
DCcPCQlu9uE6EBAoASplmWq6o/1vaTK24YYikkUnuM7wUc+K6wrTEbkFivj5OqxF/zGGynv+kjob
I28B+W/69av/irvgfSaOkl5CUwgmAnYrJQd5zO7pvvK7YBJ/f93xJ/FmpQTogAblevs7NdJLp3g2
OjMs/8iwMyXJYb2YgHoEFYKN8iqqLfoE1FTy3G1JWKcwGAYvCl6xaaPp+oYT4c7L99IYhk6R8LBP
5s2r5TkwtZsEUda1DuYu5UkWe5K0DyTVUxajXsz/s/xuKDYMlzV72Q6wQBSnBiknt/jnVyDo1tW+
PGq16LOUpjH7iA8esxtuBNSsdeu0hMvHXuCk0Q==

`pragma protect key_keyowner="Real Intent", key_keyname="RI-RSA-KEY-1", key_method="rsa"
`pragma protect encoding = (enctype="BASE64", line_length=76, bytes=256)
`pragma protect key_block
nwGINoKaVugQKTFyoLZc9O4TFbtk1/YjUn+a0zrC86S6J/PE/JSOh+4Yfyc8eQ/WKAw+4uNuqCfu
xMoblhyJX/PlyEcylGam6sg3YG3KUGmi+YfHMkiy5TIq7RfDEHBwm0OHajhuCJo+X/6WxDGrk+PS
JQwDW5IPxNMsoAYvcfcnzoAzaBMZ2IOHG/dpyyZ9tggqfcMqg7t2BLQujqkegYCW4gY5rCPGeljI
4AGn2WPvX+9JN5GQNIRdoRh8Onlhi39C7f6rkpR/zl9AY5kpcq9JW6q9bZNUXWBSnGm4Qz7GqfRv
VWcK4i0ng1C6xuiSkLkN+3y6/j1T30YlwsMfwA==

`pragma protect key_keyowner="Metrics Technologies Inc.", key_keyname="DSim", key_method="rsa"
`pragma protect encoding = (enctype="BASE64", line_length=76, bytes=256)
`pragma protect key_block
hEsll6mGIaBLE7AxJMVNtolpxRIRkhMsR1xHRjohvcUv4jFzKCjUO8d3wAnCq+Pz8f91PWDGP2f8
vGvDzErKDDVww33sDh+czaUMpdCCZkXt02jx1NfASHgkhqZt48L0UuNktllZas8HQy1w28ioRL/7
3KzyZN4uQ0f8w82zbdxC6U1l1meuVs6Ymk3Nsfmr36ARxpZj/9mbYwpjCUYyUzvUJRzuQfrAg0Ug
NoZDuxYRFYh7nfwlzgujXfMnemYGWwvjSq+iGvBWkCedSDGyNW+1BepcrFfzMd0eKQwmcj8h07J6
R97hYRxcHh/xlYrOs7brn6ldE9gjFrrjzoo8iw==

`pragma protect key_keyowner="Xilinx", key_keyname="xilinxt_2022_10", key_method="rsa"
`pragma protect encoding = (enctype="BASE64", line_length=76, bytes=256)
`pragma protect key_block
LCn+A4p440uf9LYQwYUsNjUDkTOYDGJVSfBa72VZuxFlEtdCBsGMjyJlD71i9wfT+zo8h+uKo5qg
vdv6mNq8TlFLiiopLnEQiAavSCyjdKaqzw8udtBKGsJVh0jvWBiBGYR3s//q93WXtDm9YvhHTdgy
QyzPYSyta3qQBDVoFvr9QDfszU/AgD7tMB34LAvkpr+FTkjoCCJrveOtK2B2WHLDkSZUKkPVQ4z2
NkNE0C0TKTL07EoIHcBpTszfP8fVP255K0UDLBoKbNkya3Q/UqjG6bR9pNXI4n94ocrJxFUkJyc7
WjwNxUjgC0HL2CND2aA9LS0fnSdpZ/JZhvib0Q==

`pragma protect key_keyowner="Atrenta", key_keyname="ATR-SG-RSA-1", key_method="rsa"
`pragma protect encoding = (enctype="BASE64", line_length=76, bytes=384)
`pragma protect key_block
eQmhkruRac3U1ERXu1Se13kEfZn5KP07/7J3pWhxX9QDO3A6aHkXHeiaH+qtRq5D2WFtbDCyVd5o
yPb7cQ9Sf2K6uFN+DTniB0oRADGePTdy2g3FHV68hvgIVlFrc9uf2rfs2yWR6pds+LDyYHhnSlEJ
hayggxgMxA41kth5hR2kGSybXpDjpQylauMvP+MetSb/27syf7QWVbXFhfLQE64opyObtme8TiWj
5MsRktemiPbC/x2RmB+ZklSRAvWu35tDO9u3XZ/Kk6WG2Pqj607cIfA/TsqoxZ6522ktimOnveJA
pSE6WUmDIQaOZ6pXjVIv4GEXZX73ZA3wLhRrmL7QxfXqCAeqFjouROHvM8Zv9kOfFGuDEo44JQSg
frNFO+XzslBTB/aplI0UWXL5ak9uQ1BUPtBO7xedGF/B7KKNsOoTml/48MYOaM6/3vjqSeiVUbvK
VVOQg7nBLowDBuFhcmn2RlKHia3ldFgRHkvFeuIJs5w1Ca1q0zuPaDzZ

`pragma protect key_keyowner="Cadence Design Systems.", key_keyname="CDS_RSA_KEY_VER_1", key_method="rsa"
`pragma protect encoding = (enctype="BASE64", line_length=76, bytes=256)
`pragma protect key_block
gmsDEPC6u2G9YmnQc+V5rl/+mw4IztZGOLCaixhGfvI1t16GeLgoWqRktnjvSlEQbNzx7qnrDu3r
je4CfK2ZF54ZUMDX1QZd4bL54eK/AB3GiU3wZKUDtd9ZB5j4Oq5zWBU/nuHisg8FDEXkIndNgak5
cWycB82LoueWPC88cLKbbFasGsKFV3+Cn/sQ88RAmKtwlL2bwHvI+udevI4dY8w7//0nMejfXbdJ
QRGDs0h2SxkGs298aaM464WfmgQ63xcJn9AB5LuagH+a7BdhySu3RNfRzgSw5k3AYXJd3Q8Mht1s
ztXL0X+/yquUTzhcMBTfyicvWkiYRw6CxzjnDA==

`pragma protect data_method = "AES128-CBC"
`pragma protect encoding = (enctype = "BASE64", line_length = 76, bytes = 2960)
`pragma protect data_block
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
`pragma protect end_protected
`ifndef GLBL
`define GLBL
`timescale  1 ps / 1 ps

module glbl ();

    parameter ROC_WIDTH = 100000;
    parameter TOC_WIDTH = 0;
    parameter GRES_WIDTH = 10000;
    parameter GRES_START = 10000;

//--------   STARTUP Globals --------------
    wire GSR;
    wire GTS;
    wire GWE;
    wire PRLD;
    wire GRESTORE;
    tri1 p_up_tmp;
    tri (weak1, strong0) PLL_LOCKG = p_up_tmp;

    wire PROGB_GLBL;
    wire CCLKO_GLBL;
    wire FCSBO_GLBL;
    wire [3:0] DO_GLBL;
    wire [3:0] DI_GLBL;
   
    reg GSR_int;
    reg GTS_int;
    reg PRLD_int;
    reg GRESTORE_int;

//--------   JTAG Globals --------------
    wire JTAG_TDO_GLBL;
    wire JTAG_TCK_GLBL;
    wire JTAG_TDI_GLBL;
    wire JTAG_TMS_GLBL;
    wire JTAG_TRST_GLBL;

    reg JTAG_CAPTURE_GLBL;
    reg JTAG_RESET_GLBL;
    reg JTAG_SHIFT_GLBL;
    reg JTAG_UPDATE_GLBL;
    reg JTAG_RUNTEST_GLBL;

    reg JTAG_SEL1_GLBL = 0;
    reg JTAG_SEL2_GLBL = 0 ;
    reg JTAG_SEL3_GLBL = 0;
    reg JTAG_SEL4_GLBL = 0;

    reg JTAG_USER_TDO1_GLBL = 1'bz;
    reg JTAG_USER_TDO2_GLBL = 1'bz;
    reg JTAG_USER_TDO3_GLBL = 1'bz;
    reg JTAG_USER_TDO4_GLBL = 1'bz;

    assign (strong1, weak0) GSR = GSR_int;
    assign (strong1, weak0) GTS = GTS_int;
    assign (weak1, weak0) PRLD = PRLD_int;
    assign (strong1, weak0) GRESTORE = GRESTORE_int;

    initial begin
	GSR_int = 1'b1;
	PRLD_int = 1'b1;
	#(ROC_WIDTH)
	GSR_int = 1'b0;
	PRLD_int = 1'b0;
    end

    initial begin
	GTS_int = 1'b1;
	#(TOC_WIDTH)
	GTS_int = 1'b0;
    end

    initial begin 
	GRESTORE_int = 1'b0;
	#(GRES_START);
	GRESTORE_int = 1'b1;
	#(GRES_WIDTH);
	GRESTORE_int = 1'b0;
    end

endmodule
`endif
