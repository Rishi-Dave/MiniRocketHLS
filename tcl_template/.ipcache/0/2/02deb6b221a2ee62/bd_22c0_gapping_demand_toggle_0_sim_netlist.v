// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2023 Advanced Micro Devices, Inc. All Rights Reserved.
// --------------------------------------------------------------------------------
// Tool Version: Vivado v.2023.2 (lin64) Build 4029153 Fri Oct 13 20:13:54 MDT 2023
// Date        : Thu Nov 20 11:38:50 2025
// Host        : wolverine running 64-bit Ubuntu 22.04.5 LTS
// Command     : write_verilog -force -mode funcsim -rename_top decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix -prefix
//               decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_ bd_22c0_gapping_demand_toggle_0_sim_netlist.v
// Design      : bd_22c0_gapping_demand_toggle_0
// Purpose     : This verilog netlist is a functional simulation representation of the design and should not be modified
//               or synthesized. This netlist cannot be used for SDF annotated simulation.
// Device      : xcu280-fsvh2892-2L-e
// --------------------------------------------------------------------------------
`timescale 1 ps / 1 ps

(* CHECK_LICENSE_TYPE = "bd_22c0_gapping_demand_toggle_0,c_counter_binary_v12_0_17,{}" *) (* downgradeipidentifiedwarnings = "yes" *) (* x_core_info = "c_counter_binary_v12_0_17,Vivado 2023.2" *) 
(* NotValidForBitStream *)
module decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix
   (CLK,
    CE,
    Q);
  (* x_interface_info = "xilinx.com:signal:clock:1.0 clk_intf CLK" *) (* x_interface_parameter = "XIL_INTERFACENAME clk_intf, ASSOCIATED_BUSIF q_intf:thresh0_intf:l_intf:load_intf:up_intf:sinit_intf:sset_intf, ASSOCIATED_RESET SCLR, ASSOCIATED_CLKEN CE, FREQ_HZ 50000000, FREQ_TOLERANCE_HZ 0, PHASE 0, CLK_DOMAIN cd_ctrl_00, INSERT_VIP 0" *) input CLK;
  (* x_interface_info = "xilinx.com:signal:clockenable:1.0 ce_intf CE" *) (* x_interface_parameter = "XIL_INTERFACENAME ce_intf, POLARITY ACTIVE_HIGH" *) input CE;
  (* x_interface_info = "xilinx.com:signal:data:1.0 q_intf DATA" *) (* x_interface_parameter = "XIL_INTERFACENAME q_intf, LAYERED_METADATA xilinx.com:interface:datatypes:1.0 {DATA {datatype {name {attribs {resolve_type immediate dependency {} format string minimum {} maximum {}} value data} bitwidth {attribs {resolve_type generated dependency bitwidth format long minimum {} maximum {}} value 1} bitoffset {attribs {resolve_type immediate dependency {} format long minimum {} maximum {}} value 0} integer {signed {attribs {resolve_type immediate dependency {} format bool minimum {} maximum {}} value false}}}} DATA_WIDTH 1}" *) output [0:0]Q;

  wire CE;
  wire CLK;
  wire [0:0]Q;
  wire NLW_U0_THRESH0_UNCONNECTED;

  (* C_AINIT_VAL = "0" *) 
  (* C_CE_OVERRIDES_SYNC = "0" *) 
  (* C_COUNT_BY = "1" *) 
  (* C_COUNT_MODE = "0" *) 
  (* C_COUNT_TO = "1" *) 
  (* C_FB_LATENCY = "0" *) 
  (* C_HAS_CE = "1" *) 
  (* C_HAS_LOAD = "0" *) 
  (* C_HAS_SCLR = "0" *) 
  (* C_HAS_SINIT = "0" *) 
  (* C_HAS_SSET = "0" *) 
  (* C_HAS_THRESH0 = "0" *) 
  (* C_IMPLEMENTATION = "0" *) 
  (* C_LATENCY = "1" *) 
  (* C_LOAD_LOW = "0" *) 
  (* C_RESTRICT_COUNT = "0" *) 
  (* C_SCLR_OVERRIDES_SSET = "1" *) 
  (* C_SINIT_VAL = "0" *) 
  (* C_THRESH0_VALUE = "1" *) 
  (* C_VERBOSITY = "0" *) 
  (* C_WIDTH = "1" *) 
  (* C_XDEVICEFAMILY = "virtexuplusHBM" *) 
  (* downgradeipidentifiedwarnings = "yes" *) 
  (* is_du_within_envelope = "true" *) 
  decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_c_counter_binary_v12_0_17 U0
       (.CE(CE),
        .CLK(CLK),
        .L(1'b0),
        .LOAD(1'b0),
        .Q(Q),
        .SCLR(1'b0),
        .SINIT(1'b0),
        .SSET(1'b0),
        .THRESH0(NLW_U0_THRESH0_UNCONNECTED),
        .UP(1'b1));
endmodule
`pragma protect begin_protected
`pragma protect version = 1
`pragma protect encrypt_agent = "XILINX"
`pragma protect encrypt_agent_info = "Xilinx Encryption Tool 2023.2"
`pragma protect key_keyowner="Synopsys", key_keyname="SNPS-VCS-RSA-2", key_method="rsa"
`pragma protect encoding = (enctype="BASE64", line_length=76, bytes=128)
`pragma protect key_block
eFbqyWxvTxgrA/YtdaoI20/0Oxv6heWR3Rkp9/xOWnvLDdGDhtGJBQqdO4v1RO/kikveHE3JyVBx
OMXM/QBYbcn/QmEMFud4drsy8IbaUwVstP+Mzovw04CY0e6ucHcNG8bkdAhiixaw1DGilwl8tfXo
1/LD/FGivkVY+qD5JIE=

`pragma protect key_keyowner="Aldec", key_keyname="ALDEC15_001", key_method="rsa"
`pragma protect encoding = (enctype="BASE64", line_length=76, bytes=256)
`pragma protect key_block
qZVOEd9Suj8PFYlAHZ5eNfv9g67bFY/Iau3fGJHFAIz/4EbdSAUDaGh/Aj5F/sayLnlRNhD6w+39
N7ODCROvgCW/DEQMBCPz7kcEchwyjzrqkhJexEv0Dz7kFQn1ftmdbnZ6SxsSg0bAUSqDETfwIrDN
VELNGURpq3DjO751fQLkz152JThZlONrPm6SqH+2yq0k/imlDMyhznvq+Up4EXiczfO25/APInqH
9ImfZSrqCiz3p7BNa9t1DtJtjx4nO4g/3qItwAhtZOzSyNgUZUJkS0OgYwLaNbOAMte1lEZ3aCj/
PtYFcVrRv6BV9zObKm5JRWmYYw/qLDjrN9AsCA==

`pragma protect key_keyowner="Mentor Graphics Corporation", key_keyname="MGC-VELOCE-RSA", key_method="rsa"
`pragma protect encoding = (enctype="BASE64", line_length=76, bytes=128)
`pragma protect key_block
PTqB7iZsvJeVQbxSYRkkEOB7dur2/Y+zWd7rSI4QgTOZZuY7cx4mymLcNTtY69vWs3+Ir6xtLuRI
kV9wRh0KJKuphJal6eQJChHGu6rp+AHyp8AyhIwGgID1vxyyu7xhU5nl4qM40fYe+ov2uBp5DVP0
GoOHS6Gilji9DRkCQuI=

`pragma protect key_keyowner="Mentor Graphics Corporation", key_keyname="MGC-VERIF-SIM-RSA-2", key_method="rsa"
`pragma protect encoding = (enctype="BASE64", line_length=76, bytes=256)
`pragma protect key_block
nl92noyushAx6EnMgw3oSlb0lEtv202gnVMSNN83+NLaV5DJ/HimKQF470dvcnALDIl0xa3e3Dx2
/s2hBMgu9+fSioH4xbMFQTaBWMjBfDKLVgBkEfT5zBbn1LpjuMEnd/TVHxe/dqXJ8Ev1EIyVB5r3
7KAUvfDL8CretmawtvJtixs8bH8vAxLO4BUzVNbXDL44NeL/PffK31PA74odtZbSUGIq+Gf0nEXP
yEajhFawSXpK6M+iRpsuDwKHS/YxQldY5i8FGvVQrcrDBe3XAh+jjvxUqPhZBRChKpDSo0q7V9L4
JAZoQiGn28UrFoWwrxxP1gsv7sPdry3YTRu8DA==

`pragma protect key_keyowner="Real Intent", key_keyname="RI-RSA-KEY-1", key_method="rsa"
`pragma protect encoding = (enctype="BASE64", line_length=76, bytes=256)
`pragma protect key_block
KbhPmoTx5e7VcsW2JDC/J/bcPlnz66gRyRCtg0E0qHv9wqViijoH+owrxv0IWMQoDBsXBaeQm2xz
nHRZDc5s+B+JlzwwZQGB8pQM3sXMmxGcH+jeVqy6X8gKOEQFgnIK2FJlAjHpfO0xmJkl3wxWImNr
ADPNoWEMdruR5ksSgKexng6J3lkv4vPYoEvCF+Jq91pp71EIJgPtwlY833cs1Exi28xe2Qo/nzU7
oEFG5gySNEidQa25q1QrCDnSmj7j5wDJg5xzjXYmwWk9873dPWyEXdpFMqjxovIcyph/uXidS1PQ
XxFetrAMjtseoYWmz4Lm4f+rpe89PGRhWXsiZw==

`pragma protect key_keyowner="Xilinx", key_keyname="xilinxt_2022_10", key_method="rsa"
`pragma protect encoding = (enctype="BASE64", line_length=76, bytes=256)
`pragma protect key_block
u7x25sjZWe79yzU52TdAK1EHhzoxhg0OOXYOwTp455Bu78gqkmKRv90VNHoa9foKyQc+Ui9ovV+f
Mu2Crcme1IbP51J6eQdKys/57qJrcFCxGtJs1Tw7KJ7NffFwkytoqR7pgvmtIH6+qncA8b3aZTLq
uwD9bGF9UFZVZ2XBc83+LRU+GZnNMHOa4eegWtueYHh1zUhGju1xbiGWuhliZ58pcNp6gCDiDv+p
GdiwFDT5RDj1bjrkOecRL2fvOdGLrhdqiTh7mvJeDStjjXiovaCdny21gVHf+dzrpyPE2xGgBinA
czj0D2pyt8plttBhpmMBtLm7Yegb1rHiA1UC4Q==

`pragma protect key_keyowner="Metrics Technologies Inc.", key_keyname="DSim", key_method="rsa"
`pragma protect encoding = (enctype="BASE64", line_length=76, bytes=256)
`pragma protect key_block
ax8Nn3y8qIY97tYqs/w6/65Cp2CG+WRyi20ND2JmdKrfZukanonYYzlNsDdtoOeMZdG6gzVR279V
Et2Qz8YBLQmhBZMJ13mNxEOwpSUbc5lUbLJ9CQ+4u6jvStTDzX+odxkCCqHG8GJhSSFPGX+Z3VZc
TdU/OWddzxwk6JO1tiPn+qt2Q8nMj3Ulh7gqAxPMp0gosh6z+Kx5ZXSuVE/EPNyUDXjRSXjnWPwN
NnM94gbzG23dPqFIOG6f2m5ugBmUUghvI75DFpM7vJkXsEWAfZeOV351MLISR26yMaWxONCdGWTQ
DW1hvUkse+kVt5BxF8ft6CnT7VchA/flFKvfPg==

`pragma protect key_keyowner="Atrenta", key_keyname="ATR-SG-RSA-1", key_method="rsa"
`pragma protect encoding = (enctype="BASE64", line_length=76, bytes=384)
`pragma protect key_block
e/Ry1l/vqJFJqrtAPFKKjxxp0MWhiORDa7WOuKdNY7LaztvYcFdzfNmZDmSuTrj2jPF7WDL6uDC8
FMboEovqCHZ9s014f2WS+jYxvraqlkgyGQ1Q7gAZ6yaBsdSi2RgWdbpy++ECpfVjq8/sYcJl+CZl
ZIsOc7C4BTFh3wysjt9r3cLq/k/dOO8xw2ZAarjqRzjr0h9T3TEzowI/jWVIqkEICEpCsA6k6h2U
oRuVQxQmdtSXmU/Zm/g8I4dj6axYw/zYpYJe2v9s4sy3NCBf/p+z3JFoReqcGYtminQ5ba3zzR5v
mVNiXw+YtQVCe0IsGLqjaEBWcXrs5SNFVYKce0xzBupVSQ0hP5cTBMtcToem7n2nM+9LdhZqlPBl
KimcvX+KrlUbox8H49P1OzL0A8+Eterrfdy0jE2DF+YuIuSAnudKM11WtjqK5yq8zJk5JOHQUQZl
qO7dmmZT95FjGPG/jqS+uYlHdZCNNyQ2l2SbVZw7NPkIJGq5bAKeGsSg

`pragma protect key_keyowner="Cadence Design Systems.", key_keyname="CDS_RSA_KEY_VER_1", key_method="rsa"
`pragma protect encoding = (enctype="BASE64", line_length=76, bytes=256)
`pragma protect key_block
OqTG2QUdaSjYwH6PeUiMov3bCDPXiUvE4U1Z/Vd7xbPVo7tNNwQeTcXKi/ZUR5G1tkEs1OoxIqaH
ytImL/6Ro6liEE5oT3otxdQb1Yz3ukWdnhQpgw3O2DRb5K1R4L7p5QCVIgq+/7f4oNn8VSJ1hT7o
jVMeT5IvgkJXZsqX/2c86iOEUQ/Xha6SVw6W50dS1u8Q/FzR54WBSo0MFMxZWh5Pyf5qnBJKna0d
nVdDHDBFDajINOk8j7Oonu6ynHrhNkyo1IEnNv/ZQPbHo2aynI/MIaO/5etmCQO2m+53okz5H7pu
RWYkE0zXoE1v6jl/QhZyc23SZppXzWGhd1zsRw==

`pragma protect data_method = "AES128-CBC"
`pragma protect encoding = (enctype = "BASE64", line_length = 76, bytes = 2336)
`pragma protect data_block
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
`pragma protect end_protected
`pragma protect begin_protected
`pragma protect version = 1
`pragma protect encrypt_agent = "XILINX"
`pragma protect encrypt_agent_info = "Xilinx Encryption Tool 2023.2"
`pragma protect key_keyowner="Synopsys", key_keyname="SNPS-VCS-RSA-2", key_method="rsa"
`pragma protect encoding = (enctype="BASE64", line_length=76, bytes=128)
`pragma protect key_block
eFbqyWxvTxgrA/YtdaoI20/0Oxv6heWR3Rkp9/xOWnvLDdGDhtGJBQqdO4v1RO/kikveHE3JyVBx
OMXM/QBYbcn/QmEMFud4drsy8IbaUwVstP+Mzovw04CY0e6ucHcNG8bkdAhiixaw1DGilwl8tfXo
1/LD/FGivkVY+qD5JIE=

`pragma protect key_keyowner="Aldec", key_keyname="ALDEC15_001", key_method="rsa"
`pragma protect encoding = (enctype="BASE64", line_length=76, bytes=256)
`pragma protect key_block
qZVOEd9Suj8PFYlAHZ5eNfv9g67bFY/Iau3fGJHFAIz/4EbdSAUDaGh/Aj5F/sayLnlRNhD6w+39
N7ODCROvgCW/DEQMBCPz7kcEchwyjzrqkhJexEv0Dz7kFQn1ftmdbnZ6SxsSg0bAUSqDETfwIrDN
VELNGURpq3DjO751fQLkz152JThZlONrPm6SqH+2yq0k/imlDMyhznvq+Up4EXiczfO25/APInqH
9ImfZSrqCiz3p7BNa9t1DtJtjx4nO4g/3qItwAhtZOzSyNgUZUJkS0OgYwLaNbOAMte1lEZ3aCj/
PtYFcVrRv6BV9zObKm5JRWmYYw/qLDjrN9AsCA==

`pragma protect key_keyowner="Mentor Graphics Corporation", key_keyname="MGC-VELOCE-RSA", key_method="rsa"
`pragma protect encoding = (enctype="BASE64", line_length=76, bytes=128)
`pragma protect key_block
PTqB7iZsvJeVQbxSYRkkEOB7dur2/Y+zWd7rSI4QgTOZZuY7cx4mymLcNTtY69vWs3+Ir6xtLuRI
kV9wRh0KJKuphJal6eQJChHGu6rp+AHyp8AyhIwGgID1vxyyu7xhU5nl4qM40fYe+ov2uBp5DVP0
GoOHS6Gilji9DRkCQuI=

`pragma protect key_keyowner="Mentor Graphics Corporation", key_keyname="MGC-VERIF-SIM-RSA-2", key_method="rsa"
`pragma protect encoding = (enctype="BASE64", line_length=76, bytes=256)
`pragma protect key_block
nl92noyushAx6EnMgw3oSlb0lEtv202gnVMSNN83+NLaV5DJ/HimKQF470dvcnALDIl0xa3e3Dx2
/s2hBMgu9+fSioH4xbMFQTaBWMjBfDKLVgBkEfT5zBbn1LpjuMEnd/TVHxe/dqXJ8Ev1EIyVB5r3
7KAUvfDL8CretmawtvJtixs8bH8vAxLO4BUzVNbXDL44NeL/PffK31PA74odtZbSUGIq+Gf0nEXP
yEajhFawSXpK6M+iRpsuDwKHS/YxQldY5i8FGvVQrcrDBe3XAh+jjvxUqPhZBRChKpDSo0q7V9L4
JAZoQiGn28UrFoWwrxxP1gsv7sPdry3YTRu8DA==

`pragma protect key_keyowner="Real Intent", key_keyname="RI-RSA-KEY-1", key_method="rsa"
`pragma protect encoding = (enctype="BASE64", line_length=76, bytes=256)
`pragma protect key_block
KbhPmoTx5e7VcsW2JDC/J/bcPlnz66gRyRCtg0E0qHv9wqViijoH+owrxv0IWMQoDBsXBaeQm2xz
nHRZDc5s+B+JlzwwZQGB8pQM3sXMmxGcH+jeVqy6X8gKOEQFgnIK2FJlAjHpfO0xmJkl3wxWImNr
ADPNoWEMdruR5ksSgKexng6J3lkv4vPYoEvCF+Jq91pp71EIJgPtwlY833cs1Exi28xe2Qo/nzU7
oEFG5gySNEidQa25q1QrCDnSmj7j5wDJg5xzjXYmwWk9873dPWyEXdpFMqjxovIcyph/uXidS1PQ
XxFetrAMjtseoYWmz4Lm4f+rpe89PGRhWXsiZw==

`pragma protect key_keyowner="Xilinx", key_keyname="xilinxt_2022_10", key_method="rsa"
`pragma protect encoding = (enctype="BASE64", line_length=76, bytes=256)
`pragma protect key_block
u7x25sjZWe79yzU52TdAK1EHhzoxhg0OOXYOwTp455Bu78gqkmKRv90VNHoa9foKyQc+Ui9ovV+f
Mu2Crcme1IbP51J6eQdKys/57qJrcFCxGtJs1Tw7KJ7NffFwkytoqR7pgvmtIH6+qncA8b3aZTLq
uwD9bGF9UFZVZ2XBc83+LRU+GZnNMHOa4eegWtueYHh1zUhGju1xbiGWuhliZ58pcNp6gCDiDv+p
GdiwFDT5RDj1bjrkOecRL2fvOdGLrhdqiTh7mvJeDStjjXiovaCdny21gVHf+dzrpyPE2xGgBinA
czj0D2pyt8plttBhpmMBtLm7Yegb1rHiA1UC4Q==

`pragma protect key_keyowner="Metrics Technologies Inc.", key_keyname="DSim", key_method="rsa"
`pragma protect encoding = (enctype="BASE64", line_length=76, bytes=256)
`pragma protect key_block
ax8Nn3y8qIY97tYqs/w6/65Cp2CG+WRyi20ND2JmdKrfZukanonYYzlNsDdtoOeMZdG6gzVR279V
Et2Qz8YBLQmhBZMJ13mNxEOwpSUbc5lUbLJ9CQ+4u6jvStTDzX+odxkCCqHG8GJhSSFPGX+Z3VZc
TdU/OWddzxwk6JO1tiPn+qt2Q8nMj3Ulh7gqAxPMp0gosh6z+Kx5ZXSuVE/EPNyUDXjRSXjnWPwN
NnM94gbzG23dPqFIOG6f2m5ugBmUUghvI75DFpM7vJkXsEWAfZeOV351MLISR26yMaWxONCdGWTQ
DW1hvUkse+kVt5BxF8ft6CnT7VchA/flFKvfPg==

`pragma protect key_keyowner="Atrenta", key_keyname="ATR-SG-RSA-1", key_method="rsa"
`pragma protect encoding = (enctype="BASE64", line_length=76, bytes=384)
`pragma protect key_block
e/Ry1l/vqJFJqrtAPFKKjxxp0MWhiORDa7WOuKdNY7LaztvYcFdzfNmZDmSuTrj2jPF7WDL6uDC8
FMboEovqCHZ9s014f2WS+jYxvraqlkgyGQ1Q7gAZ6yaBsdSi2RgWdbpy++ECpfVjq8/sYcJl+CZl
ZIsOc7C4BTFh3wysjt9r3cLq/k/dOO8xw2ZAarjqRzjr0h9T3TEzowI/jWVIqkEICEpCsA6k6h2U
oRuVQxQmdtSXmU/Zm/g8I4dj6axYw/zYpYJe2v9s4sy3NCBf/p+z3JFoReqcGYtminQ5ba3zzR5v
mVNiXw+YtQVCe0IsGLqjaEBWcXrs5SNFVYKce0xzBupVSQ0hP5cTBMtcToem7n2nM+9LdhZqlPBl
KimcvX+KrlUbox8H49P1OzL0A8+Eterrfdy0jE2DF+YuIuSAnudKM11WtjqK5yq8zJk5JOHQUQZl
qO7dmmZT95FjGPG/jqS+uYlHdZCNNyQ2l2SbVZw7NPkIJGq5bAKeGsSg

`pragma protect key_keyowner="Cadence Design Systems.", key_keyname="CDS_RSA_KEY_VER_1", key_method="rsa"
`pragma protect encoding = (enctype="BASE64", line_length=76, bytes=256)
`pragma protect key_block
OqTG2QUdaSjYwH6PeUiMov3bCDPXiUvE4U1Z/Vd7xbPVo7tNNwQeTcXKi/ZUR5G1tkEs1OoxIqaH
ytImL/6Ro6liEE5oT3otxdQb1Yz3ukWdnhQpgw3O2DRb5K1R4L7p5QCVIgq+/7f4oNn8VSJ1hT7o
jVMeT5IvgkJXZsqX/2c86iOEUQ/Xha6SVw6W50dS1u8Q/FzR54WBSo0MFMxZWh5Pyf5qnBJKna0d
nVdDHDBFDajINOk8j7Oonu6ynHrhNkyo1IEnNv/ZQPbHo2aynI/MIaO/5etmCQO2m+53okz5H7pu
RWYkE0zXoE1v6jl/QhZyc23SZppXzWGhd1zsRw==

`pragma protect key_keyowner="Mentor Graphics Corporation", key_keyname="MGC-PREC-RSA", key_method="rsa"
`pragma protect encoding = (enctype="BASE64", line_length=76, bytes=256)
`pragma protect key_block
IDfFkZlierVoL9jDy8mpGy15K2qXTKa1Go4HO6yLh2pXnNVIes+wz8RqYR/p3yqOXmJocrl4Jcpd
nd8A6R0Zv7+BJJCIV3VX8zn9EuGPPKpLDgTSA1dIRk9XbzJAi8RWIwbWGD413PUvNGXVAiRJZkDu
fs1sfJyOgQAewelSHMe5uYcJTWMl7OEcTbf+VbJuL0elCZku98NkwccdGzNJ1IEAX93pi+pDWs30
nbh97AjjX/rZpQxpaUT+/z2k73zyWLYaeogPNU1YYrTIxqN8tqwyB3T2+WhPXYiNC7n6PTB9vdP9
altyN3YKPbD/UzhEJM35btwO+UhsAVjp/WJudQ==

`pragma protect key_keyowner="Synplicity", key_keyname="SYNP15_1", key_method="rsa"
`pragma protect encoding = (enctype="BASE64", line_length=76, bytes=256)
`pragma protect key_block
eutUwXuFT0siwDPCRfEwSjWV7I0ZAOjVh71m/AK99Kc63NhTYjEg+ivesQenKK9djavoEzEP1X74
vLb9K6zt8aNZNqk7Ymlw6IjEKUg6Xc9VxyBwrpmi2q0Gg/ANravPxHPBkUjeZHibuRA39r1Uce2q
OYu4gJBzSrzHnMhcFqA2Omy7TNdfPoFns9FGqLLZFPU+n7ENG9HidJWuRhJmJetfTSMtC1G/JzJH
zLo50wBSiwNTKfcYdRTrl/KSfC+Vs5LdoW/57qpnJOwA9ObJy81Y3XEgVhSf033tOfcUCfq5TFkT
MAcHYCMqA1Fu/rkc1r0SN7U+Ilt1iNDlp1ozBg==

`pragma protect data_method = "AES128-CBC"
`pragma protect encoding = (enctype = "BASE64", line_length = 76, bytes = 2832)
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
