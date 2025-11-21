-- Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
-- Copyright 2022-2023 Advanced Micro Devices, Inc. All Rights Reserved.
-- --------------------------------------------------------------------------------
-- Tool Version: Vivado v.2023.2 (lin64) Build 4029153 Fri Oct 13 20:13:54 MDT 2023
-- Date        : Thu Nov 20 11:34:39 2025
-- Host        : wolverine running 64-bit Ubuntu 22.04.5 LTS
-- Command     : write_vhdl -force -mode funcsim -rename_top decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix -prefix
--               decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_ bd_22c0_aclk_kernel_01_cont_adapt_0_sim_netlist.vhdl
-- Design      : bd_22c0_aclk_kernel_01_cont_adapt_0
-- Purpose     : This VHDL netlist is a functional simulation representation of the design and should not be modified or
--               synthesized. This netlist cannot be used for SDF annotated simulation.
-- Device      : xcu280-fsvh2892-2L-e
-- --------------------------------------------------------------------------------
`protect begin_protected
`protect version = 1
`protect encrypt_agent = "XILINX"
`protect encrypt_agent_info = "Xilinx Encryption Tool 2019.1"
`protect key_keyowner="Cadence Design Systems.", key_keyname="cds_rsa_key", key_method="rsa"
`protect encoding = (enctype="BASE64", line_length=76, bytes=64)
`protect key_block
AyywBWU3I0zQc5RgrR3iChpKXBvMdAMA27xhh+MGe6Mh5tRmo/57cgaVWWooa4q9CmpFKczA4/mG
svfrQJgEag==

`protect key_keyowner="Synopsys", key_keyname="SNPS-VCS-RSA-2", key_method="rsa"
`protect encoding = (enctype="BASE64", line_length=76, bytes=128)
`protect key_block
FXgvu/Ju/3y2Q8ovtwzDK3JsyWtVUt3PS668MtsIey+72PF+sW5ddRCoSHKKZb3YYaeTtGJYWm66
NtQPyIZdVZmJdMRBGkOTpSNSqxRhcbJpqtnDUs/ZmUY0cZYoGc6VvoxUM/c199/gpwt6OTanzcMJ
JiaciI7dnneQqC/bSTk=

`protect key_keyowner="Aldec", key_keyname="ALDEC15_001", key_method="rsa"
`protect encoding = (enctype="BASE64", line_length=76, bytes=256)
`protect key_block
GYQmBI722qMjJWUzYrBkDSBJhMhoaNc/1lNtZDSH1la2aFsVzEip7QfQ4GLAWt4g5J1o6aLpwKWk
OI4rbDRp+A4f/zX753kPU5uT4121eCks8u8EIJmTZuanOBB5qyrkziVfnRNg1HpF2VJ9+YgJFwWf
qTIOkF6KiGuG3Un8OwqYx8nPvhjcPJthdhZn6L9Ww7S2lrAPeAV+MeAaCIumHosVsbwy6lvw9OUf
kEJz34ZYPagX4pdtjixCAPMZ2xN3mwr97aGqQSb3zo1S2uCs8A0m1IVst3rQIrdbTjD10gTrXqJW
u98o3qQb2Xfn8rhAsMAVkCDsLcu0RSAgsmtoKg==

`protect key_keyowner="ATRENTA", key_keyname="ATR-SG-2015-RSA-3", key_method="rsa"
`protect encoding = (enctype="BASE64", line_length=76, bytes=256)
`protect key_block
awZfD5sS6wXogZPJl5LHPZMttRTvu8nUMSmLLkxFMx3Pc2BW5ubYrL1e1lHRYYyHwXrqOZXlJFev
nA51AmrJqjQpqEqAv0jCBzGTyb3DZiPSJbK+dsjQSotZDmSH2lTQ+tSvvZ+lgmq7K/A/nfG5BiRe
7POKFw6/G+vqhDlGCRzKfdGbAEsBPq+2wdcombJa5Fd159RS8aJM7uhZgHCt7osCOALOhv5tlnqJ
krA+hOF0yevtXcsi5NKx4fIjm/ykCrrsrh1k36z0GjQwQxybZBwGhp1E9rsxFfKI1brBxsakKV8f
WZTrLzVQhPkRNWx3/46Zz/zDZfxcSWBVV2ZMug==

`protect key_keyowner="Xilinx", key_keyname="xilinxt_2019_02", key_method="rsa"
`protect encoding = (enctype="BASE64", line_length=76, bytes=256)
`protect key_block
VXQasTBG9shgYH2gH0qz0Yc+uw3zU3aSd/WGie0aEDWNUmEJFYhlvljQQBuevhEBTvcDHWmsZ1yo
SnMX+FBcYz1201CnVxpSemXo1p65r6Mhr5XptqSQiamZ7jSXaGoEYcVn0Wj0f/UmzoL2I1w5QzWS
dpGU1uXvPvv7IO/gdvuXIzvPwQtc9S0v+ktmQetDwXmh3y40DI+jKCmOZQFN8uoGk+2mHPzzoMMv
JyV0P40BdVP3qp2lSJyzhOuzWtkXpW82ge3jFqv7WBuODu2IBH/TMhdGn0IKNJq9FWEzgPe9gLiE
0kpVCipgqMCQUBWkMkOMZXudWiFCdNYX1I2trA==

`protect key_keyowner="Mentor Graphics Corporation", key_keyname="MGC-VELOCE-RSA", key_method="rsa"
`protect encoding = (enctype="BASE64", line_length=76, bytes=128)
`protect key_block
dTdpoe4SogJMN3GFRUyEh1d9OlwzQuE/wDpqK0CQnna7vbUCR4o5fxs7KD7rBoclsLHNb2wXo+2l
VwSBLpJBIG7gA2VyUOyqdyDUTZ1cGMBAr5wFYRAd5opbpFWrS0TG4FE0WFxvCU7NS3bblfiwX1Z8
5ibWdlaIZbWjyxjuZgo=

`protect key_keyowner="Mentor Graphics Corporation", key_keyname="MGC-VERIF-SIM-RSA-2", key_method="rsa"
`protect encoding = (enctype="BASE64", line_length=76, bytes=256)
`protect key_block
fKynujfRgwcb62ZQvbXCI63ZrNRBekYnDwE0bn1UnQAQHczPaMhOm45bfdZRZiyHMIWSwQjOuSFb
ocDGcz2FhWVtLjBMk2OlSj2qyscl2bbXnzoeHspbhDe0dYxh8fn/9n6onEExTRKXgYuYI37vGpvo
MIlut7iL5VIC6zoxLdtrZZaWJRfguQplAmuAUsv3p39VxHELD4v3j+6vv/i4N6q+gVR4KOMyddQC
ggiotsFObvd3KY6pfb4i7wSgj8TplT3uDv65Xkvk2NPKTygHU9mlZvDRtKxDvsCL7/1UOIjgVvua
6bZNJojrE7+sFVbgF0cvfm7vL/Cwgzax/GmIkA==

`protect key_keyowner="Real Intent", key_keyname="RI-RSA-KEY-1", key_method="rsa"
`protect encoding = (enctype="BASE64", line_length=76, bytes=256)
`protect key_block
EgTCocaIITNNFz8bK/lPvfNuRE7Y4kxq+HC0/PBrEd8vD8od3ahyOKr0g4zaX7iRXbFffAtXUUlO
rtfukr4iiQfztiWn0+EygKAzO4+oVsYHZNUlOOeq3dBxWVWELwV7PdN6TX3QR5gmddGLhw4puD8W
ah2bsZPnh0GElbFC9w4teR4HMDRw6ACP6Z4MlM/QrzXA33UDI/OtlgMIxQqlSTqdKmR2Y1UiOhCJ
v6yXAN7UO+HgFQ7BuHYPMjiUaL1FL3t1Qgmkigtey2pQ/X5TriXL69Z0oLik1xjibdf5pxjv1RqL
O0fjv08eGYZ2aRymOkJo3llorgODGMsxG6erKg==

`protect data_method = "AES128-CBC"
`protect encoding = (enctype = "BASE64", line_length = 76, bytes = 864)
`protect data_block
WJ2SNBCrn6IWP70N2PHuA/8VIjUeZ99sThD9XbQ3kO1I0nQcZuwTh6ivhjZd69824Z88HlYw+kks
vnV7giQpIaM92Mo4gILlIzm5riS7HKSXVbcc6VkoACfgicHpknFqH4SnM6Nfn+bPiudzzbwWKnSy
SGYb80W0A10uTWcuSd2SWI7+XHZHv1Si6ZI7cPp1q1OM4H9y+cv/iTDT6+Obsxa8OjCfig36amO5
K6eu0waPw5Aw5ZzPTsYlBzacHw0K/2YfMrS058DAmgOk74RMTVyA8FJyTUr5GHh5nSfng89Sq0dw
cDPtlBAlDKIUEi56jAOZV093Bb/y11K0qvx/tLW5WeTswlCFNa7sUfolPWoZl5aN/+uT08/CEq+0
uMKzGBirc07bUEtHQRFMp4azP3HUL9sz7FTfGNHGkXp61ZvlhgkV99T5hcD13ajQq/nwm14iUq95
U64mynGgy/y2hdvw6PQU2uH0K35ixpwXYVerhN/Uqt3L34ucS7RLMtWa99yAKxuLrYPK+8QlLF2s
6TMSRJ61zfisHM2zZuSTNglMrp0wQrN5tuNl7QBxy4dMbjoFgHMSW73ZzHfYin1rStBs3/Es7P99
PCLzqhqa0oBXECcy3Yz1uckK9tKN6l8ReVeh+VbjQXDkpsG63yzsyYQ7t2F+G3WQrY3WEONVIGUK
G93i+V8mvUJU5xK7R0KTCNMJzeZkc/Ej0X8vhKiRADW2YJu4BLlOEFHVw8H6Hl30C6fmiW4f9cWy
WMJ4i5a27sTT4VzvDU3ha4eXcIvwgo8CHfpvKoG2WB3RL6jOglyIvRtYXZmGuFitLtKEPw2aN80v
ziQ8OKr4/3pM+5biLawcahtib9n9tROapnlDz/uSZjRbs59xgf9JrC0Ht7Oaqobkt+7Oo3josJZx
Q+cvZQeEtP4VZgTshIyFDKZojzPIJshtCiV6kDzII4hzwCVreU2Do/VCznG90Hd2iJLYYW8ISuJk
azt4ADDW3ctyyFsDMqLO9+SDSlpo+Z0b7pYwJM4MVCrjqObjQladgWYTaCaD4/AjGIHpqVegvs2l
+mv1QiHep7ugsiht2ENLHHno2xjScFJi0yrR7pb291k+XYvijLX62zhwnwsluuofoj+dIUz4ElLl
a8/hIs6EnIRe
`protect end_protected
library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
library UNISIM;
use UNISIM.VCOMPONENTS.ALL;
entity decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix is
  port (
    clk_in : in STD_LOGIC;
    clk_out : out STD_LOGIC
  );
  attribute NotValidForBitStream : boolean;
  attribute NotValidForBitStream of decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix : entity is true;
  attribute CHECK_LICENSE_TYPE : string;
  attribute CHECK_LICENSE_TYPE of decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix : entity is "bd_22c0_aclk_kernel_01_cont_adapt_0,clk_metadata_adapter_v1_0_0,{}";
  attribute DowngradeIPIdentifiedWarnings : string;
  attribute DowngradeIPIdentifiedWarnings of decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix : entity is "yes";
  attribute X_CORE_INFO : string;
  attribute X_CORE_INFO of decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix : entity is "clk_metadata_adapter_v1_0_0,Vivado 2023.2";
end decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix;

architecture STRUCTURE of decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix is
  attribute KEEP_HIERARCHY : string;
  attribute KEEP_HIERARCHY of inst : label is "soft";
  attribute is_du_within_envelope : string;
  attribute is_du_within_envelope of inst : label is "true";
  attribute X_INTERFACE_INFO : string;
  attribute X_INTERFACE_INFO of clk_in : signal is "xilinx.com:signal:clock:1.0 CLOCK_IN CLK";
  attribute X_INTERFACE_PARAMETER : string;
  attribute X_INTERFACE_PARAMETER of clk_in : signal is "XIL_INTERFACENAME CLOCK_IN, FREQ_HZ 500000000, FREQ_TOLERANCE_HZ 0, PHASE 0.0, CLK_DOMAIN bd_22c0_clkwiz_aclk_kernel_01_0_clk_out1_buf, INSERT_VIP 0";
  attribute X_INTERFACE_INFO of clk_out : signal is "xilinx.com:signal:clock:1.0 CLOCK_OUT CLK";
  attribute X_INTERFACE_PARAMETER of clk_out : signal is "XIL_INTERFACENAME CLOCK_OUT, FREQ_HZ 500000000, FREQ_TOLERANCE_HZ 0, PHASE 0.0, CLK_DOMAIN cd_aclk_kernel_01, INSERT_VIP 0";
begin
inst: entity work.decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_clk_metadata_adapter_v1_0_0
     port map (
      clk_in => clk_in,
      clk_out => clk_out
    );
end STRUCTURE;
