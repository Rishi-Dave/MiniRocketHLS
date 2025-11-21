// ==============================================================
// Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2023.2 (64-bit)
// Tool Version Limit: 2023.10
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2023 Advanced Micro Devices, Inc. All Rights Reserved.
// 
// ==============================================================
// control
// 0x00 : Control signals
//        bit 0  - ap_start (Read/Write/COH)
//        bit 1  - ap_done (Read/COR)
//        bit 2  - ap_idle (Read)
//        bit 3  - ap_ready (Read/COR)
//        bit 7  - auto_restart (Read/Write)
//        bit 9  - interrupt (Read)
//        others - reserved
// 0x04 : Global Interrupt Enable Register
//        bit 0  - Global Interrupt Enable (Read/Write)
//        others - reserved
// 0x08 : IP Interrupt Enable Register (Read/Write)
//        bit 0 - enable ap_done interrupt (Read/Write)
//        bit 1 - enable ap_ready interrupt (Read/Write)
//        others - reserved
// 0x0c : IP Interrupt Status Register (Read/TOW)
//        bit 0 - ap_done (Read/TOW)
//        bit 1 - ap_ready (Read/TOW)
//        others - reserved
// 0x10 : Data signal of time_series_input
//        bit 31~0 - time_series_input[31:0] (Read/Write)
// 0x14 : Data signal of time_series_input
//        bit 31~0 - time_series_input[63:32] (Read/Write)
// 0x18 : reserved
// 0x1c : Data signal of prediction_output
//        bit 31~0 - prediction_output[31:0] (Read/Write)
// 0x20 : Data signal of prediction_output
//        bit 31~0 - prediction_output[63:32] (Read/Write)
// 0x24 : reserved
// 0x28 : Data signal of coefficients
//        bit 31~0 - coefficients[31:0] (Read/Write)
// 0x2c : Data signal of coefficients
//        bit 31~0 - coefficients[63:32] (Read/Write)
// 0x30 : reserved
// 0x34 : Data signal of intercept
//        bit 31~0 - intercept[31:0] (Read/Write)
// 0x38 : Data signal of intercept
//        bit 31~0 - intercept[63:32] (Read/Write)
// 0x3c : reserved
// 0x40 : Data signal of scaler_mean
//        bit 31~0 - scaler_mean[31:0] (Read/Write)
// 0x44 : Data signal of scaler_mean
//        bit 31~0 - scaler_mean[63:32] (Read/Write)
// 0x48 : reserved
// 0x4c : Data signal of scaler_scale
//        bit 31~0 - scaler_scale[31:0] (Read/Write)
// 0x50 : Data signal of scaler_scale
//        bit 31~0 - scaler_scale[63:32] (Read/Write)
// 0x54 : reserved
// 0x58 : Data signal of dilations
//        bit 31~0 - dilations[31:0] (Read/Write)
// 0x5c : Data signal of dilations
//        bit 31~0 - dilations[63:32] (Read/Write)
// 0x60 : reserved
// 0x64 : Data signal of num_features_per_dilation
//        bit 31~0 - num_features_per_dilation[31:0] (Read/Write)
// 0x68 : Data signal of num_features_per_dilation
//        bit 31~0 - num_features_per_dilation[63:32] (Read/Write)
// 0x6c : reserved
// 0x70 : Data signal of biases
//        bit 31~0 - biases[31:0] (Read/Write)
// 0x74 : Data signal of biases
//        bit 31~0 - biases[63:32] (Read/Write)
// 0x78 : reserved
// 0x7c : Data signal of time_series_length
//        bit 31~0 - time_series_length[31:0] (Read/Write)
// 0x80 : reserved
// 0x84 : Data signal of num_features
//        bit 31~0 - num_features[31:0] (Read/Write)
// 0x88 : reserved
// 0x8c : Data signal of num_classes
//        bit 31~0 - num_classes[31:0] (Read/Write)
// 0x90 : reserved
// 0x94 : Data signal of num_dilations
//        bit 31~0 - num_dilations[31:0] (Read/Write)
// 0x98 : reserved
// (SC = Self Clear, COR = Clear on Read, TOW = Toggle on Write, COH = Clear on Handshake)

#define XKRNL_TOP_CONTROL_ADDR_AP_CTRL                        0x00
#define XKRNL_TOP_CONTROL_ADDR_GIE                            0x04
#define XKRNL_TOP_CONTROL_ADDR_IER                            0x08
#define XKRNL_TOP_CONTROL_ADDR_ISR                            0x0c
#define XKRNL_TOP_CONTROL_ADDR_TIME_SERIES_INPUT_DATA         0x10
#define XKRNL_TOP_CONTROL_BITS_TIME_SERIES_INPUT_DATA         64
#define XKRNL_TOP_CONTROL_ADDR_PREDICTION_OUTPUT_DATA         0x1c
#define XKRNL_TOP_CONTROL_BITS_PREDICTION_OUTPUT_DATA         64
#define XKRNL_TOP_CONTROL_ADDR_COEFFICIENTS_DATA              0x28
#define XKRNL_TOP_CONTROL_BITS_COEFFICIENTS_DATA              64
#define XKRNL_TOP_CONTROL_ADDR_INTERCEPT_DATA                 0x34
#define XKRNL_TOP_CONTROL_BITS_INTERCEPT_DATA                 64
#define XKRNL_TOP_CONTROL_ADDR_SCALER_MEAN_DATA               0x40
#define XKRNL_TOP_CONTROL_BITS_SCALER_MEAN_DATA               64
#define XKRNL_TOP_CONTROL_ADDR_SCALER_SCALE_DATA              0x4c
#define XKRNL_TOP_CONTROL_BITS_SCALER_SCALE_DATA              64
#define XKRNL_TOP_CONTROL_ADDR_DILATIONS_DATA                 0x58
#define XKRNL_TOP_CONTROL_BITS_DILATIONS_DATA                 64
#define XKRNL_TOP_CONTROL_ADDR_NUM_FEATURES_PER_DILATION_DATA 0x64
#define XKRNL_TOP_CONTROL_BITS_NUM_FEATURES_PER_DILATION_DATA 64
#define XKRNL_TOP_CONTROL_ADDR_BIASES_DATA                    0x70
#define XKRNL_TOP_CONTROL_BITS_BIASES_DATA                    64
#define XKRNL_TOP_CONTROL_ADDR_TIME_SERIES_LENGTH_DATA        0x7c
#define XKRNL_TOP_CONTROL_BITS_TIME_SERIES_LENGTH_DATA        32
#define XKRNL_TOP_CONTROL_ADDR_NUM_FEATURES_DATA              0x84
#define XKRNL_TOP_CONTROL_BITS_NUM_FEATURES_DATA              32
#define XKRNL_TOP_CONTROL_ADDR_NUM_CLASSES_DATA               0x8c
#define XKRNL_TOP_CONTROL_BITS_NUM_CLASSES_DATA               32
#define XKRNL_TOP_CONTROL_ADDR_NUM_DILATIONS_DATA             0x94
#define XKRNL_TOP_CONTROL_BITS_NUM_DILATIONS_DATA             32

