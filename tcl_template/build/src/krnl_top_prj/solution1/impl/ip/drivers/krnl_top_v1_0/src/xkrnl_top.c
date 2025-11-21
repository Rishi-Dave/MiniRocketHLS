// ==============================================================
// Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2023.2 (64-bit)
// Tool Version Limit: 2023.10
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2023 Advanced Micro Devices, Inc. All Rights Reserved.
// 
// ==============================================================
/***************************** Include Files *********************************/
#include "xkrnl_top.h"

/************************** Function Implementation *************************/
#ifndef __linux__
int XKrnl_top_CfgInitialize(XKrnl_top *InstancePtr, XKrnl_top_Config *ConfigPtr) {
    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(ConfigPtr != NULL);

    InstancePtr->Control_BaseAddress = ConfigPtr->Control_BaseAddress;
    InstancePtr->IsReady = XIL_COMPONENT_IS_READY;

    return XST_SUCCESS;
}
#endif

void XKrnl_top_Start(XKrnl_top *InstancePtr) {
    u32 Data;

    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XKrnl_top_ReadReg(InstancePtr->Control_BaseAddress, XKRNL_TOP_CONTROL_ADDR_AP_CTRL) & 0x80;
    XKrnl_top_WriteReg(InstancePtr->Control_BaseAddress, XKRNL_TOP_CONTROL_ADDR_AP_CTRL, Data | 0x01);
}

u32 XKrnl_top_IsDone(XKrnl_top *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XKrnl_top_ReadReg(InstancePtr->Control_BaseAddress, XKRNL_TOP_CONTROL_ADDR_AP_CTRL);
    return (Data >> 1) & 0x1;
}

u32 XKrnl_top_IsIdle(XKrnl_top *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XKrnl_top_ReadReg(InstancePtr->Control_BaseAddress, XKRNL_TOP_CONTROL_ADDR_AP_CTRL);
    return (Data >> 2) & 0x1;
}

u32 XKrnl_top_IsReady(XKrnl_top *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XKrnl_top_ReadReg(InstancePtr->Control_BaseAddress, XKRNL_TOP_CONTROL_ADDR_AP_CTRL);
    // check ap_start to see if the pcore is ready for next input
    return !(Data & 0x1);
}

void XKrnl_top_EnableAutoRestart(XKrnl_top *InstancePtr) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XKrnl_top_WriteReg(InstancePtr->Control_BaseAddress, XKRNL_TOP_CONTROL_ADDR_AP_CTRL, 0x80);
}

void XKrnl_top_DisableAutoRestart(XKrnl_top *InstancePtr) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XKrnl_top_WriteReg(InstancePtr->Control_BaseAddress, XKRNL_TOP_CONTROL_ADDR_AP_CTRL, 0);
}

void XKrnl_top_Set_time_series_input(XKrnl_top *InstancePtr, u64 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XKrnl_top_WriteReg(InstancePtr->Control_BaseAddress, XKRNL_TOP_CONTROL_ADDR_TIME_SERIES_INPUT_DATA, (u32)(Data));
    XKrnl_top_WriteReg(InstancePtr->Control_BaseAddress, XKRNL_TOP_CONTROL_ADDR_TIME_SERIES_INPUT_DATA + 4, (u32)(Data >> 32));
}

u64 XKrnl_top_Get_time_series_input(XKrnl_top *InstancePtr) {
    u64 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XKrnl_top_ReadReg(InstancePtr->Control_BaseAddress, XKRNL_TOP_CONTROL_ADDR_TIME_SERIES_INPUT_DATA);
    Data += (u64)XKrnl_top_ReadReg(InstancePtr->Control_BaseAddress, XKRNL_TOP_CONTROL_ADDR_TIME_SERIES_INPUT_DATA + 4) << 32;
    return Data;
}

void XKrnl_top_Set_prediction_output(XKrnl_top *InstancePtr, u64 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XKrnl_top_WriteReg(InstancePtr->Control_BaseAddress, XKRNL_TOP_CONTROL_ADDR_PREDICTION_OUTPUT_DATA, (u32)(Data));
    XKrnl_top_WriteReg(InstancePtr->Control_BaseAddress, XKRNL_TOP_CONTROL_ADDR_PREDICTION_OUTPUT_DATA + 4, (u32)(Data >> 32));
}

u64 XKrnl_top_Get_prediction_output(XKrnl_top *InstancePtr) {
    u64 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XKrnl_top_ReadReg(InstancePtr->Control_BaseAddress, XKRNL_TOP_CONTROL_ADDR_PREDICTION_OUTPUT_DATA);
    Data += (u64)XKrnl_top_ReadReg(InstancePtr->Control_BaseAddress, XKRNL_TOP_CONTROL_ADDR_PREDICTION_OUTPUT_DATA + 4) << 32;
    return Data;
}

void XKrnl_top_Set_coefficients(XKrnl_top *InstancePtr, u64 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XKrnl_top_WriteReg(InstancePtr->Control_BaseAddress, XKRNL_TOP_CONTROL_ADDR_COEFFICIENTS_DATA, (u32)(Data));
    XKrnl_top_WriteReg(InstancePtr->Control_BaseAddress, XKRNL_TOP_CONTROL_ADDR_COEFFICIENTS_DATA + 4, (u32)(Data >> 32));
}

u64 XKrnl_top_Get_coefficients(XKrnl_top *InstancePtr) {
    u64 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XKrnl_top_ReadReg(InstancePtr->Control_BaseAddress, XKRNL_TOP_CONTROL_ADDR_COEFFICIENTS_DATA);
    Data += (u64)XKrnl_top_ReadReg(InstancePtr->Control_BaseAddress, XKRNL_TOP_CONTROL_ADDR_COEFFICIENTS_DATA + 4) << 32;
    return Data;
}

void XKrnl_top_Set_intercept(XKrnl_top *InstancePtr, u64 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XKrnl_top_WriteReg(InstancePtr->Control_BaseAddress, XKRNL_TOP_CONTROL_ADDR_INTERCEPT_DATA, (u32)(Data));
    XKrnl_top_WriteReg(InstancePtr->Control_BaseAddress, XKRNL_TOP_CONTROL_ADDR_INTERCEPT_DATA + 4, (u32)(Data >> 32));
}

u64 XKrnl_top_Get_intercept(XKrnl_top *InstancePtr) {
    u64 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XKrnl_top_ReadReg(InstancePtr->Control_BaseAddress, XKRNL_TOP_CONTROL_ADDR_INTERCEPT_DATA);
    Data += (u64)XKrnl_top_ReadReg(InstancePtr->Control_BaseAddress, XKRNL_TOP_CONTROL_ADDR_INTERCEPT_DATA + 4) << 32;
    return Data;
}

void XKrnl_top_Set_scaler_mean(XKrnl_top *InstancePtr, u64 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XKrnl_top_WriteReg(InstancePtr->Control_BaseAddress, XKRNL_TOP_CONTROL_ADDR_SCALER_MEAN_DATA, (u32)(Data));
    XKrnl_top_WriteReg(InstancePtr->Control_BaseAddress, XKRNL_TOP_CONTROL_ADDR_SCALER_MEAN_DATA + 4, (u32)(Data >> 32));
}

u64 XKrnl_top_Get_scaler_mean(XKrnl_top *InstancePtr) {
    u64 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XKrnl_top_ReadReg(InstancePtr->Control_BaseAddress, XKRNL_TOP_CONTROL_ADDR_SCALER_MEAN_DATA);
    Data += (u64)XKrnl_top_ReadReg(InstancePtr->Control_BaseAddress, XKRNL_TOP_CONTROL_ADDR_SCALER_MEAN_DATA + 4) << 32;
    return Data;
}

void XKrnl_top_Set_scaler_scale(XKrnl_top *InstancePtr, u64 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XKrnl_top_WriteReg(InstancePtr->Control_BaseAddress, XKRNL_TOP_CONTROL_ADDR_SCALER_SCALE_DATA, (u32)(Data));
    XKrnl_top_WriteReg(InstancePtr->Control_BaseAddress, XKRNL_TOP_CONTROL_ADDR_SCALER_SCALE_DATA + 4, (u32)(Data >> 32));
}

u64 XKrnl_top_Get_scaler_scale(XKrnl_top *InstancePtr) {
    u64 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XKrnl_top_ReadReg(InstancePtr->Control_BaseAddress, XKRNL_TOP_CONTROL_ADDR_SCALER_SCALE_DATA);
    Data += (u64)XKrnl_top_ReadReg(InstancePtr->Control_BaseAddress, XKRNL_TOP_CONTROL_ADDR_SCALER_SCALE_DATA + 4) << 32;
    return Data;
}

void XKrnl_top_Set_dilations(XKrnl_top *InstancePtr, u64 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XKrnl_top_WriteReg(InstancePtr->Control_BaseAddress, XKRNL_TOP_CONTROL_ADDR_DILATIONS_DATA, (u32)(Data));
    XKrnl_top_WriteReg(InstancePtr->Control_BaseAddress, XKRNL_TOP_CONTROL_ADDR_DILATIONS_DATA + 4, (u32)(Data >> 32));
}

u64 XKrnl_top_Get_dilations(XKrnl_top *InstancePtr) {
    u64 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XKrnl_top_ReadReg(InstancePtr->Control_BaseAddress, XKRNL_TOP_CONTROL_ADDR_DILATIONS_DATA);
    Data += (u64)XKrnl_top_ReadReg(InstancePtr->Control_BaseAddress, XKRNL_TOP_CONTROL_ADDR_DILATIONS_DATA + 4) << 32;
    return Data;
}

void XKrnl_top_Set_num_features_per_dilation(XKrnl_top *InstancePtr, u64 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XKrnl_top_WriteReg(InstancePtr->Control_BaseAddress, XKRNL_TOP_CONTROL_ADDR_NUM_FEATURES_PER_DILATION_DATA, (u32)(Data));
    XKrnl_top_WriteReg(InstancePtr->Control_BaseAddress, XKRNL_TOP_CONTROL_ADDR_NUM_FEATURES_PER_DILATION_DATA + 4, (u32)(Data >> 32));
}

u64 XKrnl_top_Get_num_features_per_dilation(XKrnl_top *InstancePtr) {
    u64 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XKrnl_top_ReadReg(InstancePtr->Control_BaseAddress, XKRNL_TOP_CONTROL_ADDR_NUM_FEATURES_PER_DILATION_DATA);
    Data += (u64)XKrnl_top_ReadReg(InstancePtr->Control_BaseAddress, XKRNL_TOP_CONTROL_ADDR_NUM_FEATURES_PER_DILATION_DATA + 4) << 32;
    return Data;
}

void XKrnl_top_Set_biases(XKrnl_top *InstancePtr, u64 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XKrnl_top_WriteReg(InstancePtr->Control_BaseAddress, XKRNL_TOP_CONTROL_ADDR_BIASES_DATA, (u32)(Data));
    XKrnl_top_WriteReg(InstancePtr->Control_BaseAddress, XKRNL_TOP_CONTROL_ADDR_BIASES_DATA + 4, (u32)(Data >> 32));
}

u64 XKrnl_top_Get_biases(XKrnl_top *InstancePtr) {
    u64 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XKrnl_top_ReadReg(InstancePtr->Control_BaseAddress, XKRNL_TOP_CONTROL_ADDR_BIASES_DATA);
    Data += (u64)XKrnl_top_ReadReg(InstancePtr->Control_BaseAddress, XKRNL_TOP_CONTROL_ADDR_BIASES_DATA + 4) << 32;
    return Data;
}

void XKrnl_top_Set_time_series_length(XKrnl_top *InstancePtr, u32 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XKrnl_top_WriteReg(InstancePtr->Control_BaseAddress, XKRNL_TOP_CONTROL_ADDR_TIME_SERIES_LENGTH_DATA, Data);
}

u32 XKrnl_top_Get_time_series_length(XKrnl_top *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XKrnl_top_ReadReg(InstancePtr->Control_BaseAddress, XKRNL_TOP_CONTROL_ADDR_TIME_SERIES_LENGTH_DATA);
    return Data;
}

void XKrnl_top_Set_num_features(XKrnl_top *InstancePtr, u32 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XKrnl_top_WriteReg(InstancePtr->Control_BaseAddress, XKRNL_TOP_CONTROL_ADDR_NUM_FEATURES_DATA, Data);
}

u32 XKrnl_top_Get_num_features(XKrnl_top *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XKrnl_top_ReadReg(InstancePtr->Control_BaseAddress, XKRNL_TOP_CONTROL_ADDR_NUM_FEATURES_DATA);
    return Data;
}

void XKrnl_top_Set_num_classes(XKrnl_top *InstancePtr, u32 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XKrnl_top_WriteReg(InstancePtr->Control_BaseAddress, XKRNL_TOP_CONTROL_ADDR_NUM_CLASSES_DATA, Data);
}

u32 XKrnl_top_Get_num_classes(XKrnl_top *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XKrnl_top_ReadReg(InstancePtr->Control_BaseAddress, XKRNL_TOP_CONTROL_ADDR_NUM_CLASSES_DATA);
    return Data;
}

void XKrnl_top_Set_num_dilations(XKrnl_top *InstancePtr, u32 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XKrnl_top_WriteReg(InstancePtr->Control_BaseAddress, XKRNL_TOP_CONTROL_ADDR_NUM_DILATIONS_DATA, Data);
}

u32 XKrnl_top_Get_num_dilations(XKrnl_top *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XKrnl_top_ReadReg(InstancePtr->Control_BaseAddress, XKRNL_TOP_CONTROL_ADDR_NUM_DILATIONS_DATA);
    return Data;
}

void XKrnl_top_InterruptGlobalEnable(XKrnl_top *InstancePtr) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XKrnl_top_WriteReg(InstancePtr->Control_BaseAddress, XKRNL_TOP_CONTROL_ADDR_GIE, 1);
}

void XKrnl_top_InterruptGlobalDisable(XKrnl_top *InstancePtr) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XKrnl_top_WriteReg(InstancePtr->Control_BaseAddress, XKRNL_TOP_CONTROL_ADDR_GIE, 0);
}

void XKrnl_top_InterruptEnable(XKrnl_top *InstancePtr, u32 Mask) {
    u32 Register;

    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Register =  XKrnl_top_ReadReg(InstancePtr->Control_BaseAddress, XKRNL_TOP_CONTROL_ADDR_IER);
    XKrnl_top_WriteReg(InstancePtr->Control_BaseAddress, XKRNL_TOP_CONTROL_ADDR_IER, Register | Mask);
}

void XKrnl_top_InterruptDisable(XKrnl_top *InstancePtr, u32 Mask) {
    u32 Register;

    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Register =  XKrnl_top_ReadReg(InstancePtr->Control_BaseAddress, XKRNL_TOP_CONTROL_ADDR_IER);
    XKrnl_top_WriteReg(InstancePtr->Control_BaseAddress, XKRNL_TOP_CONTROL_ADDR_IER, Register & (~Mask));
}

void XKrnl_top_InterruptClear(XKrnl_top *InstancePtr, u32 Mask) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XKrnl_top_WriteReg(InstancePtr->Control_BaseAddress, XKRNL_TOP_CONTROL_ADDR_ISR, Mask);
}

u32 XKrnl_top_InterruptGetEnabled(XKrnl_top *InstancePtr) {
    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    return XKrnl_top_ReadReg(InstancePtr->Control_BaseAddress, XKRNL_TOP_CONTROL_ADDR_IER);
}

u32 XKrnl_top_InterruptGetStatus(XKrnl_top *InstancePtr) {
    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    return XKrnl_top_ReadReg(InstancePtr->Control_BaseAddress, XKRNL_TOP_CONTROL_ADDR_ISR);
}

