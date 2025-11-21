// ==============================================================
// Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2023.2 (64-bit)
// Tool Version Limit: 2023.10
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2023 Advanced Micro Devices, Inc. All Rights Reserved.
// 
// ==============================================================
#ifndef XKRNL_TOP_H
#define XKRNL_TOP_H

#ifdef __cplusplus
extern "C" {
#endif

/***************************** Include Files *********************************/
#ifndef __linux__
#include "xil_types.h"
#include "xil_assert.h"
#include "xstatus.h"
#include "xil_io.h"
#else
#include <stdint.h>
#include <assert.h>
#include <dirent.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>
#include <stddef.h>
#endif
#include "xkrnl_top_hw.h"

/**************************** Type Definitions ******************************/
#ifdef __linux__
typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;
#else
typedef struct {
#ifdef SDT
    char *Name;
#else
    u16 DeviceId;
#endif
    u64 Control_BaseAddress;
} XKrnl_top_Config;
#endif

typedef struct {
    u64 Control_BaseAddress;
    u32 IsReady;
} XKrnl_top;

typedef u32 word_type;

/***************** Macros (Inline Functions) Definitions *********************/
#ifndef __linux__
#define XKrnl_top_WriteReg(BaseAddress, RegOffset, Data) \
    Xil_Out32((BaseAddress) + (RegOffset), (u32)(Data))
#define XKrnl_top_ReadReg(BaseAddress, RegOffset) \
    Xil_In32((BaseAddress) + (RegOffset))
#else
#define XKrnl_top_WriteReg(BaseAddress, RegOffset, Data) \
    *(volatile u32*)((BaseAddress) + (RegOffset)) = (u32)(Data)
#define XKrnl_top_ReadReg(BaseAddress, RegOffset) \
    *(volatile u32*)((BaseAddress) + (RegOffset))

#define Xil_AssertVoid(expr)    assert(expr)
#define Xil_AssertNonvoid(expr) assert(expr)

#define XST_SUCCESS             0
#define XST_DEVICE_NOT_FOUND    2
#define XST_OPEN_DEVICE_FAILED  3
#define XIL_COMPONENT_IS_READY  1
#endif

/************************** Function Prototypes *****************************/
#ifndef __linux__
#ifdef SDT
int XKrnl_top_Initialize(XKrnl_top *InstancePtr, UINTPTR BaseAddress);
XKrnl_top_Config* XKrnl_top_LookupConfig(UINTPTR BaseAddress);
#else
int XKrnl_top_Initialize(XKrnl_top *InstancePtr, u16 DeviceId);
XKrnl_top_Config* XKrnl_top_LookupConfig(u16 DeviceId);
#endif
int XKrnl_top_CfgInitialize(XKrnl_top *InstancePtr, XKrnl_top_Config *ConfigPtr);
#else
int XKrnl_top_Initialize(XKrnl_top *InstancePtr, const char* InstanceName);
int XKrnl_top_Release(XKrnl_top *InstancePtr);
#endif

void XKrnl_top_Start(XKrnl_top *InstancePtr);
u32 XKrnl_top_IsDone(XKrnl_top *InstancePtr);
u32 XKrnl_top_IsIdle(XKrnl_top *InstancePtr);
u32 XKrnl_top_IsReady(XKrnl_top *InstancePtr);
void XKrnl_top_EnableAutoRestart(XKrnl_top *InstancePtr);
void XKrnl_top_DisableAutoRestart(XKrnl_top *InstancePtr);

void XKrnl_top_Set_time_series_input(XKrnl_top *InstancePtr, u64 Data);
u64 XKrnl_top_Get_time_series_input(XKrnl_top *InstancePtr);
void XKrnl_top_Set_prediction_output(XKrnl_top *InstancePtr, u64 Data);
u64 XKrnl_top_Get_prediction_output(XKrnl_top *InstancePtr);
void XKrnl_top_Set_coefficients(XKrnl_top *InstancePtr, u64 Data);
u64 XKrnl_top_Get_coefficients(XKrnl_top *InstancePtr);
void XKrnl_top_Set_intercept(XKrnl_top *InstancePtr, u64 Data);
u64 XKrnl_top_Get_intercept(XKrnl_top *InstancePtr);
void XKrnl_top_Set_scaler_mean(XKrnl_top *InstancePtr, u64 Data);
u64 XKrnl_top_Get_scaler_mean(XKrnl_top *InstancePtr);
void XKrnl_top_Set_scaler_scale(XKrnl_top *InstancePtr, u64 Data);
u64 XKrnl_top_Get_scaler_scale(XKrnl_top *InstancePtr);
void XKrnl_top_Set_dilations(XKrnl_top *InstancePtr, u64 Data);
u64 XKrnl_top_Get_dilations(XKrnl_top *InstancePtr);
void XKrnl_top_Set_num_features_per_dilation(XKrnl_top *InstancePtr, u64 Data);
u64 XKrnl_top_Get_num_features_per_dilation(XKrnl_top *InstancePtr);
void XKrnl_top_Set_biases(XKrnl_top *InstancePtr, u64 Data);
u64 XKrnl_top_Get_biases(XKrnl_top *InstancePtr);
void XKrnl_top_Set_time_series_length(XKrnl_top *InstancePtr, u32 Data);
u32 XKrnl_top_Get_time_series_length(XKrnl_top *InstancePtr);
void XKrnl_top_Set_num_features(XKrnl_top *InstancePtr, u32 Data);
u32 XKrnl_top_Get_num_features(XKrnl_top *InstancePtr);
void XKrnl_top_Set_num_classes(XKrnl_top *InstancePtr, u32 Data);
u32 XKrnl_top_Get_num_classes(XKrnl_top *InstancePtr);
void XKrnl_top_Set_num_dilations(XKrnl_top *InstancePtr, u32 Data);
u32 XKrnl_top_Get_num_dilations(XKrnl_top *InstancePtr);

void XKrnl_top_InterruptGlobalEnable(XKrnl_top *InstancePtr);
void XKrnl_top_InterruptGlobalDisable(XKrnl_top *InstancePtr);
void XKrnl_top_InterruptEnable(XKrnl_top *InstancePtr, u32 Mask);
void XKrnl_top_InterruptDisable(XKrnl_top *InstancePtr, u32 Mask);
void XKrnl_top_InterruptClear(XKrnl_top *InstancePtr, u32 Mask);
u32 XKrnl_top_InterruptGetEnabled(XKrnl_top *InstancePtr);
u32 XKrnl_top_InterruptGetStatus(XKrnl_top *InstancePtr);

#ifdef __cplusplus
}
#endif

#endif
