// ==============================================================
// Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2023.2 (64-bit)
// Tool Version Limit: 2023.10
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2023 Advanced Micro Devices, Inc. All Rights Reserved.
// 
// ==============================================================
`timescale 1ns/1ps
module krnl_top_control_s_axi
#(parameter
    C_S_AXI_ADDR_WIDTH = 8,
    C_S_AXI_DATA_WIDTH = 32
)(
    input  wire                          ACLK,
    input  wire                          ARESET,
    input  wire                          ACLK_EN,
    input  wire [C_S_AXI_ADDR_WIDTH-1:0] AWADDR,
    input  wire                          AWVALID,
    output wire                          AWREADY,
    input  wire [C_S_AXI_DATA_WIDTH-1:0] WDATA,
    input  wire [C_S_AXI_DATA_WIDTH/8-1:0] WSTRB,
    input  wire                          WVALID,
    output wire                          WREADY,
    output wire [1:0]                    BRESP,
    output wire                          BVALID,
    input  wire                          BREADY,
    input  wire [C_S_AXI_ADDR_WIDTH-1:0] ARADDR,
    input  wire                          ARVALID,
    output wire                          ARREADY,
    output wire [C_S_AXI_DATA_WIDTH-1:0] RDATA,
    output wire [1:0]                    RRESP,
    output wire                          RVALID,
    input  wire                          RREADY,
    output wire                          interrupt,
    output wire [63:0]                   time_series_input,
    output wire [63:0]                   prediction_output,
    output wire [63:0]                   coefficients,
    output wire [63:0]                   intercept,
    output wire [63:0]                   scaler_mean,
    output wire [63:0]                   scaler_scale,
    output wire [63:0]                   dilations,
    output wire [63:0]                   num_features_per_dilation,
    output wire [63:0]                   biases,
    output wire [31:0]                   time_series_length,
    output wire [31:0]                   num_features,
    output wire [31:0]                   num_classes,
    output wire [31:0]                   num_dilations,
    output wire                          ap_start,
    input  wire                          ap_done,
    input  wire                          ap_ready,
    input  wire                          ap_idle
);
//------------------------Address Info-------------------
// Protocol Used: ap_ctrl_hs
//
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

//------------------------Parameter----------------------
localparam
    ADDR_AP_CTRL                          = 8'h00,
    ADDR_GIE                              = 8'h04,
    ADDR_IER                              = 8'h08,
    ADDR_ISR                              = 8'h0c,
    ADDR_TIME_SERIES_INPUT_DATA_0         = 8'h10,
    ADDR_TIME_SERIES_INPUT_DATA_1         = 8'h14,
    ADDR_TIME_SERIES_INPUT_CTRL           = 8'h18,
    ADDR_PREDICTION_OUTPUT_DATA_0         = 8'h1c,
    ADDR_PREDICTION_OUTPUT_DATA_1         = 8'h20,
    ADDR_PREDICTION_OUTPUT_CTRL           = 8'h24,
    ADDR_COEFFICIENTS_DATA_0              = 8'h28,
    ADDR_COEFFICIENTS_DATA_1              = 8'h2c,
    ADDR_COEFFICIENTS_CTRL                = 8'h30,
    ADDR_INTERCEPT_DATA_0                 = 8'h34,
    ADDR_INTERCEPT_DATA_1                 = 8'h38,
    ADDR_INTERCEPT_CTRL                   = 8'h3c,
    ADDR_SCALER_MEAN_DATA_0               = 8'h40,
    ADDR_SCALER_MEAN_DATA_1               = 8'h44,
    ADDR_SCALER_MEAN_CTRL                 = 8'h48,
    ADDR_SCALER_SCALE_DATA_0              = 8'h4c,
    ADDR_SCALER_SCALE_DATA_1              = 8'h50,
    ADDR_SCALER_SCALE_CTRL                = 8'h54,
    ADDR_DILATIONS_DATA_0                 = 8'h58,
    ADDR_DILATIONS_DATA_1                 = 8'h5c,
    ADDR_DILATIONS_CTRL                   = 8'h60,
    ADDR_NUM_FEATURES_PER_DILATION_DATA_0 = 8'h64,
    ADDR_NUM_FEATURES_PER_DILATION_DATA_1 = 8'h68,
    ADDR_NUM_FEATURES_PER_DILATION_CTRL   = 8'h6c,
    ADDR_BIASES_DATA_0                    = 8'h70,
    ADDR_BIASES_DATA_1                    = 8'h74,
    ADDR_BIASES_CTRL                      = 8'h78,
    ADDR_TIME_SERIES_LENGTH_DATA_0        = 8'h7c,
    ADDR_TIME_SERIES_LENGTH_CTRL          = 8'h80,
    ADDR_NUM_FEATURES_DATA_0              = 8'h84,
    ADDR_NUM_FEATURES_CTRL                = 8'h88,
    ADDR_NUM_CLASSES_DATA_0               = 8'h8c,
    ADDR_NUM_CLASSES_CTRL                 = 8'h90,
    ADDR_NUM_DILATIONS_DATA_0             = 8'h94,
    ADDR_NUM_DILATIONS_CTRL               = 8'h98,
    WRIDLE                                = 2'd0,
    WRDATA                                = 2'd1,
    WRRESP                                = 2'd2,
    WRRESET                               = 2'd3,
    RDIDLE                                = 2'd0,
    RDDATA                                = 2'd1,
    RDRESET                               = 2'd2,
    ADDR_BITS                = 8;

//------------------------Local signal-------------------
    reg  [1:0]                    wstate = WRRESET;
    reg  [1:0]                    wnext;
    reg  [ADDR_BITS-1:0]          waddr;
    wire [C_S_AXI_DATA_WIDTH-1:0] wmask;
    wire                          aw_hs;
    wire                          w_hs;
    reg  [1:0]                    rstate = RDRESET;
    reg  [1:0]                    rnext;
    reg  [C_S_AXI_DATA_WIDTH-1:0] rdata;
    wire                          ar_hs;
    wire [ADDR_BITS-1:0]          raddr;
    // internal registers
    reg                           int_ap_idle;
    reg                           int_ap_ready = 1'b0;
    wire                          task_ap_ready;
    reg                           int_ap_done = 1'b0;
    wire                          task_ap_done;
    reg                           int_task_ap_done = 1'b0;
    reg                           int_ap_start = 1'b0;
    reg                           int_interrupt = 1'b0;
    reg                           int_auto_restart = 1'b0;
    reg                           auto_restart_status = 1'b0;
    wire                          auto_restart_done;
    reg                           int_gie = 1'b0;
    reg  [1:0]                    int_ier = 2'b0;
    reg  [1:0]                    int_isr = 2'b0;
    reg  [63:0]                   int_time_series_input = 'b0;
    reg  [63:0]                   int_prediction_output = 'b0;
    reg  [63:0]                   int_coefficients = 'b0;
    reg  [63:0]                   int_intercept = 'b0;
    reg  [63:0]                   int_scaler_mean = 'b0;
    reg  [63:0]                   int_scaler_scale = 'b0;
    reg  [63:0]                   int_dilations = 'b0;
    reg  [63:0]                   int_num_features_per_dilation = 'b0;
    reg  [63:0]                   int_biases = 'b0;
    reg  [31:0]                   int_time_series_length = 'b0;
    reg  [31:0]                   int_num_features = 'b0;
    reg  [31:0]                   int_num_classes = 'b0;
    reg  [31:0]                   int_num_dilations = 'b0;

//------------------------Instantiation------------------


//------------------------AXI write fsm------------------
assign AWREADY = (wstate == WRIDLE);
assign WREADY  = (wstate == WRDATA);
assign BRESP   = 2'b00;  // OKAY
assign BVALID  = (wstate == WRRESP);
assign wmask   = { {8{WSTRB[3]}}, {8{WSTRB[2]}}, {8{WSTRB[1]}}, {8{WSTRB[0]}} };
assign aw_hs   = AWVALID & AWREADY;
assign w_hs    = WVALID & WREADY;

// wstate
always @(posedge ACLK) begin
    if (ARESET)
        wstate <= WRRESET;
    else if (ACLK_EN)
        wstate <= wnext;
end

// wnext
always @(*) begin
    case (wstate)
        WRIDLE:
            if (AWVALID)
                wnext = WRDATA;
            else
                wnext = WRIDLE;
        WRDATA:
            if (WVALID)
                wnext = WRRESP;
            else
                wnext = WRDATA;
        WRRESP:
            if (BREADY)
                wnext = WRIDLE;
            else
                wnext = WRRESP;
        default:
            wnext = WRIDLE;
    endcase
end

// waddr
always @(posedge ACLK) begin
    if (ACLK_EN) begin
        if (aw_hs)
            waddr <= AWADDR[ADDR_BITS-1:0];
    end
end

//------------------------AXI read fsm-------------------
assign ARREADY = (rstate == RDIDLE);
assign RDATA   = rdata;
assign RRESP   = 2'b00;  // OKAY
assign RVALID  = (rstate == RDDATA);
assign ar_hs   = ARVALID & ARREADY;
assign raddr   = ARADDR[ADDR_BITS-1:0];

// rstate
always @(posedge ACLK) begin
    if (ARESET)
        rstate <= RDRESET;
    else if (ACLK_EN)
        rstate <= rnext;
end

// rnext
always @(*) begin
    case (rstate)
        RDIDLE:
            if (ARVALID)
                rnext = RDDATA;
            else
                rnext = RDIDLE;
        RDDATA:
            if (RREADY & RVALID)
                rnext = RDIDLE;
            else
                rnext = RDDATA;
        default:
            rnext = RDIDLE;
    endcase
end

// rdata
always @(posedge ACLK) begin
    if (ACLK_EN) begin
        if (ar_hs) begin
            rdata <= 'b0;
            case (raddr)
                ADDR_AP_CTRL: begin
                    rdata[0] <= int_ap_start;
                    rdata[1] <= int_task_ap_done;
                    rdata[2] <= int_ap_idle;
                    rdata[3] <= int_ap_ready;
                    rdata[7] <= int_auto_restart;
                    rdata[9] <= int_interrupt;
                end
                ADDR_GIE: begin
                    rdata <= int_gie;
                end
                ADDR_IER: begin
                    rdata <= int_ier;
                end
                ADDR_ISR: begin
                    rdata <= int_isr;
                end
                ADDR_TIME_SERIES_INPUT_DATA_0: begin
                    rdata <= int_time_series_input[31:0];
                end
                ADDR_TIME_SERIES_INPUT_DATA_1: begin
                    rdata <= int_time_series_input[63:32];
                end
                ADDR_PREDICTION_OUTPUT_DATA_0: begin
                    rdata <= int_prediction_output[31:0];
                end
                ADDR_PREDICTION_OUTPUT_DATA_1: begin
                    rdata <= int_prediction_output[63:32];
                end
                ADDR_COEFFICIENTS_DATA_0: begin
                    rdata <= int_coefficients[31:0];
                end
                ADDR_COEFFICIENTS_DATA_1: begin
                    rdata <= int_coefficients[63:32];
                end
                ADDR_INTERCEPT_DATA_0: begin
                    rdata <= int_intercept[31:0];
                end
                ADDR_INTERCEPT_DATA_1: begin
                    rdata <= int_intercept[63:32];
                end
                ADDR_SCALER_MEAN_DATA_0: begin
                    rdata <= int_scaler_mean[31:0];
                end
                ADDR_SCALER_MEAN_DATA_1: begin
                    rdata <= int_scaler_mean[63:32];
                end
                ADDR_SCALER_SCALE_DATA_0: begin
                    rdata <= int_scaler_scale[31:0];
                end
                ADDR_SCALER_SCALE_DATA_1: begin
                    rdata <= int_scaler_scale[63:32];
                end
                ADDR_DILATIONS_DATA_0: begin
                    rdata <= int_dilations[31:0];
                end
                ADDR_DILATIONS_DATA_1: begin
                    rdata <= int_dilations[63:32];
                end
                ADDR_NUM_FEATURES_PER_DILATION_DATA_0: begin
                    rdata <= int_num_features_per_dilation[31:0];
                end
                ADDR_NUM_FEATURES_PER_DILATION_DATA_1: begin
                    rdata <= int_num_features_per_dilation[63:32];
                end
                ADDR_BIASES_DATA_0: begin
                    rdata <= int_biases[31:0];
                end
                ADDR_BIASES_DATA_1: begin
                    rdata <= int_biases[63:32];
                end
                ADDR_TIME_SERIES_LENGTH_DATA_0: begin
                    rdata <= int_time_series_length[31:0];
                end
                ADDR_NUM_FEATURES_DATA_0: begin
                    rdata <= int_num_features[31:0];
                end
                ADDR_NUM_CLASSES_DATA_0: begin
                    rdata <= int_num_classes[31:0];
                end
                ADDR_NUM_DILATIONS_DATA_0: begin
                    rdata <= int_num_dilations[31:0];
                end
            endcase
        end
    end
end


//------------------------Register logic-----------------
assign interrupt                 = int_interrupt;
assign ap_start                  = int_ap_start;
assign task_ap_done              = (ap_done && !auto_restart_status) || auto_restart_done;
assign task_ap_ready             = ap_ready && !int_auto_restart;
assign auto_restart_done         = auto_restart_status && (ap_idle && !int_ap_idle);
assign time_series_input         = int_time_series_input;
assign prediction_output         = int_prediction_output;
assign coefficients              = int_coefficients;
assign intercept                 = int_intercept;
assign scaler_mean               = int_scaler_mean;
assign scaler_scale              = int_scaler_scale;
assign dilations                 = int_dilations;
assign num_features_per_dilation = int_num_features_per_dilation;
assign biases                    = int_biases;
assign time_series_length        = int_time_series_length;
assign num_features              = int_num_features;
assign num_classes               = int_num_classes;
assign num_dilations             = int_num_dilations;
// int_interrupt
always @(posedge ACLK) begin
    if (ARESET)
        int_interrupt <= 1'b0;
    else if (ACLK_EN) begin
        if (int_gie && (|int_isr))
            int_interrupt <= 1'b1;
        else
            int_interrupt <= 1'b0;
    end
end

// int_ap_start
always @(posedge ACLK) begin
    if (ARESET)
        int_ap_start <= 1'b0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_AP_CTRL && WSTRB[0] && WDATA[0])
            int_ap_start <= 1'b1;
        else if (ap_ready)
            int_ap_start <= int_auto_restart; // clear on handshake/auto restart
    end
end

// int_ap_done
always @(posedge ACLK) begin
    if (ARESET)
        int_ap_done <= 1'b0;
    else if (ACLK_EN) begin
            int_ap_done <= ap_done;
    end
end

// int_task_ap_done
always @(posedge ACLK) begin
    if (ARESET)
        int_task_ap_done <= 1'b0;
    else if (ACLK_EN) begin
        if (task_ap_done)
            int_task_ap_done <= 1'b1;
        else if (ar_hs && raddr == ADDR_AP_CTRL)
            int_task_ap_done <= 1'b0; // clear on read
    end
end

// int_ap_idle
always @(posedge ACLK) begin
    if (ARESET)
        int_ap_idle <= 1'b0;
    else if (ACLK_EN) begin
            int_ap_idle <= ap_idle;
    end
end

// int_ap_ready
always @(posedge ACLK) begin
    if (ARESET)
        int_ap_ready <= 1'b0;
    else if (ACLK_EN) begin
        if (task_ap_ready)
            int_ap_ready <= 1'b1;
        else if (ar_hs && raddr == ADDR_AP_CTRL)
            int_ap_ready <= 1'b0;
    end
end

// int_auto_restart
always @(posedge ACLK) begin
    if (ARESET)
        int_auto_restart <= 1'b0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_AP_CTRL && WSTRB[0])
            int_auto_restart <=  WDATA[7];
    end
end

// auto_restart_status
always @(posedge ACLK) begin
    if (ARESET)
        auto_restart_status <= 1'b0;
    else if (ACLK_EN) begin
        if (int_auto_restart)
            auto_restart_status <= 1'b1;
        else if (ap_idle)
            auto_restart_status <= 1'b0;
    end
end

// int_gie
always @(posedge ACLK) begin
    if (ARESET)
        int_gie <= 1'b0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_GIE && WSTRB[0])
            int_gie <= WDATA[0];
    end
end

// int_ier
always @(posedge ACLK) begin
    if (ARESET)
        int_ier <= 1'b0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_IER && WSTRB[0])
            int_ier <= WDATA[1:0];
    end
end

// int_isr[0]
always @(posedge ACLK) begin
    if (ARESET)
        int_isr[0] <= 1'b0;
    else if (ACLK_EN) begin
        if (int_ier[0] & ap_done)
            int_isr[0] <= 1'b1;
        else if (w_hs && waddr == ADDR_ISR && WSTRB[0])
            int_isr[0] <= int_isr[0] ^ WDATA[0]; // toggle on write
    end
end

// int_isr[1]
always @(posedge ACLK) begin
    if (ARESET)
        int_isr[1] <= 1'b0;
    else if (ACLK_EN) begin
        if (int_ier[1] & ap_ready)
            int_isr[1] <= 1'b1;
        else if (w_hs && waddr == ADDR_ISR && WSTRB[0])
            int_isr[1] <= int_isr[1] ^ WDATA[1]; // toggle on write
    end
end

// int_time_series_input[31:0]
always @(posedge ACLK) begin
    if (ARESET)
        int_time_series_input[31:0] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_TIME_SERIES_INPUT_DATA_0)
            int_time_series_input[31:0] <= (WDATA[31:0] & wmask) | (int_time_series_input[31:0] & ~wmask);
    end
end

// int_time_series_input[63:32]
always @(posedge ACLK) begin
    if (ARESET)
        int_time_series_input[63:32] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_TIME_SERIES_INPUT_DATA_1)
            int_time_series_input[63:32] <= (WDATA[31:0] & wmask) | (int_time_series_input[63:32] & ~wmask);
    end
end

// int_prediction_output[31:0]
always @(posedge ACLK) begin
    if (ARESET)
        int_prediction_output[31:0] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_PREDICTION_OUTPUT_DATA_0)
            int_prediction_output[31:0] <= (WDATA[31:0] & wmask) | (int_prediction_output[31:0] & ~wmask);
    end
end

// int_prediction_output[63:32]
always @(posedge ACLK) begin
    if (ARESET)
        int_prediction_output[63:32] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_PREDICTION_OUTPUT_DATA_1)
            int_prediction_output[63:32] <= (WDATA[31:0] & wmask) | (int_prediction_output[63:32] & ~wmask);
    end
end

// int_coefficients[31:0]
always @(posedge ACLK) begin
    if (ARESET)
        int_coefficients[31:0] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_COEFFICIENTS_DATA_0)
            int_coefficients[31:0] <= (WDATA[31:0] & wmask) | (int_coefficients[31:0] & ~wmask);
    end
end

// int_coefficients[63:32]
always @(posedge ACLK) begin
    if (ARESET)
        int_coefficients[63:32] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_COEFFICIENTS_DATA_1)
            int_coefficients[63:32] <= (WDATA[31:0] & wmask) | (int_coefficients[63:32] & ~wmask);
    end
end

// int_intercept[31:0]
always @(posedge ACLK) begin
    if (ARESET)
        int_intercept[31:0] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_INTERCEPT_DATA_0)
            int_intercept[31:0] <= (WDATA[31:0] & wmask) | (int_intercept[31:0] & ~wmask);
    end
end

// int_intercept[63:32]
always @(posedge ACLK) begin
    if (ARESET)
        int_intercept[63:32] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_INTERCEPT_DATA_1)
            int_intercept[63:32] <= (WDATA[31:0] & wmask) | (int_intercept[63:32] & ~wmask);
    end
end

// int_scaler_mean[31:0]
always @(posedge ACLK) begin
    if (ARESET)
        int_scaler_mean[31:0] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_SCALER_MEAN_DATA_0)
            int_scaler_mean[31:0] <= (WDATA[31:0] & wmask) | (int_scaler_mean[31:0] & ~wmask);
    end
end

// int_scaler_mean[63:32]
always @(posedge ACLK) begin
    if (ARESET)
        int_scaler_mean[63:32] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_SCALER_MEAN_DATA_1)
            int_scaler_mean[63:32] <= (WDATA[31:0] & wmask) | (int_scaler_mean[63:32] & ~wmask);
    end
end

// int_scaler_scale[31:0]
always @(posedge ACLK) begin
    if (ARESET)
        int_scaler_scale[31:0] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_SCALER_SCALE_DATA_0)
            int_scaler_scale[31:0] <= (WDATA[31:0] & wmask) | (int_scaler_scale[31:0] & ~wmask);
    end
end

// int_scaler_scale[63:32]
always @(posedge ACLK) begin
    if (ARESET)
        int_scaler_scale[63:32] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_SCALER_SCALE_DATA_1)
            int_scaler_scale[63:32] <= (WDATA[31:0] & wmask) | (int_scaler_scale[63:32] & ~wmask);
    end
end

// int_dilations[31:0]
always @(posedge ACLK) begin
    if (ARESET)
        int_dilations[31:0] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_DILATIONS_DATA_0)
            int_dilations[31:0] <= (WDATA[31:0] & wmask) | (int_dilations[31:0] & ~wmask);
    end
end

// int_dilations[63:32]
always @(posedge ACLK) begin
    if (ARESET)
        int_dilations[63:32] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_DILATIONS_DATA_1)
            int_dilations[63:32] <= (WDATA[31:0] & wmask) | (int_dilations[63:32] & ~wmask);
    end
end

// int_num_features_per_dilation[31:0]
always @(posedge ACLK) begin
    if (ARESET)
        int_num_features_per_dilation[31:0] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_NUM_FEATURES_PER_DILATION_DATA_0)
            int_num_features_per_dilation[31:0] <= (WDATA[31:0] & wmask) | (int_num_features_per_dilation[31:0] & ~wmask);
    end
end

// int_num_features_per_dilation[63:32]
always @(posedge ACLK) begin
    if (ARESET)
        int_num_features_per_dilation[63:32] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_NUM_FEATURES_PER_DILATION_DATA_1)
            int_num_features_per_dilation[63:32] <= (WDATA[31:0] & wmask) | (int_num_features_per_dilation[63:32] & ~wmask);
    end
end

// int_biases[31:0]
always @(posedge ACLK) begin
    if (ARESET)
        int_biases[31:0] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_BIASES_DATA_0)
            int_biases[31:0] <= (WDATA[31:0] & wmask) | (int_biases[31:0] & ~wmask);
    end
end

// int_biases[63:32]
always @(posedge ACLK) begin
    if (ARESET)
        int_biases[63:32] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_BIASES_DATA_1)
            int_biases[63:32] <= (WDATA[31:0] & wmask) | (int_biases[63:32] & ~wmask);
    end
end

// int_time_series_length[31:0]
always @(posedge ACLK) begin
    if (ARESET)
        int_time_series_length[31:0] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_TIME_SERIES_LENGTH_DATA_0)
            int_time_series_length[31:0] <= (WDATA[31:0] & wmask) | (int_time_series_length[31:0] & ~wmask);
    end
end

// int_num_features[31:0]
always @(posedge ACLK) begin
    if (ARESET)
        int_num_features[31:0] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_NUM_FEATURES_DATA_0)
            int_num_features[31:0] <= (WDATA[31:0] & wmask) | (int_num_features[31:0] & ~wmask);
    end
end

// int_num_classes[31:0]
always @(posedge ACLK) begin
    if (ARESET)
        int_num_classes[31:0] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_NUM_CLASSES_DATA_0)
            int_num_classes[31:0] <= (WDATA[31:0] & wmask) | (int_num_classes[31:0] & ~wmask);
    end
end

// int_num_dilations[31:0]
always @(posedge ACLK) begin
    if (ARESET)
        int_num_dilations[31:0] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_NUM_DILATIONS_DATA_0)
            int_num_dilations[31:0] <= (WDATA[31:0] & wmask) | (int_num_dilations[31:0] & ~wmask);
    end
end

//synthesis translate_off
always @(posedge ACLK) begin
    if (ACLK_EN) begin
        if (int_gie & ~int_isr[0] & int_ier[0] & ap_done)
            $display ("// Interrupt Monitor : interrupt for ap_done detected @ \"%0t\"", $time);
        if (int_gie & ~int_isr[1] & int_ier[1] & ap_ready)
            $display ("// Interrupt Monitor : interrupt for ap_ready detected @ \"%0t\"", $time);
    end
end
//synthesis translate_on

//------------------------Memory logic-------------------

endmodule
