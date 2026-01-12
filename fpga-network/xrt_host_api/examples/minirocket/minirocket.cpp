// Copyright (C) 2022 Xilinx, Inc
// SPDX-License-Identifier: BSD-3-Clause

#include <chrono>
#include <experimental/xrt_ip.h>
#include <filesystem>
#include <json/json.h>
#include <limits.h>
#include <map>
#include <string>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#include <iostream>
#include <limits>

#include <vnx/cmac.hpp>
#include <vnx/networklayer.hpp>
#include <vnx/minirocket_loader.h>

#include <xrt/xrt_device.h>
#include "xrt/xrt_bo.h"
#include "xrt/xrt_kernel.h"

template <typename T>
struct aligned_allocator {
    using value_type = T;

    aligned_allocator() {}

    aligned_allocator(const aligned_allocator&) {}

    template <typename U>
    aligned_allocator(const aligned_allocator<U>&) {}

    T* allocate(std::size_t num) {
        void* ptr = nullptr;

#if defined(_WINDOWS)
        {
            ptr = _aligned_malloc(num * sizeof(T), 4096);
            if (ptr == nullptr) {
                std::cout << "Failed to allocate memory" << std::endl;
                exit(EXIT_FAILURE);
            }
        }
#else
        {
            if (posix_memalign(&ptr, 4096, num * sizeof(T))) throw std::bad_alloc();
        }
#endif
        return reinterpret_cast<T*>(ptr);
    }
    void deallocate(T* p, std::size_t num) {
#if defined(_WINDOWS)
        _aligned_free(p);
#else
        free(p);
#endif
    }
};


namespace fs = std::filesystem;

/***************
 *    CONFIG   *
 ***************/

typedef struct
{
  const char *  hostname;
  uint32_t      board_id;
  const char *  ip_address[2];
} ip_config_table_t;

#ifndef ARRAY_SIZE
#define ARRAY_SIZE(arr) sizeof(arr)/sizeof((arr)[0])
#endif



ip_config_table_t  ip_lut[] = {
  {
    .hostname       = "pc151.pyuvaraj-286064.octfpga-pg0.cloudlab.umass.edu",
    .board_id       =  0,
    .ip_address     = {"192.168.40.29","10.1.212.102"},
  },
  {
    .hostname       = "hostname2",
    .board_id       =  0,
   .ip_address      = {"10.1.212.103","10.1.212.104"},
  },
  {
    .hostname       = "hostname3",
    .board_id       =  0,
    .ip_address     = {"10.1.212.101","10.1.212.102"},
  },
};


ip_config_table_t * get_ip_config(std::string hostname, uint32_t board_id)
{
    ip_config_table_t * default_ip_config = &ip_lut[0]; // will set default to the first matched hostname.
    for (uint32_t  i = 0; i < ARRAY_SIZE(ip_lut); i++){
        if(std::string(ip_lut[i].hostname) ==  hostname){
            if(ip_lut[i].board_id == board_id){
                printf("Found the ip configure for host %s\n",  ip_lut[i].hostname);
                return &ip_lut[i];
            }
        }
    }
    printf("Host %s with board %d does not exist in configure\n",  hostname.c_str(), board_id);
    printf("Use default config: %s, if0 ip: %s, if1 ip: %s\n",
                              default_ip_config->hostname,
                              default_ip_config->ip_address[0],
                              default_ip_config->ip_address[1] );

    return default_ip_config;
}



enum xclbin_types { if0, if1, if3, minirocket };

struct xclbin_path {
  std::string path;
  xclbin_types type;
};

/***************
 * END CONFIG  *
 ***************/

xclbin_path parse_xclbin(const std::string &platform, const char *arg) {
  // Determine content of xclbin based on filename and platform.
  xclbin_path xclbin;
  xclbin.path = arg;
  std::string filename = fs::path(arg).filename();

  bool found = false;
  xclbin.type = minirocket;

  return xclbin;
}

Json::Value parse_json(const std::string &string) {
  Json::Reader reader;
  Json::Value json;
  reader.parse(string, json);
  return json;
}

int main(int argc, char *argv[]) {
  // Retrieve host and device information
  char hostname[HOST_NAME_MAX];
  gethostname(hostname, HOST_NAME_MAX);
  int device_id = 0;
  std::size_t ip_index = 0;

  // Read xclbin files from commandline
  std::vector<const char *> args(argv + 1, argv + argc);

  if (args.size() < 1) {
    std::cerr << "No xclbin provided" << std::endl;
    return 1;
  }

  if (args.size() >= 2) {
    device_id = std::stoi(args[1]);
    std::cerr << "Loading XRT device " << device_id << std::endl;
  }


  std::string model_file = argv[3];
  std::string test_file = argv[4];


  xrt::device device = xrt::device(device_id);
  // Collect platform info from xclbin
  const std::string platform_json = device.get_info<xrt::info::device::platform>();
  const Json::Value platform_dict = parse_json(platform_json);
  const std::string platform = platform_dict["platforms"][0]["static_region"]["vbnv"].asString();
  std::cout << "FPGA platform: " << platform << std::endl;

  const xclbin_path xclbin = parse_xclbin(platform, args[0]);
  auto xclbin_uuid = device.load_xclbin(xclbin.path);
  std::cout << "Loaded " << xclbin.path << " onto FPGA on " << hostname << std::endl;
  // Give time for xclbin to be loaded completely before attempting to read
  // the link status.

  std::cout << "Press Enter to continue...";
  std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');  


  std::this_thread::sleep_for(std::chrono::seconds(1));

  // Loop over compute units in xclbin

  const char * CMAC = "cmac_0";
  const char * NETLAYER = "networklayer_0";
  const char * MINIROCKET = "minirocket_inference_1";

  auto cmac = vnx::CMAC(xrt::ip(device, xclbin_uuid, std::string(CMAC) + ":{" + std::string(CMAC) + "}"));
  // Enable rsfec if necessary
  cmac.set_rs_fec(false);
  
  auto nl_ip = xrt::ip(device, xclbin_uuid, "networklayer:{" + std::string(NETLAYER) + "}");
  auto minirocket_ip = xrt::kernel(device, xclbin_uuid, "minirocket_inference:{minirocket_inference_1}");

  auto networklayer = vnx::Networklayer(nl_ip);
  bool link_status;

  // Can take a few tries before link is ready.
  for (std::size_t i = 0; i < 5; ++i) {
    auto status = cmac.link_status();
    link_status = status["rx_status"];
    if (link_status) {
      break;
    }
    std::this_thread::sleep_for(std::chrono::seconds(1));
  }


  std::cout << "Link interface " << CMAC << ": "<< (link_status ? "true" : "false") << std::endl;
  std::cout << "RS-FEC enabled: " << (cmac.get_rs_fec() ? "true" : "false") << std::endl;


  std::string ip = get_ip_config(hostname, device_id)->ip_address[ip_index];

  std::cout << "setting up IP " << ip << " to interface " << CMAC << std::endl;
  networklayer.update_ip_address(ip);
  std::this_thread::sleep_for(std::chrono::seconds(1));

  nl_ip.write_register(vnx::udp_number_sockets, 16);
  auto udp_debug_reg = nl_ip.read_register(vnx::udp_number_sockets);
  std::cout << "# Sockets " << udp_debug_reg << std::endl; 

  networklayer.configure_socket(0, "192.168.40.31", 5005, 6006, true);
  networklayer.configure_socket(1, "192.168.40.31", 5005, 6006, true);
  networklayer.populate_socket_table();
  networklayer.print_socket_table(2);
  networklayer.arp_discovery();
  std::this_thread::sleep_for(std::chrono::seconds(5));
  auto table = networklayer.read_arp_table(255);
  for (const auto& [id, value] : table){
    std::cout << "arp table: [" << id << "] = " << value.first << "  " <<value.second << "; "<< std::endl;
  }


  //setup and run MiniRocket

  std::cout << "Loading MiniRocket model and test data" << std::endl;
  std::cout << "Model file: " << model_file << std::endl;
  std::cout << "Test file: " << test_file << std::endl;


  // Initialize loader
  MiniRocketTestbenchLoader loader;
  
  // HLS arrays (heap allocated for testbench to avoid stack overflow)
  data_t (*coefficients)[MAX_FEATURES] = new data_t[MAX_CLASSES][MAX_FEATURES];

  std::vector<data_t, aligned_allocator<data_t>> time_series_input(MAX_TIME_SERIES_LENGTH);
  std::vector<data_t, aligned_allocator<data_t>> prediction_output(MAX_CLASSES);
  std::vector<data_t, aligned_allocator<data_t>> flattened_coefficients(MAX_CLASSES * MAX_FEATURES);

  std::vector<data_t, aligned_allocator<data_t>> intercept(MAX_CLASSES);
  std::vector<data_t, aligned_allocator<data_t>> scaler_mean(MAX_FEATURES);
  std::vector<data_t, aligned_allocator<data_t>> scaler_scale(MAX_FEATURES);
  std::vector<data_t, aligned_allocator<data_t>> biases(MAX_FEATURES);
  std::vector<int_t, aligned_allocator<int_t>> dilations(MAX_DILATIONS);
  std::vector<int_t, aligned_allocator<int_t>> num_features_per_dilation(MAX_DILATIONS);
  int_t num_dilations, num_features, num_classes, time_series_length;
  
  // Load model into HLS arrays
  std::cout << "Loading model..." << std::endl;
  if (!loader.load_model_to_hls_arrays(model_file, coefficients, intercept.data(), 
                                      scaler_mean.data(), scaler_scale.data(), dilations.data(),
                                      num_features_per_dilation.data(), biases.data(),
                                      num_dilations, num_features, num_classes,
                                      time_series_length)) {
      std::cerr << "Failed to load model!" << std::endl;
      return 1;
  }

  time_series_input.resize(time_series_length);
  prediction_output.resize(num_classes);

  for (int i = 0; i < num_classes * num_features; i++) {
      int row = i / num_features;
      int col = i % num_features;
      flattened_coefficients[i] = coefficients[row][col];
  }
  
  // Load test data
  std::cout << "Loading test data..." << std::endl;
  std::vector<std::vector<float>> test_inputs, expected_outputs;
  if (!loader.load_test_data(test_file, test_inputs, expected_outputs)) {
      std::cerr << "Failed to load test data!" << std::endl;
      return 1;
  }
  


  std::cout << "Allocate Buffer in Global Memory\n";
  auto flattened_coefficients_b02 = xrt::bo(device, sizeof(data_t) * MAX_CLASSES * MAX_FEATURES, minirocket_ip.group_id(2));
  auto intercept_b03 = xrt::bo(device, sizeof(data_t) * MAX_CLASSES, minirocket_ip.group_id(3));
  auto scaler_mean_b04 = xrt::bo(device, sizeof(data_t) * MAX_FEATURES, minirocket_ip.group_id(4));
  auto scaler_scale_b05 = xrt::bo(device, sizeof(data_t) * MAX_FEATURES, minirocket_ip.group_id(5));
  auto dilations_b06 = xrt::bo(device, sizeof(int_t) * MAX_DILATIONS, minirocket_ip.group_id(6));
  auto num_features_per_dilation_b07 = xrt::bo(device, sizeof(int_t) * MAX_DILATIONS, minirocket_ip.group_id(7));
  auto biases_b08 = xrt::bo(device, sizeof(data_t) * MAX_FEATURES, minirocket_ip.group_id(8));


  auto flattened_coefficients_b02_map = flattened_coefficients_b02.map<data_t*>();
  auto intercept_b03_map = intercept_b03.map<data_t*>();
  auto scaler_mean_b04_map = scaler_mean_b04.map<data_t*>();
  auto scaler_scale_b05_map = scaler_scale_b05.map<data_t*>();
  auto dilations_b06_map = dilations_b06.map<data_t*>();
  auto num_features_per_dilation_b07_map = num_features_per_dilation_b07.map<data_t*>();
  auto biases_b08_map = biases_b08.map<data_t*>(); 

  for (int i = 0; i < MAX_CLASSES * MAX_FEATURES; i++) {
    flattened_coefficients_b02_map[i] = flattened_coefficients[i];
  }
  for (int i = 0 ; i < MAX_CLASSES; i++) {
    intercept_b03_map[i] = intercept[i];
  }
  for (int i = 0; i < MAX_FEATURES; i++) {
    scaler_mean_b04_map[i] = scaler_mean[i];
    scaler_scale_b05_map[i] = scaler_scale[i];
    biases_b08_map[i] = biases[i];
  }
  for (int i = 0; i < MAX_DILATIONS; i++) {
    dilations_b06_map[i] = dilations[i];
    num_features_per_dilation_b07_map[i] = num_features_per_dilation[i];
  }

  std::cout << "synchronize input buffer data to device global memory\n";
  flattened_coefficients_b02.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  intercept_b03.sync(XCL_BO_SYNC_BO_TO_DEVICE); 
  scaler_mean_b04.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  scaler_scale_b05.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  dilations_b06.sync(XCL_BO_SYNC_BO_TO_DEVICE); 
  num_features_per_dilation_b07.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  biases_b08.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  std::cout << "Start MiniRocket server\n";
  auto run = xrt::run(minirocket_ip);
  run.set_arg(2, flattened_coefficients_b02);
  run.set_arg(3, intercept_b03);
  run.set_arg(4, scaler_mean_b04);
  run.set_arg(5, scaler_scale_b05);
  run.set_arg(6, dilations_b06);
  run.set_arg(7, num_features_per_dilation_b07);
  run.set_arg(8, biases_b08);
  run.set_arg(9, time_series_length);
  run.set_arg(10, num_features);
  run.set_arg(11, num_classes);
  run.set_arg(12, num_dilations);
  run.set_arg(13, 0);
  run.start();
  run.wait();

  std::cout << "Press Enter to continue...";
  std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

  std::cout << "Start of debug information dump" << std::endl;
  networklayer.get_icmp_in_pkts();
  networklayer.get_icmp_out_pkts();
  networklayer.get_udp_in_pkts();
  networklayer.get_udp_out_pkts();

  std::cout << "End of debug information dump" << std::endl;
  std::cout << "Success!" << std::endl;


  return 0;
}
