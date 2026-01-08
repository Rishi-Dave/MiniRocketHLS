cmake_minimum_required(VERSION 3.0)

set(PROJECT_NAME hydra_inference)
project(${PROJECT_NAME})

find_package(Vitis REQUIRED)

set(EXAMPLE_HLS_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hydra.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hydra_pooling.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/hydra.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test/test_hydra.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test/hydra_hls_testbench_loader.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test/hydra_hls_testbench_loader.h
)

set(EXAMPLE_HLS_SCRIPTS
    ${CMAKE_CURRENT_SOURCE_DIR}/make.tcl.in
)

# Configure TCL script
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/make.tcl.in
               ${CMAKE_CURRENT_BINARY_DIR}/make.tcl @ONLY)

# IP synthesis target
add_custom_target(synthesis.${PROJECT_NAME}
    COMMAND ${VITIS_HLS_BINARY} -f make.tcl -tclargs synthesis
    DEPENDS ${EXAMPLE_HLS_DEPENDS} ${EXAMPLE_HLS_SCRIPTS}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Synthesizing ${PROJECT_NAME} IP"
)

# C simulation target
add_custom_target(csim.${PROJECT_NAME}
    COMMAND ${VITIS_HLS_BINARY} -f make.tcl -tclargs csim
    DEPENDS ${EXAMPLE_HLS_DEPENDS} ${EXAMPLE_HLS_SCRIPTS}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Running C simulation for ${PROJECT_NAME}"
)

# Co-simulation target
add_custom_target(cosim.${PROJECT_NAME}
    COMMAND ${VITIS_HLS_BINARY} -f make.tcl -tclargs cosim
    DEPENDS ${EXAMPLE_HLS_DEPENDS} ${EXAMPLE_HLS_SCRIPTS} synthesis.${PROJECT_NAME}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Running co-simulation for ${PROJECT_NAME}"
)

# IP export target
add_custom_target(ip.${PROJECT_NAME}
    COMMAND ${VITIS_HLS_BINARY} -f make.tcl -tclargs ip
    DEPENDS ${EXAMPLE_HLS_DEPENDS} ${EXAMPLE_HLS_SCRIPTS} synthesis.${PROJECT_NAME}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Exporting ${PROJECT_NAME} IP"
)

# Install IP to repository
set(IP_REPO ${CMAKE_SOURCE_DIR}/build/iprepo)
add_custom_target(installip.${PROJECT_NAME}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${IP_REPO}
    COMMAND ${CMAKE_COMMAND} -E copy
            ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}/solution/impl/export.zip
            ${IP_REPO}/${PROJECT_NAME}.xo
    DEPENDS ip.${PROJECT_NAME}
    COMMENT "Installing ${PROJECT_NAME}.xo to ${IP_REPO}"
)
