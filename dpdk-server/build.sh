#!/bin/bash

mkdir build
/usr/bin/cmake -S . -B build
cd build && make