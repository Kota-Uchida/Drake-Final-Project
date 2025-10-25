#!/bin/bash

cd /root/workspace
cmake -S . -B build
cmake --build build --config Release --target install -- -j 2
