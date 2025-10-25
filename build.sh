#!/bin/bash

cd /home/drake
cmake -S . -B build
cmake --build build --config Release --target install -- -j 2
