#!/bin/bash
rm -r build
echo "**************************************************************"
echo "Building int quantization kernels"
echo "**************************************************************"
python build_int_quantization.py install
echo "Done"
echo "**************************************************************"

