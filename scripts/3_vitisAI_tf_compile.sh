#!/bin/bash

# Compile

vai_c_tensorflow --arch /workspace/ministNumber/dnndk/dnndk/dpu.json  -f quantize_results/deploy_model.pb --output_dir compile_result -n yolo_car


echo "#####################################"
echo "COMPILATION COMPLETED"
echo "#####################################"
