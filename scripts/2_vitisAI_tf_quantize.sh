#!/bin/bash


# run quantization

vai_q_tensorflow quantize \
  --input_frozen_graph ./model_data/model.pb \
  --input_nodes input_1 \
  --input_shapes ?,320,320,3 \
  --output_nodes conv2d_20/BiasAdd,conv2d_23/BiasAdd \
  --method 1 \
  --input_fn input_fn.calib_input \
  --gpu 0 \
  --calib_iter 100 \

echo "#####################################"
echo "QUANTIZATION COMPLETED"
echo "#####################################"

