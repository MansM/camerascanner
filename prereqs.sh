#!/bin/bash

curl http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2017_11_17.tar.gz |tar xvz
curl https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/ssd_mobilenet_v1_coco_2017_11_17.pbtxt \
      -o ssd_mobilenet_v1_coco_2017_11_17/graph.pbtxt

[[ -d models ]] || git clone --depth 1 https://github.com/tensorflow/models
cd models/research
protoc object_detection/protos/*.proto --python_out=.
python -m pip install .
cd ../..