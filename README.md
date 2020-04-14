# Quantization Test
Sanity checking mobilenetv3 quantization

## Issue

After using `post_training_quantization.py` to quantize the floating-point MobileNetV3 checkpoint, the resulting `tflite` model
performs very poorly (around 0% top-1 accuracy).

## Workflow

Tensorflow version: 1.15.2

Currently, I've been trying to quantize the pretrained floating-point MobileNetV3 checkpoint found in the folder `v3-large_224_1.0_float`.
I use the script `quantize_net.sh` to call the post-training quantization code (taken directly from Tensorflow Slim). My
quantized `.tflite` models can be found named `model_quantized_fixed.tflite` in each folder respectively. I can only successfully quantize
the minimalistic model, which leads me to believe the issue might be with the advanced layers.

Then, I evaluate the model performance using [Tensorflow's ImageNet accuracy tool](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/accuracy/ilsvrc).
