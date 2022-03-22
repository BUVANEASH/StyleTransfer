# AdaConv (WIP)

A Tensorflow implementation of [`Adaptive Convolutions for Structure-Aware Style Transfer`](https://studios.disneyresearch.com/app/uploads/2021/04/Adaptive-Convolutions-for-Structure-Aware-Style-Transfer.pdf). An Encoder-Decoder based style-transfer model with **`AdaConv`** at it's core to allow for the simultaneous transfer of both statistical and structural information from style images to content images.

### AdaConv Style-Transfer Architecture (from original paper)

![AdaConv](/documents/images/AdaConv.png)

### AdaConv Style-Transfer Encoder

For the encoder part in AdaConv Style-Transfer, used **`VGG19`** from **`tf.keras.applications.vgg19`** with default arguements.

## Python Requirements
---

- tqdm
- numpy
- matplotlib
- opencv-python
- tensorflow-addons
- tensorflow-gpu==2.8.0