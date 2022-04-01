# Style Transfer

A Tensorflow implementation of **`AdaIn`** and **`AdaConv`**

Style Transfer is the process of transferring the style of one image onto the content of another. Compared to slow iterative optimization process and Fast approximations with feed-forward neural networks, there are two more novel methods like the Adaptive Instance normalization (**`AdaIN`**) layer and Adaptive Convolutions (**`AdaConv`**).

Both **`AdaIN`** and **`AdaConv`** are Encoder-Decoder based models where content and style images features are extracted from pre-trained Encoder model and is used by decoder to generate a style transfered image. For the encoder part in **`AdaIn`** and **`AdaConv`** Style-Transfer models, used **`VGG19`** from **`tf.keras.applications.vgg19`** with default arguements.

## Python Requirements
---

- tqdm
- numpy
- matplotlib
- opencv-python
- tensorflow-addons
- tensorflow-gpu==2.8.0

## Dataset
---

Used [cocodataset](https://cocodataset.org/#download) for content images and [Painter by Numbers](https://www.kaggle.com/competitions/painter-by-numbers/data) for style images.

Download, extract and move the train and test folders from content and style dataset into the folder tree strucuture as shown below 

	StyleTransfer/
	 └─ data/
	     └─ raw/
             ├─ content/
             │   ├─ train2017/
             │   │   ├─ 0.jpg
             │   │   ├─ ...
             │   │   └─ N.jpg
             │   └─ test2017/
             │       ├─ 0.jpg
             │       ├─ ...
             │       └─ N.jpg
             └─ stytle/
                 ├─ train/
                 │   ├─ 0.jpg
                 │   ├─ ...
                 │   └─ N.jpg
                 └─ test/
                     ├─ 0.jpg
                     ├─ ...
                     └─ N.jpg

## Training
---

Modify the `hyperparams.py`'s params *data_path* to point to the correct dataset folder, *logdir* for checkpoints/tensorboard log directory and *model_type* to build model with **`AdaIn`** or **`AdaConv`** for style-transfer.

> **RUN**

	python train.py

Monitor the training using tensorboard with the tensorboard log file under the *`tensorboard`* folder under the log directory 

***Observe whether the content and style loss are converging and styled_content images have content and styles combined.***

Once the training is done, the checkpoints can be found in the *`ckpts`* folder in log directory

	StyleTransfer\
	 └─ results\
	     └─ models\
	    	 ├─ ckpts\
	    	 │	 ├─ checkpoint
	    	 │	 ├─ ckpt-100000.data-00000-of-00001
	    	 │	 └─ ckpt-100000.index
	    	 └─ tensorboard\
	    	 	 └─ events.out.tfevents.x.x.x.x.v2

## Testing

Basic inference testing on content and style image to create a contingency table

> **RUN**

	python test.py -c <Input content image or Input content images dir> -s -c <Input style image or input style images dir> -o <Output image filename or Output dir>

---
---

## AdaIn

[`Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization`](https://arxiv.org/pdf/1703.06868.pdf). An Encoder-Decoder based style-transfer model with **`AdaIn`** layer that aligns the mean and variance of the content features with those of the style features without any trainable parameters.

### AdaIn Style-Transfer Architecture (from original paper)

![AdaIn](/documents/images/AdaIn.png)

### AdaIn Results (Lambda = 10.0)

![AdaIn_Results_L10.0](/documents/images/AdaIn_Results_L10.0.png)

### Models
---


| Model | Image Size | Encoder | Lambda | # iter | Sample Output |
| --- | --- | --- | --- | --- | --- |
| [AdaIn #1](https://drive.google.com/drive/folders/13BZgMevRkuOPuJQbNjZ3qk8WMk0ePcV_?usp=sharing) | 256 x 256 | VGG19 |  4.0 | 100K | [Link](https://drive.google.com/file/d/13ug0zkmmaothmOMaNRAZGAZ1Zx37cea-/view?usp=sharing) |
| [AdaIn #2](https://drive.google.com/drive/folders/1pXqJkG40sop4a0K_uTNNCBbIYloh4z-S?usp=sharing) | 256 x 256 | VGG19 | 10.0 | 100K | [Link](https://drive.google.com/file/d/1IPFPg_XoBacjfoaTVT7LMvLQHTng31Mb/view?usp=sharing) |
| [AdaIn #3](https://drive.google.com/drive/folders/1j5YZj4FrfZWvKhRv5bb9JqXLJZzcvOnG?usp=sharing) | 256 x 256 | VGG19 | 50.0 | 100K | [Link](https://drive.google.com/file/d/1vFkK_pPxay8N802av1dxz8nb-gkV4aa3/view?usp=sharing) |

---
---

## AdaConv (WIP)

[`Adaptive Convolutions for Structure-Aware Style Transfer`](https://studios.disneyresearch.com/app/uploads/2021/04/Adaptive-Convolutions-for-Structure-Aware-Style-Transfer.pdf). An Encoder-Decoder based style-transfer model with **`AdaConv`** at it's core to allow for the simultaneous transfer of both statistical and structural information from style images to content images.

### AdaConv Style-Transfer Architecture (from original paper)

![AdaConv](/documents/images/AdaConv.png)
