import enum
import os
import cv2
import glob
import argparse
import numpy as np
import tensorflow as tf
from trainer import Trainer
from hyperparams import Hyperparams as hp

def get_images(input_path : str) -> list[str]:
    image_paths = []
    if os.path.isfile(input_path):
        image_paths = [input_path]
    elif os.path.isdir(input_path):
        for ext in ['png','jpg','jpeg']:
            image_paths += sorted(glob.glob(os.path.join(input_path,f'*.{ext}')))
    return image_paths

def read_image(image_path : str) -> np.ndarray:
    image = cv2.imread(image_path)
    image = cv2.resize(image, (hp.image_size,hp.image_size))
    image = image / 255
    return image

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--content','-c', type=str,
                        help='Input Content Image or Input Content Images Dir')
    parser.add_argument('--style','-s', type=str,
                        help='Input Style Image or Input Style Images Dir')
    parser.add_argument('--style_weight','-sw', type=float, default = 1.0,
                        help='Style Weight')
    parser.add_argument('--output','-o', type=str, default = r"..\results\outputs",
                        help='Output Dir')

    args = parser.parse_args()

    model = Trainer()

    model.update(hp.__dict__)

    _log_str = "\n"
    params = "{0:25} | {1:25}"
    for k,v in model.__dict__.items():
        _log_str += params.format(k,str(v)) + "\n"
    print(_log_str)

    model.build()

    model.load_ckpts(partial=True)

    content_image_paths = get_images(args.content)
    style_image_paths = get_images(args.style)

    content_list = [read_image(content_path) for content_path in content_image_paths]
    style_list = [read_image(style_path) for style_path in style_image_paths]

    style_cols = np.concatenate([np.zeros((hp.image_size,hp.image_size,3), dtype = np.float32)] + style_list, axis = 1)
    
    _style_content_rows = [style_cols]
    for i,c_img in enumerate(content_list):
        _style_content_cols = [c_img]
        for j,s_img in enumerate(style_list):
            model_inp = [tf.cast(c_img[None,...][...,::-1], dtype = tf.float32), 
                         tf.cast(s_img[None,...][...,::-1], dtype = tf.float32)]
            style_content_image = model.styletransfer(model_inp, training = False, alpha = args.style_weight)[0]
            style_content_image = style_content_image.numpy()[...,::-1]
            _style_content_cols.append(style_content_image)
            print(f"{os.path.basename(content_image_paths[i])} content image styled with {os.path.basename(style_image_paths[i])} style image")
        _style_content_rows.append(np.concatenate(_style_content_cols, axis = 1))

    styled_contents = np.uint8(np.concatenate(_style_content_rows, axis = 0)*255)
    
    output_path = os.path.join(args.output, 'styled_contents.png') if os.path.isdir(args.output) else args.output
    img_saved = cv2.imwrite(output_path, styled_contents)
    if img_saved:
        print(f"Output saved at {output_path}")
        
if __name__ == '__main__':
    main()