import os
import glob
import tensorflow as tf

class DataLoad():
    """
    Dataset loader class
    """
    def __init__(self):
        pass
    
    def get_dataset(self):

        self.content_train_list = sorted(glob.glob(os.path.join(self.raw_data,'content/train2017/*.jpg')))
        self.content_test_list = sorted(glob.glob(os.path.join(self.raw_data,'content/test2017/*.jpg')))

        self.style_train_list = sorted(glob.glob(os.path.join(self.raw_data,'style/train/*.jpg')))
        self.style_test_list = sorted(glob.glob(os.path.join(self.raw_data,'style/test/*.jpg')))

        _content_train_len = len(self.content_train_list)
        _content_test_len = len(self.content_test_list)
        _style_train_len = len(self.style_train_list)
        _style_test_len = len(self.style_test_list)

        print(f"Train ---> Content {_content_train_len} | Style {_style_train_len} || Test ---> Content {_content_test_len} | Style {_style_test_len}")

    def map_fn(self, image_path: str) -> tuple[tf.Tensor]:
        """
        Args:
            image_path: The RGB image path.

        Returns:
            The augmented image
        """
        # Read Content and Style
        image = tf.io.decode_jpeg(tf.io.read_file(image_path), channels = 3)

        if tf.random.uniform((), 0, 1) > self.random_crop_prob:
            # Resize
            image = tf.image.resize(image,(self.resize_size,self.resize_size), method = 'bilinear')
            # RandomCrop
            image = tf.image.random_crop(image, size = [self.image_size,self.image_size,3])
        else:
            # Resize
            image = tf.image.resize(image,(self.image_size,self.image_size), method = 'bilinear')
        
        image = tf.cast(tf.clip_by_value(image / 255, clip_value_min = 0, clip_value_max = 1), dtype = tf.float32)

        return image
    
    def create_dataloader(self, image_set: list[str]) -> tf.data.Dataset:
        """
        Args:
            image_set: The RGB image paths set
            
        Yield:
            The image set dataloader
        """
        def generator():
            for image_path in image_set: 
                yield str(image_path)

        dataset = tf.data.Dataset.from_generator(generator,
                                                 output_signature=tf.TensorSpec(shape=(), dtype=tf.string))
        dataset = dataset.map(map_func = lambda img_path: self.map_fn(img_path), 
                              num_parallel_calls = tf.data.AUTOTUNE,
                              deterministic = False)
        dataset = dataset.apply(tf.data.experimental.ignore_errors())
        dataset = dataset.shuffle(buffer_size = 1000)
        dataset = dataset.repeat()
        dataset = dataset.batch(self.batch_size)
        
        return dataset