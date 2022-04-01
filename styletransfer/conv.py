import tensorflow as tf

class Conv2D(tf.keras.layers.Layer):
    """ Conv2D
    """
    def __init__(self, **kwargs):
        super(Conv2D, self).__init__(name = kwargs.get('name','Conv2D'))
        self.filters = kwargs.get('filters')
        self.kernels = kwargs.get('kernels',3)
        self.strides = kwargs.get('strides',1)
        self.padding_values = tf.constant([[0,0], 
                                            [(self.kernels[0]-1)//2,(self.kernels[0]-1)//2], 
                                            [(self.kernels[1]-1)//2,(self.kernels[1]-1)//2], 
                                            [0,0]])
        self.conv2d = tf.keras.layers.Conv2D(filters = self.filters,
                                             kernel_size = self.kernels, 
                                             strides = self.strides, 
                                             padding = "VALID", 
                                             name = 'Conv2D')

    def call(self, inputs, training = False):
        
        x = inputs
        x = tf.pad(x, self.padding_values, mode = "REFLECT")
        x = self.conv2d(x, training = training)
        
        return x

class SubPixelConv2D(tf.keras.layers.Layer):
    """ SubPixelConv2D
    """
    def __init__(self, **kwargs):
        super(SubPixelConv2D, self).__init__(name = kwargs.get('name','SubPixelConv2D'))
        self.filters = kwargs.get('filters')
        self.r = kwargs.get('r',2)
        self.kernels = kwargs.get('kernels',3)
        self.strides = kwargs.get('strides',1)
        self.padding_values = tf.constant([[0,0], 
                                            [(self.kernels[0]-1)//2,(self.kernels[0]-1)//2], 
                                            [(self.kernels[1]-1)//2,(self.kernels[1]-1)//2], 
                                            [0,0]])
        self.conv2d = tf.keras.layers.Conv2D(filters = self.filters * self.r * self.r, 
                                             kernel_size = self.kernels, 
                                             strides = self.strides, 
                                             padding = "VALID", 
                                             name = 'Conv2D')

    def call(self, inputs, training = False):
        
        x = inputs
        x = tf.pad(x, self.padding_values, mode = "REFLECT")
        x = self.conv2d(x, training = training)
        x = tf.nn.depth_to_space(x, block_size = self.r)
        
        return x