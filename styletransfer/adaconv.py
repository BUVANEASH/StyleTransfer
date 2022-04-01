import tensorflow as tf
from conv import Conv2D

class GlobalStyleEncoder(tf.keras.layers.Layer):
    """ GlobalStyleEncoder
    """
    def __init__(self, **kwargs):
        super(GlobalStyleEncoder, self).__init__(name = kwargs.get('name','GlobalStyleEncoder'))
        self.Sd = kwargs.get('Sd',512)
        self.Sh = kwargs.get('Sh',3)
        self.Sw = kwargs.get('Sw',3)
        self.kernels = (self.Sh, self.Sw)
        self.padding_values = tf.constant([[0,0], 
                                           [(self.kernels[0]-1)//2,(self.kernels[0]-1)//2], 
                                           [(self.kernels[1]-1)//2,(self.kernels[1]-1)//2], 
                                           [0,0]])
        self.conv2d_1 = tf.keras.layers.Conv2D(filters = self.Sd,
                                               kernel_size = self.kernels, 
                                               strides=1, 
                                               padding="VALID", 
                                               name = 'Conv2D_1')
        self.conv2d_2 = tf.keras.layers.Conv2D(filters = self.Sd,
                                               kernel_size = self.kernels, 
                                               strides=1, 
                                               padding="VALID", 
                                               name = 'Conv2D_2')
        self.conv2d_3 = tf.keras.layers.Conv2D(filters = self.Sd,
                                               kernel_size = self.kernels, 
                                               strides=1, 
                                               padding="VALID", 
                                               name = 'Conv2D_3')
        self.fc = tf.keras.layers.Dense(units = self.Sh * self.Sw * self.Sd, 
                                        name = 'FullyConnected')

    def call(self, inputs, training = False):
        
        x = inputs
        
        x = tf.pad(x, self.padding_values, mode = "REFLECT")
        x = self.conv2d_1(x, training = training)
        x = tf.nn.leaky_relu(x)
        x = tf.nn.avg_pool2d(x, ksize = 2, strides = 2, padding = 'VALID')
        
        x = tf.pad(x, self.padding_values, mode = "REFLECT")
        x = self.conv2d_2(x, training = training)
        x = tf.nn.leaky_relu(x)
        x = tf.nn.avg_pool2d(x, ksize = 2, strides = 2, padding = 'VALID')
        
        x = tf.pad(x, self.padding_values, mode = "REFLECT")
        x = self.conv2d_3(x, training = training)
        x = tf.nn.leaky_relu(x)
        x = tf.nn.avg_pool2d(x, ksize = 2, strides = 2, padding = 'VALID')
        
        _, _h, _w, _c = x.shape.as_list()
        x = tf.reshape(x, shape = [-1, _h*_w*_c])
        
        x = self.fc(x, training = training)
        x = tf.reshape(x, shape = [-1, self.Sh, self.Sw, self.Sd])
        
        return x

class KernelPredictor(tf.keras.layers.Layer):
    """ Kernel Predictor
    """
    def __init__(self, **kwargs):
        super(KernelPredictor, self).__init__(name = kwargs.get('name','KernelPredictor'))
        self.Cin = kwargs.get('Cin')
        self.Cout = kwargs.get('Cout')
        self.Ng = kwargs.get('Ng')
        self.Kh  = kwargs.get('Kh')
        self.Kw = kwargs.get('Kw')
        self.kernels = (self.Kh, self.Kw)
        self.padding_values = tf.constant([[0,0], 
                                            [(self.kernels[0]-1)//2,(self.kernels[0]-1)//2], 
                                            [(self.kernels[1]-1)//2,(self.kernels[1]-1)//2], 
                                            [0,0]])

        self.depthwise_conv_kernel_predictor = tf.keras.layers.Conv2D(filters = (self.Cin//self.Ng)*self.Cout, 
                                                                      kernel_size = self.kernels, 
                                                                      padding = "VALID", 
                                                                      strides = 1,
                                                                      name = 'DepthwiseConv2D_Kernel_Predcitor')
        self.pointwise_conv_kernel_predictor = tf.keras.layers.Conv2D(filters = (self.Cout//self.Ng)*self.Cout, 
                                                                      kernel_size = (1, 1), 
                                                                      strides = 1,
                                                                      padding = "VALID", 
                                                                      name = 'PointwiseConv2D_Kernel_Predcitor')
        self.bias_predictor = tf.keras.layers.Conv2D(filters = self.Cout, 
                                                     kernel_size = (1, 1), 
                                                     strides = 1,
                                                     padding = "VALID", 
                                                     name = 'Bias_Predcitor')

    def call(self, inputs, training = False):
        
        W = inputs
        assert W.shape.as_list()[1] == self.Kh and W.shape.as_list()[1] == self.Kw
        
        pW = tf.pad(W, self.padding_values, mode = "REFLECT")
        depthwise_kernels = self.depthwise_conv_kernel_predictor(pW, training = training)
        depthwise_kernels = tf.reshape(depthwise_kernels, [-1, self.Kh, self.Kw, (self.Cin//self.Ng), self.Cout])
        
        _W = tf.nn.avg_pool2d(W, ksize = 2, strides = 2, padding = 'VALID')

        pointwise_kernels = self.pointwise_conv_kernel_predictor(_W, training = training)
        pointwise_kernels = tf.reshape(pointwise_kernels, [-1, 1, 1, (self.Cout//self.Ng), self.Cout])
        
        biases = self.bias_predictor(_W, training = training)
        biases = tf.reshape(biases, [-1, self.Cout])
        
        return depthwise_kernels, pointwise_kernels, biases
        
class AdaConv(tf.keras.layers.Layer):
    """ Adaptive Convolution
    """
    def __init__(self, **kwargs):
        super(AdaConv, self).__init__(name = kwargs.get('name','AdaConv'))
        self.channels = kwargs.get('channels')
        self.kernels = kwargs.get('kernels')
        self.epsilon = 1e-5
        self.reduction_axes = [1,2]
        self.padding_values = tf.constant([[0,0], 
                                            [(self.kernels[0]-1)//2,(self.kernels[0]-1)//2], 
                                            [(self.kernels[1]-1)//2,(self.kernels[1]-1)//2], 
                                            [0,0]])
        self.conv2d = Conv2D(filters = self.channels, kernels = self.kernels, strides = 1, name = 'Conv2D')

    def _instance_normalization(self, x):
        x = tf.cast(x, dtype = tf.float32)
        mean, variance = tf.nn.moments(x, axes = self.reduction_axes, keepdims=True)
        standard_deviation = tf.sqrt(variance + self.epsilon)
        x = (x - mean) / standard_deviation
        return x

    def _single_depthwise_separable_conv2d(self, inputs):
        x, depthwise_kernel, pointwise_kernel, bias = inputs
        _x_dtype = x.dtype
        x = tf.cast(x, dtype = tf.float32)
        depthwise_kernel = tf.cast(depthwise_kernel, dtype = tf.float32)
        pointwise_kernel = tf.cast(pointwise_kernel, dtype = tf.float32)
        bias = tf.cast(bias, dtype = tf.float32)
        x = tf.expand_dims(x, axis = 0)
        x = tf.pad(x, self.padding_values, mode = "REFLECT")
        x = tf.nn.conv2d(x, depthwise_kernel, strides = 1, padding = "VALID")
        x = tf.nn.conv2d(x, pointwise_kernel, strides = 1, padding = "VALID")
        x = tf.nn.bias_add(x, bias)
        x = tf.squeeze(x, axis = 0)
        x = tf.cast(x, dtype = _x_dtype)
        return x

    def call(self, inputs, training = False):
        
        x, depthwise_kernels, pointwise_kernels, biases = inputs
        
        x = self._instance_normalization(x)

        _shape = [x.shape.as_list()[1],x.shape.as_list()[2],biases.shape.as_list()[-1]]
        x = tf.map_fn(self._single_depthwise_separable_conv2d, 
                      elems = (x, depthwise_kernels, pointwise_kernels, biases),
                      fn_output_signature = tf.TensorSpec(shape = _shape, dtype = x.dtype))

        x = self.conv2d(inputs = x, training = training)

        return x