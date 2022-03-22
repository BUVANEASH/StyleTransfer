import tensorflow as tf

class Block(tf.keras.layers.Layer):
    """ Block
    """
    def __init__(self, name = 'Block', **kwargs):
        super(Block, self).__init__(name = name, **kwargs)

    def call(self, inputs, training = False):
        x = inputs
        return x

class Conv2D(tf.keras.layers.Layer):
    """ Conv2D
    """
    def __init__(self, filters, kernels = (3, 3), strides = 1, name = 'Conv2D', **kwargs):
        super(Conv2D, self).__init__(name = name, **kwargs)
        self.kernels = kernels
        self.padding_values = tf.constant([[0,0], 
                                            [(self.kernels[0]-1)//2,(self.kernels[0]-1)//2], 
                                            [(self.kernels[1]-1)//2,(self.kernels[1]-1)//2], 
                                            [0,0]])
        self.conv2d = tf.keras.layers.Conv2D(filters, self.kernels, padding="VALID", strides=strides, name = 'Conv2D')

    def call(self, inputs, training = False):
        x = inputs
        x = tf.pad(x, self.padding_values, mode = "REFLECT")
        x = self.conv2d(x, training = training)
        return x

class SubPixelConv2D(tf.keras.layers.Layer):
    """ SubPixelConv2D
    """
    def __init__(self, filters, r = 2, kernels = (3, 3), strides = 1, name = 'SubPixelConv2D', **kwargs):
        super(SubPixelConv2D, self).__init__(name = name, **kwargs)
        self.r = r
        self.kernels = kernels
        self.padding_values = tf.constant([[0,0], 
                                            [(self.kernels[0]-1)//2,(self.kernels[0]-1)//2], 
                                            [(self.kernels[1]-1)//2,(self.kernels[1]-1)//2], 
                                            [0,0]])
        self.conv2d = tf.keras.layers.Conv2D(filters*self.r*self.r, self.kernels, padding="VALID", strides=strides, name = 'Conv2D')

    def call(self, inputs, training = False):
        x = inputs
        x = tf.pad(x, self.padding_values, mode = "REFLECT")
        x = self.conv2d(x, training = training)
        x = tf.nn.depth_to_space(x, block_size = self.r)
        return x

class GlobalStyleEncoder(tf.keras.layers.Layer):
    """ GlobalStyleEncoder
    """
    def __init__(self, Sd = 512 , Sh = 3, Sw = 3, name = 'GlobalStyleEncoder', **kwargs):
        super(GlobalStyleEncoder, self).__init__(name = name, **kwargs)
        self.Sd = Sd
        self.Sh = Sh
        self.Sw = Sw
        self.kernels = (Sh, Sw)
        self.padding_values = tf.constant([[0,0], 
                                           [(self.kernels[0]-1)//2,(self.kernels[0]-1)//2], 
                                           [(self.kernels[1]-1)//2,(self.kernels[1]-1)//2], 
                                           [0,0]])
        self.conv2d_1 = tf.keras.layers.Conv2D(self.Sd, self.kernels, padding="VALID", strides=1, name = 'Conv2D_1')
        self.conv2d_2 = tf.keras.layers.Conv2D(self.Sd, self.kernels, padding="VALID", strides=1, name = 'Conv2D_2')
        self.conv2d_3 = tf.keras.layers.Conv2D(self.Sd, self.kernels, padding="VALID", strides=1, name = 'Conv2D_3')
        self.fc = tf.keras.layers.Dense(self.Sh*self.Sw*self.Sd, name = 'FullyConnected')

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
    def __init__(self, Cin, Cout, Ng, Kh, Kw, name = 'KernelPredictor', **kwargs):
        super(KernelPredictor, self).__init__(name = name, **kwargs)
        self.Cin = Cin
        self.Cout = Cout
        self.Ng = Ng
        self.Kh  = Kh
        self.Kw = Kw
        self.kernels = (Kh, Kw)
        self.padding_values = tf.constant([[0,0], 
                                            [(self.kernels[0]-1)//2,(self.kernels[0]-1)//2], 
                                            [(self.kernels[1]-1)//2,(self.kernels[1]-1)//2], 
                                            [0,0]])

        self.depthwise_conv_kernel_predictor = tf.keras.layers.Conv2D((self.Cin//self.Ng)*self.Cout, 
                                                                       self.kernels, 
                                                                       padding="VALID", 
                                                                       strides=1,
                                                                       name = 'DepthwiseConv2D_Kernel_Predcitor')
        self.pointwise_conv_kernel_predictor = tf.keras.layers.Conv1D((self.Cout//self.Ng)*self.Cout, 
                                                                       kernel_size=1, 
                                                                       padding="valid", 
                                                                       strides=1,
                                                                       name = 'PointwiseConv2D_Kernel_Predcitor')
        self.bias_predictor = tf.keras.layers.Conv1D(self.Cout, 
                                                     kernel_size=1, 
                                                     padding="valid", 
                                                     strides=1,
                                                     name = 'Bias_Predcitor')

    def call(self, inputs, training = False):
        W = inputs
        assert W.shape.as_list()[1] == self.Kh and W.shape.as_list()[1] == self.Kw
        
        pW = tf.pad(W, self.padding_values, mode = "REFLECT")
        depthwise_kernels = self.depthwise_conv_kernel_predictor(pW, training = training)
        depthwise_kernels = tf.reshape(depthwise_kernels, [-1, self.Kh, self.Kw, (self.Cin//self.Ng), self.Cout])
        
        _W = tf.nn.avg_pool2d(W, ksize = 2, strides = 2, padding = 'VALID')
        _W = tf.squeeze(_W, axis = 1)

        pointwise_kernels = self.pointwise_conv_kernel_predictor(_W, training = training)
        pointwise_kernels = tf.reshape(pointwise_kernels, [-1, 1, 1, (self.Cout//self.Ng), self.Cout])
        
        biases = self.bias_predictor(_W, training = training)
        biases = tf.reshape(biases, [-1, self.Cout])
        
        return depthwise_kernels, pointwise_kernels, biases
        
class AdaConv(tf.keras.layers.Layer):
    """ Adaptive Convolution
    """
    def __init__(self, channels, kernels = (3,3), name = 'AdaConv', **kwargs):
        super(AdaConv, self).__init__(name = name, **kwargs)
        self.channels = channels
        self.kernels = kernels
        self.padding_values = tf.constant([[0,0], 
                                            [(self.kernels[0]-1)//2,(self.kernels[0]-1)//2], 
                                            [(self.kernels[1]-1)//2,(self.kernels[1]-1)//2], 
                                            [0,0]])
        self.conv2d = Conv2D(channels, kernels = self.kernels, strides = 1, name = 'Conv2D')

    def _instance_normalization(self, x):
        mean, variance = tf.nn.moments(x, [1,2], keepdims=True)
        x = (x - mean) * tf.math.rsqrt(variance + 1e-3)
        return x

    def _single_depthwise_separable_conv2d(self, inputs):
        x, depthwise_kernel, pointwise_kernel, bias = inputs
        x = tf.expand_dims(x, axis = 0)
        x = tf.pad(x, self.padding_values, mode = "REFLECT")
        x = tf.nn.conv2d(x, depthwise_kernel, strides = 1, padding = "VALID")
        x = tf.nn.conv2d(x, pointwise_kernel, strides = 1, padding = "VALID")
        x = tf.nn.bias_add(x, bias)
        x = tf.squeeze(x, axis = 0)
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

class AdaConvBlock(tf.keras.layers.Layer):
    """ AdaConvBlock
    """
    def __init__(self, Cin, Cout, Ng, Kh, Kw, n_conv, last_act = True, upsample = True, name = 'AdaConvBlock', **kwargs):
        super(AdaConvBlock, self).__init__(name = name, **kwargs)
        self.kernel_predictor = KernelPredictor(Cin, Cin, Ng, Kh, Kw, name='KernelPredictor')
        self.adaconv = AdaConv(Cin, (Kh, Kw), name='AdaConv')

        self._layers = list()
        for i in range(n_conv):
            is_last_layer = i == n_conv - 1
            # if is_last_layer and upsample:
            #     _layer = SubPixelConv2D(Cout, r=2, kernels=(3, 3), strides=1, name=f"SubPixelConv2D_{i}")
            # else:
            _layer = Conv2D(Cout if is_last_layer else Cin, kernels = (3,3), strides = 1, name = f"Conv2D_{i}")
            self._layers.append(_layer)
            if not is_last_layer or last_act:
                self._layers.append(tf.keras.layers.Activation(tf.nn.leaky_relu, name = f"LeakyRelu_{i}"))
            if is_last_layer and upsample:
                self._layers.append(tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear', name = f"Upsample2D_{i}"))

    def call(self, inputs, training = False):
        x, W = inputs
        depthwise_kernels, pointwise_kernels, biases = self.kernel_predictor(inputs = W, training = training)
        x = self.adaconv(inputs = [x, depthwise_kernels, pointwise_kernels, biases], training = training)
        for _layer in self._layers:
            x = _layer(x, training = training)
        return x

class Decoder(tf.keras.Model):
    """
    Decoder Model
    """
    def __init__(self, in_channels, out_channels, n_groups, n_convs, Kh, Kw, name = "Decoder", **kwargs):
        """
        Model Subclass init

        Args:
            
        """
        super(Decoder, self).__init__(name = name, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_groups = n_groups
        self.Kh = Kh
        self.Kw = Kw
        self.n_convs = n_convs

        self.decoder_layers = list()
        for i, (Cin, Cout, Ng, Nc) in enumerate(zip(self.in_channels,self.out_channels,self.n_groups,self.n_convs)):
            is_last_layer = i == len(self.n_groups) - 1
            self.decoder_layers.append(AdaConvBlock(Cin, Cout, Ng, self.Kh, self.Kw, Nc, 
                                                    last_act = not is_last_layer, upsample = not is_last_layer, 
                                                    name = f"AdaConvBlock_{i}"))

        self.final_act = tf.keras.layers.Activation('linear', dtype = tf.float32)

    def call(self, inputs, training = False):
        x, W = inputs
        for _layer in self.decoder_layers:
            x = _layer(inputs = [x, W], training = training)
        x = self.final_act(x)
        return x

class StyleTransfer(tf.keras.Model):
    """
    StyleTransfer Model
    """
    def __init__(self, 
                 image_size, 
                 in_channels, 
                 out_channels, 
                 n_groups, 
                 n_convs, 
                 Sd, 
                 Sh, 
                 Sw, 
                 mean, 
                 freeze_encoder = True, 
                 name = "StyleTransfer", 
                 **kwargs):
        """
        Model Subclass init

        Args:
            image_size (int): Input image size
            channels (list): Decoder channels
            n_groups (list): AdaConv groups at each scale
            n_convs (list): Decoder number of conv layers at each scale
            Sd (int): Style Dimension
            Sw (int): Style Kernel Height
            Kw (int): Style Kernel Width
            mean (list): Image normalization mean
            freeze_encoder (bool): Whether to freeze encoder
        """
        super(StyleTransfer, self).__init__(name = name, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_groups = n_groups
        self.n_convs = n_convs
        self.Sd = Sd
        self.Sh = Sh
        self.Sw = Sw
        self.mean = tf.constant(mean)
        self.image_size = image_size
        self.freeze_encoder = freeze_encoder
        self.backbone_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1']
        backbone = tf.keras.applications.vgg19.VGG19(input_shape=(self.image_size,self.image_size,3), include_top=False, pooling=None)

        self.backbone_out_list = []
        for l in backbone.layers:
            if l.name in self.backbone_layers:
                self.backbone_out_list.append(tf.keras.layers.Activation('linear', dtype = tf.float32)(l.output))
        
        self.encoder = tf.keras.Model(inputs = backbone.input, outputs = self.backbone_out_list)

        if self.freeze_encoder:
            self.encoder.trainable = False

        self.globalstyleencoder = GlobalStyleEncoder(Sd=self.Sd, Sh=self.Sh, Sw=self.Sw,  name='GlobalStyleEncoder')
        self.decoder = Decoder(self.in_channels, self.out_channels, self.n_groups, self.n_convs, self.Sh, self.Sw, name = 'Decoder')
    
    def _preprocess_input(self, x):
        x = x[..., ::-1] # 'RGB'->'BGR'
        x = tf.nn.bias_add(x, tf.cast(-self.mean, dtype = x.dtype))
        return x
    
    def _postprocess_output(self, x):
        x = tf.nn.bias_add(x, tf.cast(self.mean, dtype = x.dtype))
        x = tf.clip_by_value(x, clip_value_min = 0, clip_value_max = 255)
        x = tf.cast(x, dtype = tf.uint8)
        x = x[..., ::-1] # 'BGR'->'RGB'
        return x

    def call(self, inputs, training = False, compute_feats = False):
        
        C, S = inputs
        C = self._preprocess_input(C)
        S = self._preprocess_input(S)
        
        C_feats = self.encoder(C, training = training)
        S_feats = self.encoder(S, training = training)
        W = self.globalstyleencoder(S_feats[-1], training = training)
        x = self.decoder(inputs = [C_feats[-1], W], training = training)

        if training or compute_feats:
            x_feats = self.encoder(x, training = training)
            x = self._postprocess_output(x)
            return x, [C_feats, S_feats, x_feats]
        
        else:
            x = self._postprocess_output(x)
            return x