import tensorflow as tf
from conv import Conv2D
from adain import AdaIn
from adaconv import AdaConv, KernelPredictor, GlobalStyleEncoder

class DecoderBlock(tf.keras.layers.Layer):
    """ DecoderBlock
    """
    def __init__(self, **kwargs):
        super(DecoderBlock, self).__init__(name = kwargs.get('name','DecoderBlock'))
        
        self.last_act = kwargs.get('last_act',True)
        self.upsample = kwargs.get('upsample',True)
        self.model_type = kwargs.get('model_type')

        self.Cin = kwargs.get('Cin')
        self.Cout = kwargs.get('Cout')
        self.n_conv = kwargs.get('n_conv')

        if self.model_type == "AdaConv":
            self.Kh = kwargs.get('Kh')
            self.Kw = kwargs.get('Kw')
            self.Ng = kwargs.get('Ng')

            self.kernel_predictor = KernelPredictor(Cin = self.Cin,
                                                    Cout = self.Cin,
                                                    Ng = self.Ng, 
                                                    Kh = self.Kh, 
                                                    Kw = self.Kw, 
                                                    name ='KernelPredictor')
            self.adaconv = AdaConv(channels = self.Cin, 
                                   kernels = (self.Kh, self.Kw), name='AdaConv')

        self._layers = list()
        for i in range(self.n_conv):
            is_last_layer = i == self.n_conv - 1
            # if is_last_layer and upsample:
            #     self._layers.append(SubPixelConv2D(filters = Cout, 
            #                                        r = 2, 
            #                                        kernels = (3, 3), 
            #                                        strides = 1, 
            #                                        name = f"SubPixelConv2D_{i}")
            # else:
            self._layers.append(Conv2D(filters = self.Cout if is_last_layer else self.Cin, 
                                       kernels = (3,3), strides = 1, 
                                       name = f"Conv2D_{i}"))

            if not is_last_layer or self.last_act:
                self._layers.append(tf.keras.layers.Activation(activation = tf.nn.leaky_relu, 
                                                               name = f"LeakyRelu_{i}"))
            if is_last_layer and self.upsample:
                self._layers.append(tf.keras.layers.UpSampling2D(size = (2, 2), 
                                                                 interpolation = 'nearest', 
                                                                 name = f"Upsample2D_{i}"))

    def call(self, inputs, training = False):
        
        if self.model_type == "AdaConv":
            x, W = inputs
            depthwise_kernels, pointwise_kernels, biases = self.kernel_predictor(inputs = W, training = training)
            x = self.adaconv(inputs = [x, depthwise_kernels, pointwise_kernels, biases], training = training)
        else:
            x = inputs
        
        for _layer in self._layers:
            x = _layer(inputs = x, training = training)
        
        return x

class Decoder(tf.keras.Model):
    """
    Decoder Model
    """
    def __init__(self, **kwargs):
        """
        Model Subclass init

        Args:
            
        """
        super(Decoder, self).__init__(name = kwargs.get('name','Decoder'))

        self.model_type = kwargs.get('model_type')

        self.in_channels = kwargs.get('in_channels')
        self.out_channels = kwargs.get('out_channels')
        self.n_convs = kwargs.get('n_convs')
        if self.model_type == "AdaConv":
            self.n_groups = kwargs.get('n_groups')
            self.Kh = kwargs.get('Kh')
            self.Kw = kwargs.get('Kw')

        self.decoder_layers = list()
        for i in range(len(self.n_convs)):
            is_last_layer = i == len(self.n_convs) - 1
            kwargs = {"last_act" : not is_last_layer,
                      "upsample" : not is_last_layer,
                      "name" : f"DecoderBlock_{i}",
                      "Cin" : self.in_channels[i], 
                      "Cout" : self.out_channels[i], 
                      "n_conv" : self.n_convs[i],
                      "model_type" : self.model_type}
            if self.model_type == "AdaConv":
                kwargs.update({"Ng" : self.n_groups[i], 
                               "Kh" : self.Kh, 
                               "Kw" : self.Kw})
            self.decoder_layers.append(DecoderBlock(**kwargs))

        self.final_act = tf.keras.layers.Activation(activation = 'sigmoid', dtype = tf.float32)

    def call(self, inputs, training = False):
        
        if self.model_type == "AdaConv":
            x, W = inputs
        else:
            x = inputs
        
        for _layer in self.decoder_layers:
            x = _layer(inputs = [x, W] if self.model_type == "AdaConv" else x, training = training)

        x = self.final_act(x)
        return x

class StyleTransfer(tf.keras.Model):
    """
    StyleTransfer Model
    """
    def __init__(self, **kwargs):
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
        super(StyleTransfer, self).__init__(name = kwargs.get('name','StyleTransfer'))

        self.model_type = kwargs.get('model_type')
        self.in_channels = kwargs.get("in_channels")
        self.out_channels = kwargs.get("out_channels")
        self.n_convs = kwargs.get("n_convs")
        # self.mean = tf.constant(kwargs.get("mean"))
        self.image_size = kwargs.get("image_size")
        self.freeze_encoder = kwargs.get("freeze_encoder", True)
        self.backbone_output_layers = kwargs.get("backbone_output_layers")
        if self.model_type == "AdaConv":
            self.n_groups = kwargs.get("n_groups")
            self.Sd = kwargs.get("Sd")
            self.Sh = kwargs.get("Sh")
            self.Sw = kwargs.get("Sw")
        backbone = tf.keras.applications.vgg19.VGG19(input_shape = (self.image_size,self.image_size,3), include_top = False, pooling = None)

        backbone_outputs = list()
        for l in backbone.layers:
            if l.name in self.backbone_output_layers:
                backbone_outputs.append(tf.keras.layers.Activation(activation = 'linear', dtype = tf.float32)(l.output))
        
        self.encoder = tf.keras.Model(inputs = backbone.input, outputs = backbone_outputs)

        if self.freeze_encoder:
            self.encoder.trainable = False

        kwargs = {"model_type" : self.model_type,
                  "in_channels" : self.in_channels,
                  "out_channels" : self.out_channels, 
                  "n_convs" : self.n_convs, 
                  "name" : 'Decoder'}

        if self.model_type == "AdaConv":
            kwargs.update({"n_groups" : self.n_groups, 
                           "Kh" : self.Sh, 
                           "Kw" : self.Sw})

            self.globalstyleencoder = GlobalStyleEncoder(Sd = self.Sd, 
                                                         Sh = self.Sh, 
                                                         Sw = self.Sw,  
                                                         name = 'GlobalStyleEncoder')
        else:
            self.adaptive_instance_norm = AdaIn(name = 'AdaIn')          

        self.decoder = Decoder(**kwargs)

    # def _preprocess_input(self, x):
    #     x = x[..., ::-1] # 'RGB'->'BGR'
    #     x = tf.nn.bias_add(x, tf.cast(-self.mean, dtype = x.dtype))
    #     return x
    
    # def _postprocess_output(self, x):
    #     x = tf.nn.bias_add(x, tf.cast(self.mean, dtype = x.dtype))
    #     x = tf.clip_by_value(x, clip_value_min = 0, clip_value_max = 255)
    #     x = tf.cast(x, dtype = tf.uint8)
    #     x = x[..., ::-1] # 'BGR'->'RGB'
    #     return x

    def call(self, inputs, training = False, **kwargs):
        
        C, S = inputs
        # C = self._preprocess_input(C)
        # S = self._preprocess_input(S)
        
        C_feats = self.encoder(C, training = False)
        S_feats = self.encoder(S, training = False)
        
        if self.model_type == "AdaConv":
            W = self.globalstyleencoder(inputs = S_feats[-1], training = training)
        else:
            adain_output = self.adaptive_instance_norm(inputs = [C_feats[-1], S_feats[-1]])
            
            if 'alpha' in kwargs.keys():
                alpha = kwargs.get('alpha')
                adain_output = C_feats[-1] * (1 - alpha) + adain_output * alpha
        
        x = self.decoder(inputs = [C_feats[-1], W] if self.model_type == "AdaConv" else adain_output, training = training)

        if kwargs.get('get_feats'):
            x_feats = self.encoder(x, training = False)
            # x = self._postprocess_output(x)

            return x, {"content_features" : C_feats if self.model_type == "AdaConv" else [adain_output], 
                       "style_features" : S_feats, 
                       "combined_features" : x_feats}

        # x = self._postprocess_output(x)
        return x