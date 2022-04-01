import tensorflow as tf

class AdaIn(tf.keras.layers.Layer):
    """ AdaIn (Adaptive Instance Normalization)
    """
    def __init__(self, **kwargs):
        super(AdaIn, self).__init__(name = kwargs.get('name','AdaIn'))
        self.reduction_axes = [1,2]
        self.epsilon = 1e-5
    
    def _get_mean_std(self, x):
        mean, variance = tf.nn.moments(x, axes = self.reduction_axes, keepdims=True)
        standard_deviation = tf.sqrt(variance + self.epsilon)
        return mean, standard_deviation
    
    def call(self, inputs):
        
        content, style = inputs
        content = tf.cast(content, dtype = tf.float32)
        style = tf.cast(style, dtype = tf.float32)

        content_mean, content_std = self._get_mean_std(content)
        style_mean, style_std = self._get_mean_std(style)

        x = (style_std * ((content - content_mean) / content_std)) + style_mean

        return x