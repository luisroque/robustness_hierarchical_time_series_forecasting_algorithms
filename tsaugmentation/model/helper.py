import tensorflow as tf
from tensorflow.keras.layers import Layer, InputSpec
from tensorflow.keras.backend import repeat_elements
from tensorflow.python.ops import array_ops


class Sampling(Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        n_features = tf.shape(z_mean)[2]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim, n_features))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class RepeatVector3D(Layer):
    """
    Repeats the input n times.
    Example:
    ```python
    inp = tf.keras.Input(shape=(4,4))
    # now: model.output_shape == (None, 4,4)
    # note: `None` is the batch dimension
    output = RepeatVector3D(3)(inp)
    # now: model.output_shape == (None, 3, 4, 4)
    model = tf.keras.Model(inputs=inp, outputs=output)
    ```
    Args:
      n: Integer, repetition factor
    Input shape:
      3D tensor of shape `(None, x, y)`
    Output shape:
      4D tensor of shape `(None, n, x, y)`
    """

    def __init__(self, n, **kwargs):
        super(RepeatVector3D, self).__init__(**kwargs)
        self.n = n
        self.input_spec = InputSpec(ndim=3)

    def compute_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape).as_list()
        return tf.TensorShape([input_shape[0], self.n, input_shape[1]])

    def call(self, inputs):
        inputs = array_ops.expand_dims(inputs, 1)
        repeat = repeat_elements(inputs, self.n, axis=1)
        return repeat

    def get_config(self):
        config = {'n': self.n}
        base_config = super(RepeatVector3D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))