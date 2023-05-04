import tensorflow as tf
import pandas as pd
from tensorflow import keras
from keras.layers import (
    Input,
    Bidirectional,
    Concatenate,
    LSTM,
    Reshape,
    Flatten,
    Dense,
    TimeDistributed,
)
from keras import backend as K
from .helper import RepeatVector3D, Sampling
from keras.models import Model


class VAE(keras.Model):
    """
    Variational Autoencoder class that implements a custom architecture of encoder and decoder
    that handles raw data plus dynamic and static features as well as custom metrics to track

    Attributes
    ----------
    encoder : keras.Model
        encoder model
    decoder : keras.Model
        decoder model
    window_size : int
        rolling window
    reconstruction_loss_tracker : keras.metrics
        loss computing mean square error on the reconstruction and original data
    kl_loss_tracker: keras.metrics
        kl divergency between simpler learned distribution and actual distribution
    total_loss_tracker : keras.metrics
        sum of reconstruction and kl loss

    """

    def __init__(self, encoder: keras.Model, decoder: keras.Model, window_size: int, **kwargs) -> None:
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.window_size = window_size
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    def call(self, inputs, training=None, mask=None):
        dynamic_features, inp_data, static_features = inputs

        z_mean, z_log_var, z = self.encoder(
            dynamic_features + inp_data + static_features
        )
        pred = self.decoder([z] + dynamic_features + static_features)
        return pred

    @property
    def metrics(self) -> list:
        """
        Metrics to track for the VAE

        :return: metrics to track
        """
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data: list) -> dict:
        """
        Custom training procedure designed to train a VAE and report the relevant metrics

        :param data: input data to the model
        :param window_size: rolling window

        :return: metrics
        """
        dynamic_features, inp_data, static_features = data[0]
        dynamic_features = list(dynamic_features)
        inp_data = list(inp_data)
        static_features = list(static_features)

        device = "/gpu:0" if tf.config.list_physical_devices("GPU") else "/cpu:0"

        with tf.device(device):
            with tf.GradientTape() as tape:
                z_mean, z_log_var, z = self.encoder(
                    dynamic_features + inp_data + static_features
                )
                pred = self.decoder([z] + dynamic_features + static_features)

                reconstruction_loss = K.mean(K.square(inp_data - pred)) * self.window_size
                kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
                kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
                total_loss = reconstruction_loss + kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }


def get_flatten_size_encoder(
    static_features: dict,
    dynamic_features_df: pd.DataFrame,
    window_size: int,
    n_features: int,
    n_features_concat: int,
) -> int:
    inp = Input(shape=(window_size, n_features))

    dynamic_features_inp = []
    for feature in range(len(dynamic_features_df.columns)):
        inp_dyn_feat = Input(shape=(window_size, 1))
        dynamic_features_inp.append(inp_dyn_feat)

    static_feat_inp = []
    for feature, arr in static_features.items():
        inp_static_feat = Input(shape=(n_features, 1))
        static_feat_inp.append(inp_static_feat)

    enc = Concatenate()(dynamic_features_inp + [inp])
    enc = Bidirectional(
        LSTM(
            n_features,
            kernel_initializer="random_uniform",
            input_shape=(window_size, n_features_concat),
        ),
        merge_mode="ave",
    )(enc)

    enc = Reshape((-1, 1))(enc)
    enc = Concatenate()([enc] + static_feat_inp)
    enc = Flatten()(enc)

    # Create a temporary model to compute the flatten size
    temp_model = Model(dynamic_features_inp + [inp] + static_feat_inp, enc)
    flatten_size = temp_model.output_shape[1]
    return flatten_size


def get_mv_model(
    mv_normal_dim: int,
    static_features: dict,
    dynamic_features_df: pd.DataFrame,
    window_size: int,
    n_features: int,
    n_features_concat: int,
    latent_dim: int,
    s: int
) -> tuple[keras.Model, keras.Model]:
    """
    Creating the encoder and decoder models using dynamic and static features
    and learning multivariate normal distribution with dim equal to mv_normal_dim.
    If mv_normal_dim is equal to number of features the multivariate normal has the
    same dimensionality as the input/output series.

    :param mv_normal_dim: dimensionality of the multivariate normal distribution learned
    :param static_features: static features to be inputed to the model
    :param dynamic_features_df: dynamic features to be inputed to the model
    :param window_size: size of the rolling window
    :param n_features: number of features inputed to the model or time series
    :param n_features_concat: number of features inputed to the network after concatenation (raw data + dynamic features)
    :param latent_dim: number of latent dimensions

    :return: encoder model
    :return: decoder model
    """

    # encoder

    inp = Input(shape=(window_size, n_features))

    dynamic_features_inp = []
    for feature in range(len(dynamic_features_df.columns)):
        inp_dyn_feat = Input(shape=(window_size, 1))
        dynamic_features_inp.append(inp_dyn_feat)

    static_feat_inp = []
    for feature, arr in static_features.items():
        inp_static_feat = Input(shape=(n_features, 1))
        static_feat_inp.append(inp_static_feat)

    enc = Concatenate()(dynamic_features_inp + [inp])
    enc = Bidirectional(
        LSTM(
            n_features,
            kernel_initializer="random_uniform",
            input_shape=(window_size, n_features_concat),
        ),
        merge_mode="ave",
    )(enc)

    enc = Reshape((-1, 1))(enc)
    enc = Concatenate()([enc] + static_feat_inp)
    enc = Flatten()(enc)
    enc = Reshape((mv_normal_dim, -1))(enc)
    z = Dense(s, activation="relu")(enc)

    z_mean = Dense(latent_dim)(z)
    z_log_var = Dense(latent_dim)(z)

    z = Sampling()([z_mean, z_log_var])

    encoder = Model(dynamic_features_inp + [inp] + static_feat_inp, [z_mean, z_log_var, z])

    # decoder

    inp_z = Input(shape=(mv_normal_dim, latent_dim))

    dec = RepeatVector3D(window_size)(inp_z)
    dec = Reshape((window_size, -1))(dec)
    dec = Concatenate()([dec] + dynamic_features_inp)

    dec = Bidirectional(
        LSTM(
            n_features,
            kernel_initializer="random_uniform",
            input_shape=(window_size, latent_dim),
            return_sequences=True,
        ),
        merge_mode="ave",
    )(dec)

    dec = TimeDistributed(Dense(n_features))(dec)
    dec = K.permute_dimensions(dec, (0, 2, 1))
    dec = Concatenate(axis=2)([dec] + static_feat_inp)
    dec = Dense(window_size)(dec)
    dec = K.permute_dimensions(dec, (0, 2, 1))
    out = TimeDistributed(Dense(n_features))(dec)

    decoder = Model([inp_z] + dynamic_features_inp + static_feat_inp, out)

    return encoder, decoder
