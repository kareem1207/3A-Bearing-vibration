"""Autoencoders (Keras) and helpers. All models are trained on healthy data only."""
import numpy as np
import keras
from keras import layers


def build_dense(d, bottleneck, hidden=64, noise=0.0, l1=0.0):
    """Dense autoencoder. bottleneck == d gives the 'as wide as the input' case."""
    inp = keras.Input((d,))
    x = layers.GaussianNoise(noise)(inp) if noise > 0 else inp
    x = layers.Dense(hidden, activation="relu")(x)
    if l1 > 0:
        z = layers.Dense(bottleneck, activation="relu",
                         activity_regularizer=keras.regularizers.L1(l1))(x)
    else:
        z = layers.Dense(bottleneck)(x)
    x = layers.Dense(hidden, activation="relu")(z)
    return keras.Model(inp, layers.Dense(d)(x))


def build_lstm(T, d, units=32, bottleneck=8, noise=0.1):
    """Sequence autoencoder over a causal window of T consecutive readings."""
    inp = keras.Input((T, d))
    x = layers.GaussianNoise(noise)(inp) if noise > 0 else inp
    x = layers.LSTM(units)(x)
    z = layers.Dense(bottleneck, activation="tanh")(x)
    x = layers.RepeatVector(T)(z)
    x = layers.LSTM(units, return_sequences=True)(x)
    return keras.Model(inp, layers.TimeDistributed(layers.Dense(d))(x))


def make_windows(X, T):
    """Causal windows (n, T, d); the start of a series is padded with its first reading."""
    pad = np.concatenate([np.repeat(X[:1], T - 1, axis=0), X]) if T > 1 else X
    return pad[np.arange(len(X))[:, None] + np.arange(T)[None, :]]


class Detector:
    """Wraps a spec dict; fit on healthy readings, score = reconstruction MSE of the newest reading."""

    def __init__(self, spec, d, seed):
        self.spec, self.d, self.seed = dict(spec), d, seed
        keras.utils.set_random_seed(seed)
        s = self.spec
        if s["kind"] == "dense":
            b = d if s["bottleneck"] == "input" else s["bottleneck"]
            self.model = build_dense(d, b, s.get("hidden", 64), s.get("noise", 0.0), s.get("l1", 0.0))
        else:
            self.model = build_lstm(s["T"], d, s.get("units", 32), s["bottleneck"], s.get("noise", 0.1))
        self.model.compile(keras.optimizers.Adam(1e-3), loss="mse")

    def _inputs(self, X):
        return make_windows(X, self.spec["T"]) if self.spec["kind"] == "lstm" else X

    def fit(self, X_train_list, X_val_list, epochs=100, patience=10):
        tr = np.concatenate([self._inputs(x) for x in X_train_list])
        va = np.concatenate([self._inputs(x) for x in X_val_list])
        cb = keras.callbacks.EarlyStopping(patience=patience, restore_best_weights=True)
        h = self.model.fit(tr, tr, validation_data=(va, va), epochs=epochs, batch_size=32,
                           callbacks=[cb], verbose=0, shuffle=True)
        return h.history

    def score(self, X):
        W = self._inputs(X)
        R = self.model.predict(W, verbose=0, batch_size=256)
        if W.ndim == 3:
            W, R = W[:, -1], R[:, -1]
        return np.mean((W - R) ** 2, axis=1)
