"""Variable Selection Network and its building blocks.

Source: T3.03_PS6_Solutions.ipynb (Madmoun, L6). GatedLinearUnits (GLU),
GatedResidualNetwork (GRN), InputTransformation (Dense for numerical + Embedding
for categorical), VariableSelectionNetwork (per-feature GRNs combined by softmax
selection weights), and a FinalModel for binary classification. Reproduced
faithfully; the softmax weights double as per-sample feature importances.

Stack: tensorflow.
"""
import tensorflow as tf
from tensorflow.keras.layers import Dense, Embedding, Dropout, LayerNormalization


class GatedLinearUnits(tf.keras.layers.Layer):
    """GLU: sigmoid filter (in [0,1]) gates a linear projection element-wise."""

    def __init__(self, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.dense_filter = Dense(output_dim, activation='sigmoid')
        self.dense_vector = Dense(output_dim)

    def call(self, x):
        return tf.multiply(self.dense_filter(x), self.dense_vector(x))


class GatedResidualNetwork(tf.keras.layers.Layer):
    """GRN: Dense(ELU)->Dense->Dropout->GLU, residual skip, LayerNorm."""

    def __init__(self, hidden_dim, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.output_dim = output_dim
        self.projection = Dense(output_dim)
        self.dense_1 = Dense(hidden_dim, activation='elu')
        self.dense_2 = Dense(hidden_dim)
        self.dropout = Dropout(0.1)
        self.glu = GatedLinearUnits(output_dim)
        self.layer_norm = LayerNormalization()

    def call(self, x):
        z = self.glu(self.dropout(self.dense_2(self.dense_1(x))))
        if x.shape[-1] != self.output_dim:
            x = self.projection(x)                    # match dims for the skip
        return self.layer_norm(x + z)                 # residual + layer norm


class InputTransformation(tf.keras.layers.Layer):
    """Project each numerical feature (Dense) and embed each categorical
    feature (Embedding) into a common embedding space, then stack."""

    def __init__(self, embedding_dim, num_numerical, num_categorical, cardinalities, **kwargs):
        super().__init__(**kwargs)
        self.list_projection_layers = [Dense(embedding_dim) for _ in range(num_numerical)]
        self.list_embedding_layers = [Embedding(input_dim=c, output_dim=embedding_dim)
                                      for c in cardinalities]

    def call(self, x_num, x_cat):
        num = [proj(x_num[:, i:i + 1]) for i, proj in enumerate(self.list_projection_layers)]
        cat = [emb(x_cat[:, i]) for i, emb in enumerate(self.list_embedding_layers)]
        return tf.stack(num + cat, axis=-1)           # (batch, embedding_dim, n_features)


class VariableSelectionNetwork(tf.keras.layers.Layer):
    """Per-feature GRNs combined by softmax selection weights (= importances)."""

    def __init__(self, num_features, hidden_dim, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.list_grns = [GatedResidualNetwork(hidden_dim, output_dim) for _ in range(num_features)]
        self.flatten_grn = GatedResidualNetwork(hidden_dim, num_features)

    def call(self, stack_features):                   # (batch, embedding_dim, num_features)
        feats = [grn(stack_features[:, :, i]) for i, grn in enumerate(self.list_grns)]
        stacked = tf.stack(feats, axis=-1)            # (batch, output_dim, num_features)
        batch = tf.shape(stacked)[0]
        flat = tf.reshape(stacked, (batch, stacked.shape[1] * stacked.shape[2]))
        weights = tf.nn.softmax(self.flatten_grn(flat), axis=-1)         # (batch, num_features)
        output = tf.reduce_sum(stacked * weights[:, tf.newaxis, :], axis=-1)
        return output, weights                         # weights sum to 1 per sample


class FinalModel(tf.keras.models.Model):
    """InputTransformation -> VSN -> Dense(sigmoid). Returns (prediction, alpha)."""

    def __init__(self, embedding_dim, num_numerical, num_categorical,
                 cardinalities, hidden_dim, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.input_transformation = InputTransformation(
            embedding_dim, num_numerical, num_categorical, cardinalities)
        self.vsn = VariableSelectionNetwork(
            num_numerical + num_categorical, hidden_dim, output_dim)
        self.dense = Dense(1, activation='sigmoid')

    def call(self, inputs):
        x_num, x_cat = inputs
        stack = self.input_transformation(x_num, x_cat)
        out, alpha = self.vsn(stack)
        return tf.squeeze(self.dense(out)), alpha


if __name__ == "__main__":
    import numpy as np
    tf.random.set_seed(42); np.random.seed(42)
    N, Dn, cards = 64, 4, [5, 3, 7]
    x_num = tf.constant(np.random.randn(N, Dn), dtype=tf.float32)
    x_cat = tf.constant(np.column_stack([np.random.randint(0, c, N) for c in cards]),
                        dtype=tf.int32)
    fm = FinalModel(embedding_dim=32, num_numerical=Dn, num_categorical=len(cards),
                    cardinalities=cards, hidden_dim=8, output_dim=14)
    pred, alpha = fm((x_num, x_cat))
    assert abs(float(tf.reduce_mean(tf.reduce_sum(alpha, axis=1))) - 1.0) < 1e-4
    print("prediction:", pred.shape, " alpha (importances):", alpha.shape)
