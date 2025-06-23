import tensorflow as tf
from tensorflow.keras.layers import Layer

class OnlineKalmanFilterLayer(Layer):
    def __init__(self, learning_rate=0.01, **kwargs):
        super(OnlineKalmanFilterLayer, self).__init__(**kwargs)
        self.learning_rate = learning_rate

    def build(self, input_shape):
        feature_dim = input_shape[0][-1]

        self.R = self.add_weight(name='measurement_noise',
                                 shape=(feature_dim,),
                                 initializer='ones',
                                 trainable=True)
        self.Q = self.add_weight(name='process_noise',
                                 shape=(feature_dim,),
                                 initializer='ones',
                                 trainable=True)

        self.P = self.add_weight(name='error_covariance',
                                 shape=(feature_dim,),
                                 initializer='ones',
                                 trainable=False)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

    def call(self, inputs):
        measurements, predictions = inputs

        if tf.executing_eagerly():
            with tf.GradientTape() as tape:
                P_pred = self.P + self.Q
                K = P_pred / (P_pred + self.R + 1e-7)
                estimate = predictions + K * (measurements - predictions)
                loss = tf.reduce_mean(tf.abs(measurements - estimate))

            grads = tape.gradient(loss, [self.R, self.Q])
            self.optimizer.apply_gradients(zip(grads, [self.R, self.Q]))
            self.P.assign((1 - K) * P_pred)
        else:
            P_pred = self.P + self.Q
            K = P_pred / (P_pred + self.R + 1e-7)
            estimate = predictions + K * (measurements - predictions)

        return estimate

    def get_config(self):
        config = super(OnlineKalmanFilterLayer, self).get_config()
        config.update({
            'learning_rate': self.learning_rate
        })
        return config
