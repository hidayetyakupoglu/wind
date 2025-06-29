import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer, Input, Conv1D, MaxPooling1D, concatenate, Bidirectional, GRU, TimeDistributed, Dense


# Sayfa yapılandırması
st.set_page_config(page_title="GF Anomali Tespiti", layout="wide")
st.title("🔌 (GF) Anomali Tespit Paneli")

# Model ve veri yolu
MODEL_PATH = "cnn_bigru_online_kalman.h5"
DATA_PATH = "data_test.xlsx"

# GF için kullanılacak 12 sensör
selected_features = [
    'WEC: ava. blade angle A',
    'WEC: max. Rotation',
    'WEC: max. Power',
    'Inverter std dev',
    'WEC: max. windspeed',
    'WEC: max. reactive Power',
    'Stator temp. 2',
    'WEC: min. Rotation',
    'WEC: ava. Power',
    'Rotor temp. 1',
    'WEC: ava. Rotation',
    'Rear bearing temp.'
]
'''
# Model yükleme
@st.cache_resource
def load_model(path):
    try:
        model = load_model(path)
        return model
    except Exception as e:
        st.error(f"Model yüklenirken hata: {e}")
        return None
'''
class OnlineLearningKalmanFilterLayer(Layer):
    def __init__(self, learning_rate=0.01, **kwargs):
        super(OnlineLearningKalmanFilterLayer, self).__init__(**kwargs)
        self.learning_rate = learning_rate

    def build(self, input_shape):
        feature_dim = input_shape[0][-1]

        self.R = self.add_weight(name='R', shape=(feature_dim,), initializer='ones', trainable=True)
        self.Q = self.add_weight(name='Q', shape=(feature_dim,), initializer='ones', trainable=True)

        # Bu şekilde P'nin ilk değerini tf.Variable ile oluştur
        self.P = self.add_weight(name='P', shape=(feature_dim,), initializer='ones', trainable=False)

        # Optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

    def call(self, inputs):
        measurements, predictions = inputs

        # Eğitim modunda online öğrenme
        if tf.executing_eagerly():
            with tf.GradientTape() as tape:
                # Hata kovaryansını güncelle
                P_pred = self.P + self.Q

                # Kalman kazancını hesapla
                K = P_pred / (P_pred + self.R + 1e-7)

                # Filtrelenmiş tahmin
                estimate = predictions + K * (measurements - predictions)

                # Online loss hesapla (MAE)
                loss = tf.reduce_mean(tf.abs(measurements - estimate))

            # R ve Q için gradientleri al
            grads = tape.gradient(loss, [self.R, self.Q])

            # Gradientleri uygula
            self.optimizer.apply_gradients(zip(grads, [self.R, self.Q]))

            # Hata kovaryansını güncelle
            self.P.assign((1 - K) * P_pred)
        else:
            # Sembolik hesaplama (model oluşturma aşaması)
            P_pred = self.P + self.Q
            K = P_pred / (P_pred + self.R + 1e-7)
            estimate = predictions + K * (measurements - predictions)

        return estimate
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanAbsoluteError

# Özel katmanlar ve mae kaybı fonksiyonu için custom_objects
custom_objects = {
    'OnlineLearningKalmanFilterLayer': OnlineLearningKalmanFilterLayer,
    'AttentionLayer': AttentionLayer,
    'mae': MeanAbsoluteError()  # mae kaybı fonksiyonunu ekle
}

# Modeli yükleme
model = load_model('cnn_bigru_online_kalman.h5', custom_objects=custom_objects)
# Veri yükleme
@st.cache_data
def load_data(path):
    try:
        df = pd.read_excel(path)
        return df[selected_features]
    except Exception as e:
        st.error(f"Veri yüklenirken hata: {e}")
        return None

df = load_data(DATA_PATH)
if df is None:
    st.stop()

# Eşik değeri seçimi
threshold = st.slider("GF Anomali Eşik Değeri", min_value=0.0, max_value=1.0, value=0.05, step=0.001)

# Başlatma düğmesi
if st.button("🚀 Testi Başlat"):
    score_history = []
    placeholder = st.empty()

    for i in range(len(df)):
        try:
            # Ölçüm verisi
            measurements = df.iloc[i].values.reshape(1, len(selected_features)).astype(np.float32)
            # Tahmin verisi (örnek olarak sıfır vektörü, modelinize göre ayarlayın)
            predictions = np.zeros_like(measurements).astype(np.float32)
            inputs = [measurements, predictions]
            score = model.predict(inputs, verbose=0)[0]
            if isinstance(score, np.ndarray):
                score = score.item()
        except Exception as e:
            st.error(f"Model tahmini sırasında hata: {e}")
            score = 0.0

        score_history.append(score)
        status = "🟢 Normal" if score < threshold else "🔴 Anomali"

        with placeholder.container():
            st.subheader(f"🧪 Zaman Adımı: {i+1}")
            st.metric("Anomali Skoru", f"{score:.5f}", delta="Normal" if status == "🟢 Normal" else "Anomali")
            st.markdown(f"**Durum:** {status}")

            fig, ax = plt.subplots()
            ax.plot(score_history, color="blue", label="Skor")
            ax.axhline(threshold, color="red", linestyle="--", label="Threshold")
            ax.set_title("GF Modeli - Anomali Skoru Zaman Serisi")
            ax.set_xlabel("Zaman")
            ax.set_ylabel("Skor")
            ax.legend()
            st.pyplot(fig)

        time.sleep(5)  # 5 saniyede bir veri gönderimi
