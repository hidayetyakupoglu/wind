import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import time
from tensorflow.keras.models import load_model
from my_layers import OnlineKalmanFilterLayer

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

# Model yükleme
@st.cache_resource
def load_model(path):
    try:
        model = load_model("cnn_bigru_online_kalman.h5")
        return model
    except Exception as e:
        st.error(f"Model yüklenirken hata: {e}")
        return None

model = load_model(MODEL_PATH)
if model is None:
    st.stop()

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
        row = df.iloc[i].values.reshape(1, len(selected_features), 1)  # Modelin giriş formatına göre ayarlayın
        try:
            score = model.predict(row, verbose=0)[0]
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
