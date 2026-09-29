import pandas as pd
import streamlit as st

from experiment import REPORTS
from predict import Predictor

st.set_page_config(page_title="Intent Lab · Классификация обращений", page_icon="↗", layout="centered")
st.title("Куда направить обращение?")
st.caption("Определите тему банковского обращения. Введите сообщение на английском языке.")

models = [name for name in ["tfidf", "bert_tiny"] if (REPORTS / f"{name}.json").exists()]
if not models:
    st.warning("Сначала запустите обучение: python experiment.py --model all")
    st.stop()


@st.cache_resource
def load_predictor(name, report_version):
    return Predictor(name)


model = st.selectbox("Модель", models, format_func=lambda name: "TF-IDF + Logistic Regression" if name == "tfidf" else "BERT Tiny")
text = st.text_area("Обращение на английском", value="My card was stolen. How can I block it?", height=120)

if st.button("Определить тему", type="primary", disabled=not text.strip()):
    try:
        with st.spinner("Классифицируем…"):
            result = load_predictor(model, (REPORTS / f"{model}.json").stat().st_mtime_ns).predict(text)
        if result["decision"] == "automatic":
            st.success("Тема обращения: " + result["top3"][0]["label"])
        else:
            st.warning("Нужна проверка человеком: модель недостаточно уверена в ответе.")
        with st.expander("Другие варианты и оценки модели"):
            st.dataframe(pd.DataFrame(result["top3"]).rename(columns={"label": "Тема", "score": "Оценка модели"}), hide_index=True)
            st.caption("Оценка модели не гарантирует правильность ответа.")
    except (ValueError, OSError) as exc:
        st.error(str(exc))
