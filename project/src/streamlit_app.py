import pandas as pd
import streamlit as st

from src.batch_analysis import analyze_dataframe, get_sentiment_distribution
from src.predict import predict_sentiment


def show_single_prediction(sentiment: str, confidence: float | None) -> None:
    message = f"Тональность: {sentiment}"

    if confidence is not None:
        message += f"\n\nУверенность: {confidence:.2%}"

    if sentiment == "positive":
        st.success(message)
    elif sentiment == "neutral":
        st.info(message)
    elif sentiment == "negative":
        st.error(message)
    else:
        st.write(message)


st.set_page_config(
    page_title="Анализ тональности отзывов",
    page_icon="💬",
    layout="wide",
)

st.title("Анализ тональности пользовательских отзывов")

single_tab, csv_tab = st.tabs(["Один отзыв", "CSV-анализ"])

with single_tab:
    text = st.text_area("Текст отзыва")

    if st.button("Определить тональность"):
        if not text.strip():
            st.warning("Введите текст отзыва.")
        else:
            try:
                prediction = predict_sentiment(text)
                show_single_prediction(
                    prediction["sentiment"],
                    prediction["confidence"],
                )
            except Exception as error:
                st.error(f"Ошибка анализа: {error}")

with csv_tab:
    uploaded_file = st.file_uploader("Загрузите CSV-файл", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
        except pd.errors.EmptyDataError:
            st.warning("CSV-файл пустой.")
        except Exception as error:
            st.error(f"Не удалось прочитать CSV: {error}")
        else:
            if df.empty:
                st.warning("CSV-файл пустой.")
            elif len(df.columns) == 0:
                st.warning("В CSV-файле нет колонок.")
            else:
                text_column = st.selectbox("Текстовая колонка", df.columns)

                if st.button("Проанализировать CSV"):
                    try:
                        result_df = analyze_dataframe(df, text_column=text_column)
                        distribution_df = get_sentiment_distribution(result_df)

                        st.dataframe(result_df, use_container_width=True)
                        st.bar_chart(
                            distribution_df,
                            x="sentiment",
                            y="count",
                            use_container_width=True,
                        )

                        csv_data = result_df.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            label="Скачать результаты CSV",
                            data=csv_data,
                            file_name="sentiment_results.csv",
                            mime="text/csv",
                        )
                    except Exception as error:
                        st.error(f"Ошибка CSV-анализа: {error}")
