import re


def clean_text(text: str) -> str:
    """
    Basic text preprocessing for Russian review sentiment classification.
    """
    if text is None:
        return ""

    text = str(text).lower()
    text = text.replace("ё", "е")
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^а-яa-z0-9\s!?.,-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text