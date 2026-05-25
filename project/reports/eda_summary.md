
# EDA Summary

## Dataset

В проекте используется датасет RuReviews: русскоязычные отзывы о женской одежде и аксессуарах.

## Raw dataset

- Rows: 90000
- Columns: 2
- Text column: review
- Target column: sentiment

## Prepared dataset

- Rows after preprocessing: 86146
- Columns after preprocessing: 2

## Class distribution after preprocessing

sentiment
positive    28956
negative    28817
neutral     28373

## Preprocessing steps

- Исправлена метка `neautral` на `neutral`.
- Текст приведён к нижнему регистру.
- Символ `ё` заменён на `е`.
- Удалены ссылки.
- Удалены лишние символы.
- Удалены пустые строки.
- Удалены дубликаты по тексту отзыва.

## Main observations

1. Исходный датасет сбалансирован по трём классам.
2. В датасете есть три класса: negative, neutral, positive.
3. В исходных метках была опечатка `neautral`, она исправлена.
4. Отзывы имеют разную длину: есть короткие и более длинные тексты.
5. Датасет подходит для обучения baseline-моделей на основе TF-IDF.

## Risks and limitations

1. Датасет относится к конкретной товарной категории: одежда и аксессуары.
2. Модель может хуже работать на отзывах из других доменов.
3. Возможны ошибки в автоматической разметке.
4. Короткие отзывы могут быть неоднозначными для классификации.
