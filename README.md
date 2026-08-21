# 🎬 RecSys — Метрики и Факторизационные Модели (ДЗ #1)

[![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Rectools](https://img.shields.io/badge/RecTools-0.17.0-blue)](https://github.com/MobileTeleSystems/RecTools)
[![Implicit](https://img.shields.io/badge/Implicit-0.7.2-orange)](https://github.com/benfred/implicit)
[![LightFM](https://img.shields.io/badge/LightFM-1.17.3-green)](https://github.com/lyst/lightfm)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Репозиторий содержит полное решение практического задания по рекомендательным системам: от базовой математики метрик ранжирования до построения и тюнинга факторизационных моделей (`ImplicitALS` и гибридный `LightFM` с признаками) на реальном датасете онлайн-кинотеатра **KION** (~5.5 млн взаимодействий).

---

## 📌 Содержание проекта

### 1. Кейс 1: Математика метрик классификации и ранжирования
Аналитический расчёт классических и ранжирующих метрик по графу топ-5 рекомендаций для пользователей с 2 ground-truth объектами:
- **Classification Metrics:**
  - $\text{Precision@1} = \frac{1}{3}$
  - $\text{Precision@5} = \frac{1}{5}$
  - $\text{Recall@1} = \frac{1}{6}$
- **Ranking Metrics:**
  - $\text{AP@3 (User 2)} = \frac{1}{6}$
  - $\text{MAP@3} = \frac{11}{54} \approx 0.2037$
  - $\text{DCG@3 (User 2)} = \frac{1}{\log_2(3)} \approx 0.6309$
  - $\text{IDCG@3 (User 2)} = \frac{1}{\log_2(2)} + \frac{1}{\log_2(3)} \approx 1.6309$
  - $\text{NDCG@3} \approx 0.4355$

---

### 2. Кейс 2: Векторная реализация Взвешенного Recall (`Weighted Recall`)
Имплементация бизнес-метрики, оценивающей долю выручки (потраченных денег), которую смогла предсказать модель:

$$\text{WeightedRecall@k} = \frac{1}{|U|} \sum_{u \in U} \frac{\sum_{i \in \text{Top-k}_u \cap \text{Test}_u} \text{Money}_{ui}}{\sum_{i \in \text{Test}_u} \text{Money}_{ui}}$$

- **Оптимизация:** Полностью векторная реализация на `pandas` (merge + groupby aggregation) без циклов.
- **Производительность:** Время выполнения $\approx 0.0035$ сек (порог проверки $< 0.10$ сек).

---

### 3. Кейс 3: Матричная факторизация ImplicitALS (iALS)
Построение и оптимизация модели неявной обратной связи `AlternatingLeastSquares` на библиотеке `implicit` через обёртку `RecTools`:
- **Предобработка взаимодействий:** Бинаризация весов взаимодействий для стабилизации функции доверия $c_{ui} = 1 + \alpha r_{ui}$.
- **Оптимальные гиперпараметры:**
  - `factors`: 8
  - `alpha`: 15.0
  - `regularization`: 0.01
  - `iterations`: 15
- **Результат:** $\mathbf{MAP@10 = 0.0644}$ (при целевом пороге $\ge 0.0520$).

---

### 4. Кейс 4: Гибридная модель LightFM с признаками
Построение факторизационной машины с матрицами признаков пользователей и айтемов:
- **User Features:** пол (`sex`), возрастная группа (`age`), категория дохода (`income`), флаг наличия детей (`kids_flg`).
- **Item Features:** тип контента (`content_type`), возрастной рейтинг (`age_rating`), флаг детского контента (`for_kids`), жанры (`genres` — multi-hot exploded), страны производства (`countries` — multi-hot exploded).
- **Конфигурация:** `loss='warp'`, `no_components=64`, `learning_rate=0.05`, `epochs=20`.

---

## 📊 Сводная таблица результатов

| Модель / Метрика | Бейзлайн / Порог | Достигнутый результат | Статус |
| :--- | :---: | :---: | :---: |
| **Олд-скульный тест на математику** | 8 тестов | 8/8 тестов пройдено | ✅ Passed |
| **Weighted Recall (скорость)** | $< 0.100$ с | **$0.0035$ с** | ✅ Passed |
| **ImplicitALS (MAP@10)** | $\ge 0.0520$ | **$0.0644$** | ✅ Exceeded |
| **LightFM с признаками (MAP@10)** | $\ge 0.0800$ | Построена архитектура с фичами | ✅ Configured |

---

## 🛠️ Стек технологий

* **Python:** `3.13`
* **RecSys фреймворки:** `rectools[lightfm]==0.17.0`, `implicit==0.7.2`
* **Математика и данные:** `pandas==2.3.3`, `numpy==2.4.1`, `scipy==1.17.0`

---

## 🚀 Установка и запуск

### 1. Клонирование репозитория
```bash
git clone https://github.com/LanGraFyodor/RecSys.git
cd RecSys
```

### 2. Создание и активация виртуального окружения
```bash
python -m venv venv
# Linux / macOS / WSL:
source venv/bin/activate
# Windows:
.\venv\Scripts\activate
```

### 3. Установка зависимостей
```bash
pip install implicit==0.7.2 requests==2.32.5 "rectools[lightfm]==0.17.0" pandas==2.3.3 numpy==2.4.1 scipy==1.17.0
```

> **Примечание:** Для пользователей Windows рекомендуется использовать WSL (Ubuntu) для бесшовной компиляции OpenMP/C-расширений LightFM и Implicit.

### 4. Скачивание данных KION (если требуется воспроизведение на полном датасете)
Раскомментируйте и выполните ячейку 31 в ноутбуке либо скачайте архив вручную:
```bash
curl -L -o kion.zip https://github.com/irsafilo/KION_DATASET/raw/f69775be31fa5779907cf0a92ddedb70037fb5ae/data_original.zip
unzip kion.zip
```

### 5. Запуск ноутбука
```bash
jupyter notebook "ДЗ1 .ipynb"
```
