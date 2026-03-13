# HW06 – Report

> Файл: `homeworks/HW06/report.md`  
> Важно: не меняйте названия разделов (заголовков). Заполняйте текстом и/или вставляйте результаты.

## 1. Dataset

- Какой датасет выбран: `S06-hw-dataset-01.csv`
- Размер: (12001, 30)
- Целевая переменная: `target` принимает 0 и 1
- Признаки: float64 все, кроме id и target, которые int64, категориальных признаков нет

## 2. Protocol

- Разбиение: train/test (использовано стандартное разделение 80/20 с random_state=42 и stratify=y для сохранения распределения классов)
- Подбор: CV на train (сколько фолдов, что оптимизировали)
- Метрики: accuracy, F1, ROC-AUC (и почему эти метрики уместны именно здесь)

## 3. Models

В качестве бейзлайнов использовались DummyClassifier с стратегией most_frequent и LogisticRegression со StandardScaler в пайплайне. DecisionTreeClassifier исследовался с контролем сложности через комбинацию гиперпараметров max_depth, min_samples_split, min_samples_leaf и criterion. RandomForestClassifier настраивался по ключевым "лесным" параметрам: n_estimators, max_depth, min_samples_leaf и max_features. В качестве бустинговой модели был выбран GradientBoostingClassifier с подбором learning_rate, n_estimators, max_depth и min_samples_leaf. Все модели сравнивались по метрикам F1-score (macro), Accuracy и ROC-AUC с использованием корректной схемы кросс-валидации GridSearchCV на тренировочных данных

## 4. Results

- Таблица/список финальных метрик на test по всем моделям
- Победитель (по ROC-AUC или по согласованному критерию) и краткое объяснение

## 5. Analysis

- Устойчивость: что будет, если поменять `random_state` (хотя бы 5 прогонов для 1-2 моделей) – кратко
- Ошибки: confusion matrix для лучшей модели + комментарий
- Интерпретация: permutation importance (top-10/15) + выводы

## 6. Conclusion

3-6 коротких тезисов: что вы поняли про деревья/ансамбли и про честный ML-протокол.
