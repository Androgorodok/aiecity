from __future__ import annotations

import pandas as pd

from eda_cli.core import (
    compute_quality_flags,
    correlation_matrix,
    flatten_summary_for_print,
    missing_table,
    summarize_dataset,
    top_categories,
)


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [10, 20, 30, None],
            "height": [140, 150, 160, 170],
            "city": ["A", "B", "A", None],
        }
    )


def test_summarize_dataset_basic():
    df = _sample_df()
    summary = summarize_dataset(df)

    assert summary.n_rows == 4
    assert summary.n_cols == 3
    assert any(c.name == "age" for c in summary.columns)
    assert any(c.name == "city" for c in summary.columns)

    summary_df = flatten_summary_for_print(summary)
    assert "name" in summary_df.columns
    assert "missing_share" in summary_df.columns


def test_missing_table_and_quality_flags():
    df = _sample_df()
    missing_df = missing_table(df)

    assert "missing_count" in missing_df.columns
    assert missing_df.loc["age", "missing_count"] == 1

    summary = summarize_dataset(df)
    flags = compute_quality_flags(summary, missing_df)
    assert 0.0 <= flags["quality_score"] <= 1.0


def test_correlation_and_top_categories():
    df = _sample_df()
    corr = correlation_matrix(df)
    # корреляция между age и height существует
    assert "age" in corr.columns or corr.empty is False

    top_cats = top_categories(df, max_columns=5, top_k=2)
    assert "city" in top_cats
    city_table = top_cats["city"]
    assert "value" in city_table.columns
    assert len(city_table) <= 2


def test_compute_quality_flags_has_suspicious_id_duplicates():
    """Тест проверяет обнаружение дубликатов в user_id."""
    
    # Создание DataFrame с дублирующимися user_id
    df = pd.DataFrame({
        'user_id': [1, 2, 3, 3, 4, 5],  # user_id=3 встречается дважды
        'name': ['Alice', 'Bob', 'Charlie', 'Charlie', 'David', 'Eve'],
        'age': [25, 30, 35, 35, 40, 45],
        'score': [85, 90, 78, 78, 92, 88]
    })
    
    # Создаем DatasetSummary с динамическим созданием ColumnSummary
    summary = type('DatasetSummary', (), {
        'n_rows': len(df),
        'n_cols': len(df.columns),
        'columns': [
            type('ColumnSummary', (), {
                'name': 'user_id',
                'dtype': 'int64',
                'unique': df['user_id'].nunique(),  # 5 уникальных значений
                'is_numeric': True,
            })(),
            # Можно добавить остальные колонки или оставить только user_id
        ]
    })()
    
    # Создаем пустой missing_df для теста
    missing_df = pd.DataFrame(columns=['missing_share'])
    
    # Импортируем функцию
    from eda_cli.core import compute_quality_flags
    
    # Вызываем тестируемую функцию
    flags = compute_quality_flags(summary, missing_df)
    
    # Проверяем, что флаг установлен в True (есть дубликаты)
    assert flags['has_suspicious_id_duplicates'] == True, (
        f"Ожидалось has_suspicious_id_duplicates=True, "
        f"так как user_id имеет дубликаты. Получено: {flags}"
    )
    
    # Проверяем, что другие флаги тоже существуют
    assert 'quality_score' in flags
    assert 'has_constant_columns' in flags
    
    # Проверяем, что quality_score снижен из-за дубликатов
    assert flags['quality_score'] < 1.0


def test_compute_quality_flags_has_constant_columns():
    """Тест проверяет обнаружение константных колонок."""
    
    # Создаем мок-объекты без явного объявления классов
    # Используем type() для создания объектов на лету
    
    # 1. Создаем колонки
    constant_col = type('ColumnSummary', (), {
        'name': 'constant_col',
        'unique': 1,  # Константная колонка
    })()
    
    variable_col = type('ColumnSummary', (), {
        'name': 'variable_col',
        'unique': 5,  # Переменная колонка
    })()
    
    # 2. Создаем DatasetSummary
    summary = type('DatasetSummary', (), {
        'n_rows': 5,
        'n_cols': 2,
        'columns': [constant_col, variable_col]
    })()
    
    # 3. Создаем missing_df
    missing_df = pd.DataFrame({'missing_share': [0.0, 0.0]})
    
    # 4. Вызываем функцию и проверяем
    flags = compute_quality_flags(summary, missing_df)
    
    assert flags['has_constant_columns'] == True, (
        f"Ожидалось has_constant_columns=True, "
        f"так как есть константная колонка. Получено: {flags}"
    )
