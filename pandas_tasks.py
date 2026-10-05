"""Задачи первой части лабораторной: Pandas и Titanic."""
from __future__ import annotations

import pandas as pd

from grader_contracts.pandas_tasks import TitanicInput, TitanicSummary


def analyze_titanic(data: TitanicInput) -> TitanicSummary:
    """Выполните загрузку и анализ датасета Titanic."""

    data_frame = pd.read_csv(data.csv_path)

    row_count = len(data_frame)

    # кол-во пропусков в каждом столбце
    missing_by_column = data_frame.isnull().sum().to_dict()

    # число пассажиров > 30 лет
    adults_over_30_count = (data_frame["Age"] > 30).sum()

    # средний возраст для каждого Pclass
    mean_age_by_pclass = (
        data_frame.groupby("Pclass")["Age"]
        .mean()
        .to_dict()
    )

    # доля выживших для каждого Pclass
    survival_rate_by_pclass = (
        data_frame.groupby("Pclass")["Survived"]
        .mean()
        .to_dict()
    )

    # пять наибольших значений Fare
    highest_fares = data_frame["Fare"].nlargest(5).tolist()

    return TitanicSummary(
        row_count=row_count,
        missing_by_column=missing_by_column,
        adults_over_30_count=adults_over_30_count,
        mean_age_by_pclass=mean_age_by_pclass,
        survival_rate_by_pclass=survival_rate_by_pclass,
        highest_fares=highest_fares,
    )