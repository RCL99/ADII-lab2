"""Задачи первой части лабораторной: Pandas и Titanic."""
from __future__ import annotations

import pandas as pd

from grader_contracts.pandas_tasks import TitanicInput, TitanicSummary


def analyze_titanic(data: TitanicInput) -> TitanicSummary:
    """Выполните загрузку и анализ датасета Titanic.

    Нужно: посчитать пропуски, число пассажиров старше 30 лет, средний возраст
    и долю выживших по классам, а также пять наибольших тарифов по убыванию.
    """

    # датафрейм
    titanic_df = pd.read_csv(data.csv_path)

    # общее число строк
    row_count = len(titanic_df)

    # кол-во пропусков в каждом столбце
    missing_count_by_column = titanic_df.isnull().sum()
    missing_by_column = {
        column_name: int(missing_count)
        for column_name, missing_count in missing_count_by_column.items()
    }

    # число пассажиров старше 30 лет (NaN в Age дают False при сравнении)
    adults_over_30_count = int((titanic_df["Age"] > 30).sum())

    # средний возраст для каждого значения Pclass (NaN игнорируются в mean)
    mean_age_by_pclass = {
        int(pclass_value): float(mean_age)
        for pclass_value, mean_age in titanic_df.groupby("Pclass")["Age"].mean().items()
    }

    # доля выживших для каждого Pclass
    survival_rate_by_pclass = {
        int(pclass_value): float(survival_rate)
        for pclass_value, survival_rate in titanic_df.groupby("Pclass")["Survived"].mean().items()
    }

    # пять наибольших значений Fare в порядке убывания
    highest_fares = [float(fare_value) for fare_value in titanic_df["Fare"].nlargest(5).tolist()]

    return TitanicSummary(
        row_count=row_count,
        missing_by_column=missing_by_column,
        adults_over_30_count=adults_over_30_count,
        mean_age_by_pclass=mean_age_by_pclass,
        survival_rate_by_pclass=survival_rate_by_pclass,
        highest_fares=highest_fares,
    )