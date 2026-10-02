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

    # кол-во пропусков в каждом столбце
    missing_count_by_column = titanic_df.isnull().sum()

    # число пассажиров > 30 лет
    passengers_over_30_count = (titanic_df["Age"] > 30).sum()

    # вычислить средний возраст для каждого значения Pclass
    mean_age_by_pclass = titanic_df.groupby("Pclass")["Age"].mean()

    # вычислить долю выживших для каждого Pclass
    survival_rate_by_pclass = titanic_df.groupby("Pclass")["Survived"].mean()  # [1+0+1+1+0...].mean

    # вернуть пять наибольших значений Fare в порядке убывания
    highest_fares = titanic_df["Fare"].nlargest(5).tolist()
    print(highest_fares)

if __name__ == "__main__":
    analyze_titanic(TitanicInput(csv_path="titanic.csv"))