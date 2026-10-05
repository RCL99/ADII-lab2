"""Задачи второй части лабораторной: корреляционный анализ."""
from __future__ import annotations

from grader_contracts.correlation_tasks import BrainCorrelationSummary, BrainDataInput

import pandas as pd


def analyze_brain_correlations(data: BrainDataInput) -> BrainCorrelationSummary:
    """Проанализируйте brainsize.txt.

    Разделите наблюдения по полу и для каждой группы вычислите корреляции
    признаков FSIQ, VIQ, PIQ, Weight, Height с MRI_Count методом Пирсона.
    В strongest_mri_feature верните название признака с наибольшим модулем
    корреляции с MRI_Count среди объединённых результатов двух групп.
    """

    data_frame = pd.read_csv(
        data.csv_path,
        sep="\t",
        na_values="NA",
    )

    features = ["FSIQ", "VIQ", "PIQ", "Weight", "Height"]

    groups = data_frame.groupby("Gender")

    male = groups["Male"]
    female = groups["Female"]

    # корреляция каждой фичи с mri count
    male_corr = male[features].corrwith(male["MRI_Count"], method="pearson")
    female_corr = female[features].corrwith(female["MRI_Count"], method="pearson")

    general_corr = pd.concat([male_corr, female_corr])

    # самое макисмальное значение корреляции по модулю
    strongest_mri_feature = str(general_corr.abs().idxmax())
    
    return BrainCorrelationSummary(
        men_count=len(male),
        women_count=len(female),
        women_mri_correlation=male_corr,
        men_mri_correlation=female_corr,
        strongest_mri_feature=strongest_mri_feature,
    )