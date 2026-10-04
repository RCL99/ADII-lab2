import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from correlation_tasks import analyze_brain_correlations
from grader_contracts.correlation_tasks import BrainCorrelationSummary, BrainDataInput


def BrainCorrelationGraph(data: BrainCorrelationSummary):
    """хитмап корреляций"""
    # датафрейм для хитмапа
    correlations = {
        "Male": data.men_mri_correlation,
        "Female": data.women_mri_correlation,
    }
    correlation_frame = pd.DataFrame(correlations)

    # хитмап
    sns.heatmap(
        correlation_frame,
        annot=True,
        cmap="viridis",
        center=0,
    )

    # подписи
    plt.title("Корреляция признаков с MRI count")
    plt.xlabel("Пол")
    plt.ylabel("Признак")
    plt.savefig("correlation_graph.png")
    plt.show()

if __name__ == "__main__":
    summary = analyze_brain_correlations(BrainDataInput(csv_path="brainsize.txt"))
    print(summary.strongest_mri_feature)
    BrainCorrelationGraph(summary)