from matplotlib import pyplot as plt
import numpy as np

def visualize_analysis_results(results):
    """
    Visualizes the skin analysis results using charts and graphs.

    Parameters:
    results (dict): A dictionary containing analysis results such as metrics and scores.

    Returns:
    None
    """
    metrics = results.get('metrics', {})
    scores = results.get('scores', {})

    # Create subplots for metrics and scores
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))

    # Plot metrics
    axs[0].bar(metrics.keys(), metrics.values(), color='skyblue')
    axs[0].set_title('Skin Metrics')
    axs[0].set_ylabel('Values')
    axs[0].set_xticklabels(metrics.keys(), rotation=45)

    # Plot scores
    axs[1].bar(scores.keys(), scores.values(), color='salmon')
    axs[1].set_title('Skin Scores')
    axs[1].set_ylabel('Scores')
    axs[1].set_xticklabels(scores.keys(), rotation=45)

    plt.tight_layout()
    plt.show()