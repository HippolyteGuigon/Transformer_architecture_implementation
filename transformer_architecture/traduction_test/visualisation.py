import os
import json
import matplotlib.pyplot as plt
from typing import List, Optional

def get_metric_plots(json_path: Optional[str] = "metrics/metrics_epochs.json") -> None:
    """
    Generate and save metric plots for a given model run.
    
    Args:
        json_path (str): Path to the JSON file where metrics are saved.
    
    Returns:
        None
    """
    
    with open(json_path, "r") as f:
        metrics = json.load(f)
    
    metric_data = {
        "Training Loss": [d["train_loss"] for d in metrics.values()],
        "Validation Loss": [d["val_loss"] for d in metrics.values()],
        "BLEU Score": [d["bleu_score"] for d in metrics.values()],
        "ROUGE-1 Score": [d["rouge1_score"] for d in metrics.values()],
        "ROUGE-L Score": [d["rougeL_score"] for d in metrics.values()],
    }
    
    os.makedirs("graphs", exist_ok=True)
    
    def save_plot(values: List, title: str, filename: str)->None:
        """
        The goal of this function is, once
        the metrics were produced to create
        and save a graph accordingly
        
        Arguments:
            -values: List: The produced metrics
            -title: str: The title to be given
            to the graph
            -filename: str: The file in which the
            graph should be saved
        """
        
        plt.figure()
        plt.plot(values, marker='o')
        plt.title(title)
        plt.xlabel("Epoch")
        plt.ylabel(title)
        plt.grid(True)
        plt.savefig(filename)
        plt.close()
    
    for title, values in metric_data.items():
        filename = f"graphs/{title.lower().replace(' ', '_')}.png"
        save_plot(values, title, filename)

if __name__=="__main__":
    get_metric_plots()