import json
import matplotlib.pyplot as plt

def plot_single_model_metrics(json_path, save_path=None):
    """
    Visualisiert F1-Score, Precision, Recall und Accuracy für ein einzelnes Modell.
    Die Werte werden als farbige Balken in einem Diagramm dargestellt.
    """

    # JSON laden
    with open(json_path, "r") as f:
        data = json.load(f)

    # Werte extrahieren
    model_label = f"{data['model']} ({data['model_version']})"
    metrics = {
        "Precision": data["precision"],
        "Recall": data["recall"],
        "F1-Score": data["f1_score"],
        "Accuracy": data["accuracy"]
    }

    labels = list(metrics.keys())
    values = list(metrics.values())
    colors = ['cornflowerblue', 'salmon', 'mediumseagreen', 'goldenrod']

    # Plot erstellen
    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, values, color=colors)
    plt.ylim(0.8, 1.05)
    plt.title(f"Leistungsmetriken für {model_label}")
    plt.ylabel("Wert")

    # Balken beschriften
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f"{value:.3f}", ha='center', va='bottom')

    plt.tight_layout()

    # Optional speichern
    if save_path:
        plt.savefig(save_path)
        print(f"💾 Diagramm gespeichert unter: {save_path}")

    plt.show()

# Beispiel-Aufruf
if __name__ == "__main__":
    plot_single_model_metrics("results/metrics_albert-base-v2.json", save_path="results/single_model_metrics.png")
