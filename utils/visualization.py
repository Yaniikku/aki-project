import os
import json
import matplotlib.pyplot as plt

def plot_metrics_separately(results_folder="results", save_prefix="results/metric_"):
    """
    Lädt alle metrics_*.json-Dateien und erzeugt drei getrennte Diagramme:
    - F1-Score
    - Precision
    - Recall
    Die Y-Achse beginnt bei 0.8 für bessere Unterscheidung.
    """

    entries = []

    # Dateien laden
    for filename in os.listdir(results_folder):
        if filename.startswith("metrics_") and filename.endswith(".json"):
            with open(os.path.join(results_folder, filename), "r") as f:
                data = json.load(f)
                print(f"✅ Datei geladen: {filename}")
                print(f"🔹 Modell: {data.get('model')} {data.get('model_version')}")
                print(f"🔹 F1: {data.get('f1_score')}, Precision: {data.get('precision')}, Recall: {data.get('recall')}")
                label = f"{data['model']} {data['model_version']}"
                entries.append({
                    "label": label,
                    "f1": data["f1_score"],
                    "precision": data["precision"],
                    "recall": data["recall"]
                })

    if not entries:
        print("⚠️ Keine Ergebnisse gefunden.")
        return

    # Drei Diagramme erzeugen
    for metric in ["f1", "precision", "recall"]:
        sorted_entries = sorted(entries, key=lambda x: x[metric], reverse=True)
        labels = [entry["label"] for entry in sorted_entries]
        values = [entry[metric] for entry in sorted_entries]

        plt.figure(figsize=(10, 5))
        bars = plt.bar(labels, values, color='skyblue')
        plt.ylim(0.8, 1.05)  # Y-Achse bei 0.8 starten
        plt.ylabel(metric.capitalize())
        plt.title(f"Modellvergleich: {metric.capitalize()}")
        plt.xticks(rotation=45, ha='right')

        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                     f"{value:.3f}", ha='center', va='bottom')

        plt.tight_layout()

        file_path = f"{save_prefix}{metric}.png"
        plt.savefig(file_path)
        print(f"💾 Diagramm gespeichert unter: {file_path}")
        plt.show()


if __name__ == "__main__":
    plot_metrics_separately()
