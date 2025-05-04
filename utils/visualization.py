import os
import json
import matplotlib.pyplot as plt


def plot_f1_scores_from_results(results_folder="results", save_path=None):
    """
    Lädt alle metrics_*.json-Dateien im angegebenen Ordner und visualisiert
    die F1-Scores sortiert nach Höhe.

    :param results_folder: Ordner mit metrics_*.json-Dateien
    :param save_path: Optionaler Pfad, um das Diagramm als PNG zu speichern
    """
    entries = []

    # Alle Metrik-Dateien einlesen
    for filename in os.listdir(results_folder):
        if filename.startswith("metrics_") and filename.endswith(".json"):
            with open(os.path.join(results_folder, filename), "r") as f:
                data = json.load(f)
                label = f"{data['model']} {data['model_version']}"
                f1 = data["f1_score"]
                entries.append((label, f1))

    if not entries:
        print("⚠️ Keine Ergebnisse gefunden.")
        return

    # Sortieren nach F1 absteigend
    entries.sort(key=lambda x: x[1], reverse=True)
    labels, f1_scores = zip(*entries)

    # Balkendiagramm zeichnen
    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, f1_scores, color='skyblue')
    plt.ylim(0.85, 1.0)
    plt.ylabel("F1-Score")
    plt.title("Modellvergleich: F1-Score (sortiert)")

    for bar, f1 in zip(bars, f1_scores):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f"{f1:.3f}", ha='center', va='bottom')

    plt.xticks(rotation=45)
    plt.tight_layout()

    # Optional speichern
    if save_path:
        plt.savefig(save_path)
        print(f"💾 Diagramm gespeichert unter: {save_path}")

    plt.show()


if __name__ == "__main__":
    # Beispiel: Diagramm anzeigen & speichern
    plot_f1_scores_from_results(save_path="results/f1_comparison.png")
