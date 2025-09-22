import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# =========================
# Funktioner
# =========================

def load_data(filename):
    """
    Laddar datapunkter: width, height, label
    """
    return np.loadtxt(filename, delimiter=',', skiprows=1)

def load_testpoints(filename):
    """
    Läser in testpunkter i formatet:
    (25, 32)
    (24.2, 31.5)
    ...
    """
    points = []
    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("(") and line.endswith(")"):
                nums = line[1:-1].split(",")
                width, height = float(nums[0]), float(nums[1])
                points.append(np.array([width, height]))
    return points

def classify_knn(test_point, training_data, k=1):
    """
    Klassificerar testpunkt med k-NN.
    0 = Pichu, 1 = Pikachu
    """
    distances = np.linalg.norm(training_data[:, :2] - test_point, axis=1)
    k_indices = np.argpartition(distances, k)[:k]
    k_labels = training_data[k_indices, 2].astype(int)
    return Counter(k_labels).most_common(1)[0][0]

def split_data_balanced(data, n_train_per_class=50):
    """
    Splittar datan i balanserad tränings- och testdata.
    50 per klass i träning, resten i test.
    """
    pichu = data[data[:, 2] == 0]
    pikachu = data[data[:, 2] == 1]

    np.random.shuffle(pichu)
    np.random.shuffle(pikachu)

    train = np.vstack((pichu[:n_train_per_class], pikachu[:n_train_per_class]))
    test = np.vstack((pichu[n_train_per_class:], pikachu[n_train_per_class:]))

    np.random.shuffle(train)
    np.random.shuffle(test)
    return train, test

def calculate_accuracy(predictions, true_labels):
    """
    Beräknar accuracy och confusion matrix.
    """
    correct = np.sum(predictions == true_labels)
    acc = correct / len(true_labels)

    tp = np.sum((predictions == 1) & (true_labels == 1))
    tn = np.sum((predictions == 0) & (true_labels == 0))
    fp = np.sum((predictions == 1) & (true_labels == 0))
    fn = np.sum((predictions == 0) & (true_labels == 1))

    confusion = {"TP": tp, "TN": tn, "FP": fp, "FN": fn}
    return acc, confusion

def run_evaluation(data, k=10, n_runs=10):
    """
    Kör bonusutvärdering: splitta, klassificera, räkna accuracy flera gånger.
    """
    accuracies = []
    for i in range(n_runs):
        train, test = split_data_balanced(data, n_train_per_class=50)
        preds = np.array([classify_knn(p[:2], train, k) for p in test])
        acc, conf = calculate_accuracy(preds, test[:, 2])
        accuracies.append(acc)
        print(f"Körning {i+1}: Accuracy = {acc:.3f}, Confusion = {conf}")
    return accuracies

def plot_accuracies(accuracies):
    """
    Plottar accuracy per körning + medelaccuracy.
    """
    runs = np.arange(1, len(accuracies) + 1)
    mean_acc = np.mean(accuracies)

    plt.figure(figsize=(10, 6))
    plt.bar(runs, accuracies, color="skyblue", edgecolor="black")
    plt.axhline(mean_acc, color="red", linestyle="--", label=f"Medel = {mean_acc:.3f}")
    plt.ylim(0, 1)
    plt.xlabel("Körning")
    plt.ylabel("Accuracy")
    plt.title("k-NN Utvärdering (10 körningar)")
    plt.legend()
    plt.show()

# =========================
# Huvudprogram
# =========================

def main():
    # Ladda data
    data = load_data("datapoints.txt")
    test_points = load_testpoints("testpoints.txt")

    print("Antal datapunkter i träningsdata:", len(data))
    print("Testpunkter:", test_points)

    # Grunduppgift: 1-NN
    print("\n=== Klassificering med 1-NN ===")
    facit = ["Pikachu", "Pikachu", "Pikachu", "Pichu"]
    for i, p in enumerate(test_points):
        label = classify_knn(p, data, k=1)
        name = "Pikachu" if label else "Pichu"
        print(f"Punkt {tuple(p)} → {name} | Facit: {facit[i]} | {'✅' if name==facit[i] else '❌'}")

    # Uppgift 2: 10-NN
    print("\n=== Klassificering med 10-NN ===")
    for p in test_points:
        label = classify_knn(p, data, k=10)
        name = "Pikachu" if label else "Pichu"
        print(f"Punkt {tuple(p)} → {name}")

    # Bonus: utvärdering
    print("\n=== Bonusutvärdering (10 körningar) ===")
    accuracies = run_evaluation(data, k=10, n_runs=10)
    print("\nMedelaccuracy:", np.mean(accuracies))
    plot_accuracies(accuracies)

if __name__ == "__main__":
    main()
