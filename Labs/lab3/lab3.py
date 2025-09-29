import numpy as np
import matplotlib.pyplot as plt

def classify_point(x, y, k=1.0, m=0.0):
    
    return int(y >= k * x + m)

def main():
    try:
        # Läs in data
        data = np.loadtxt("unlabelled_data.csv", delimiter=",")
        print(f"Successfully loaded data with shape: {data.shape}")
    except FileNotFoundError:
        print("Error: unlabelled_data.csv not found in current directory")
        return
    except ValueError as e:
        print(f"Error reading CSV file: {e}")
        return
    
    # Check data format
    if data.shape[1] != 2:
        print(f"Error: Expected 2 columns, but got {data.shape[1]}")
        return
    
    x = data[:, 0]
    y = data[:, 1]
    
    # Klassificera med linjen y = x (k=1, m=0) - vectorized approach
    labels = (y >= 1.0 * x + 0.0).astype(int)
    
    try:
        # Spara till labelled_data.csv
        labelled = np.column_stack((x, y, labels))
        np.savetxt("labelled_data.csv", labelled, delimiter=",", fmt=["%.10f", "%.10f", "%d"])
        print("Successfully saved labeled data to labelled_data.csv")
    except Exception as e:
        print(f"Error saving file: {e}")
        return
    
    # Plotta
    plt.figure(figsize=(8, 6))
    colors = ['red', 'blue']
    for label in [0, 1]:
        mask = labels == label
        plt.scatter(x[mask], y[mask], label=f"Klass {label}", color=colors[label], s=20)
    
    # Rita linjen y = x
    x_range = x.max() - x.min()
    y_range = y.max() - y.min()
    padding = max(x_range, y_range) * 0.1  # Dynamic padding based on data range
    
    line_x = np.linspace(x.min() - padding, x.max() + padding, 100)
    line_y = 1.0 * line_x + 0.0
    plt.plot(line_x, line_y, 'k--', linewidth=2, label="y = x")
    
    # Set plot limits to ensure good visualization
    plt.xlim(x.min() - padding, x.max() + padding)
    plt.ylim(y.min() - padding, y.max() + padding)
    
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Klassificerad data med beslutsgräns y = x")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Print classification summary
    n_class_0 = np.sum(labels == 0)
    n_class_1 = np.sum(labels == 1)
    print(f"Classification summary: {n_class_0} points in class 0, {n_class_1} points in class 1")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()