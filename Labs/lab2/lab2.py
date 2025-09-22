import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import random

def read_data(filename):
    """
    Read Pokémon data from a file and parse into features and labels.
    
    Parameters:
    filename (str): Path to the data file
    
    Returns:
    tuple: (widths, heights, labels) as numpy arrays
    """
    widths = []
    heights = []
    labels = []
    
    try:
        with open(filename, 'r') as file:
            # Skip the header line
            next(file)
            for line in file:
                if line.strip():  # Skip empty lines
                    parts = line.strip().split(',')
                    widths.append(float(parts[0]))
                    heights.append(float(parts[1]))
                    labels.append(int(parts[2]))
        
        return np.array(widths), np.array(heights), np.array(labels)
    
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return None, None, None
    except Exception as e:
        print(f"Error reading file: {e}")
        return None, None, None

def split_data(widths, heights, labels, test_size=50, random_state=None):
    """
    Split data into training and testing sets while maintaining class balance.
    
    Parameters:
    widths, heights, labels: Input data
    test_size: Number of samples to use for testing (per class)
    random_state: Seed for reproducible results
    
    Returns:
    tuple: (train_widths, train_heights, train_labels, test_widths, test_heights, test_labels)
    """
    if random_state is not None:
        random.seed(random_state)
        np.random.seed(random_state)
    
    # Separate data by class
    pichu_indices = np.where(labels == 0)[0]
    pikachu_indices = np.where(labels == 1)[0]
    
    # Randomly select test samples for each class
    pichu_test_indices = random.sample(list(pichu_indices), min(test_size, len(pichu_indices)))
    pikachu_test_indices = random.sample(list(pikachu_indices), min(test_size, len(pikachu_indices)))
    
    # Create test set
    test_indices = pichu_test_indices + pikachu_test_indices
    test_widths = widths[test_indices]
    test_heights = heights[test_indices]
    test_labels = labels[test_indices]
    
    # Create training set (all samples not in test set)
    train_mask = np.ones(len(widths), dtype=bool)
    train_mask[test_indices] = False
    train_widths = widths[train_mask]
    train_heights = heights[train_mask]
    train_labels = labels[train_mask]
    
    # Combine features for easier processing
    train_data = np.column_stack((train_widths, train_heights))
    test_data = np.column_stack((test_widths, test_heights))
    
    return train_data, test_data, train_labels, test_labels

def euclidean_distance(point1, point2):
    """
    Calculate Euclidean distance between two points.
    
    Parameters:
    point1, point2: Arrays representing points in n-dimensional space
    
    Returns:
    float: Euclidean distance between the points
    """
    return np.sqrt(np.sum((point1 - point2) ** 2))

def classify_knn(training_data, training_labels, test_point, k=1):
    """
    Classify a test point using k-Nearest Neighbors algorithm.
    
    Parameters:
    training_data: Array of training points (width, height)
    training_labels: Array of corresponding labels
    test_point: Point to classify (width, height)
    k: Number of nearest neighbors to consider
    
    Returns:
    int: Predicted label (0 for Pichu, 1 for Pikachu)
    """
    # Calculate distances to all training points using vectorization for efficiency
    distances = np.sqrt(np.sum((training_data - test_point) ** 2, axis=1))
    
    # Get indices of k nearest neighbors
    k_indices = np.argpartition(distances, k)[:k]
    
    # Get labels of k nearest neighbors
    k_nearest_labels = training_labels[k_indices]
    
    # Return the most common class label
    most_common = Counter(k_nearest_labels).most_common(1)
    return most_common[0][0]

def evaluate_model(training_data, training_labels, test_data, test_labels, k=10):
    """
    Evaluate the k-NN model using test data.
    
    Parameters:
    training_data, training_labels: Training set
    test_data, test_labels: Test set for evaluation
    k: Number of neighbors to consider
    
    Returns:
    tuple: (accuracy, confusion_matrix)
    """
    predictions = []
    
    # Classify each test point
    for point in test_data:
        label = classify_knn(training_data, training_labels, point, k)
        predictions.append(label)
    
    predictions = np.array(predictions)
    
    # Calculate confusion matrix
    tp = np.sum((predictions == 1) & (test_labels == 1))  # True Positives (Pikachu)
    tn = np.sum((predictions == 0) & (test_labels == 0))  # True Negatives (Pichu)
    fp = np.sum((predictions == 1) & (test_labels == 0))  # False Positives
    fn = np.sum((predictions == 0) & (test_labels == 1))  # False Negatives
    
    confusion_matrix = {
        'TP': tp, 'TN': tn,
        'FP': fp, 'FN': fn
    }
    
    # Calculate accuracy
    accuracy = (tp + tn) / len(test_labels)
    
    return accuracy, confusion_matrix

def plot_data(widths, heights, labels, test_points=None, test_labels=None, title="Pokémon Classification"):
    """
    Visualize the training data and optional test points.
    
    Parameters:
    widths, heights, labels: Training data
    test_points, test_labels: Optional test points to highlight
    title: Plot title
    """
    plt.figure(figsize=(12, 8))
    
    # Plot training data
    pichu_idx = labels == 0
    pikachu_idx = labels == 1
    
    plt.scatter(widths[pichu_idx], heights[pichu_idx], 
                color='blue', label='Pichu (0)', alpha=0.6, s=50)
    plt.scatter(widths[pikachu_idx], heights[pikachu_idx], 
                color='red', label='Pikachu (1)', alpha=0.6, s=50)
    
    # Plot test points if provided
    if test_points is not None and test_labels is not None:
        test_points = np.array(test_points)
        test_labels = np.array(test_labels)
        test_pichu_idx = test_labels == 0
        test_pikachu_idx = test_labels == 1
        
        if np.any(test_pichu_idx):
            plt.scatter(test_points[test_pichu_idx, 0], test_points[test_pichu_idx, 1],
                       color='purple', marker='s', s=100, label='Test Pichu', edgecolors='black')
        if np.any(test_pikachu_idx):
            plt.scatter(test_points[test_pikachu_idx, 0], test_points[test_pikachu_idx, 1],
                       color='green', marker='^', s=100, label='Test Pikachu', edgecolors='black')
    
    plt.xlabel('Width (cm)')
    plt.ylabel('Height (cm)')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def run_cross_validation(widths, heights, labels, k=10, n_runs=10, test_size=50):
    """
    Run repeated cross-validation to evaluate model performance.
    
    Parameters:
    widths, heights, labels: Full dataset
    k: Number of neighbors for k-NN
    n_runs: Number of validation runs
    test_size: Number of test samples per class
    
    Returns:
    tuple: (accuracies, mean_accuracy)
    """
    accuracies = []
    
    for i in range(n_runs):
        # Split data with different random state each time
        train_data, test_data, train_labels, test_labels = split_data(
            widths, heights, labels, test_size=test_size, random_state=i
        )
        
        # Evaluate model
        accuracy, _ = evaluate_model(train_data, train_labels, test_data, test_labels, k)
        accuracies.append(accuracy)
    
    mean_accuracy = np.mean(accuracies)
    return accuracies, mean_accuracy

def plot_accuracy_results(accuracies, mean_accuracy):
    """
    Plot accuracy results from cross-validation.
    
    Parameters:
    accuracies: List of accuracy values from each run
    mean_accuracy: Mean accuracy across all runs
    """
    plt.figure(figsize=(10, 6))
    runs = range(1, len(accuracies) + 1)
    plt.plot(runs, accuracies, 'o-', label='Accuracy per run')
    plt.axhline(y=mean_accuracy, color='r', linestyle='--', label=f'Mean accuracy: {mean_accuracy:.3f}')
    plt.xlabel('Run')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy Across Multiple Runs')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.show()

def interactive_classification(training_data, training_labels, k=10):
    """
    Allow user to interactively classify points.
    
    Parameters:
    training_data, training_labels: Training set
    k: Number of neighbors to consider
    """
    print("\nInteractive Classification Mode (enter 'quit' to exit)")
    print("Enter points as 'width,height' (e.g., '23.5,32.1')")
    
    while True:
        user_input = input("\nEnter a point: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            break
        
        try:
            # Parse input
            width, height = map(float, user_input.split(','))
            
            # Validate input
            if width <= 0 or height <= 0:
                print("Error: Width and height must be positive numbers.")
                continue
            
            # Classify point
            point = np.array([width, height])
            label = classify_knn(training_data, training_labels, point, k)
            pokemon = "Pikachu" if label == 1 else "Pichu"
            
            print(f"Point ({width}, {height}) is classified as {pokemon}")
            
        except ValueError:
            print("Error: Please enter two numbers separated by a comma (e.g., '23.5,32.1').")
        except Exception as e:
            print(f"An error occurred: {e}")

def main():
    """
    Main function to run the Pokémon classification program.
    """
    print("Pokémon Classifier: Pichu vs Pikachu")
    print("====================================")
    
    # Read data
    widths, heights, labels = read_data('datapoints.txt')
    
    if widths is None:
        return  # Exit if data loading failed
    
    # Original test points from assignment
    test_points = [
        [25, 32],
        [24.2, 31.5],
        [22, 34],
        [20.5, 34]
    ]
    
    # Convert to numpy array for easier manipulation
    training_data = np.column_stack((widths, heights))
    
    # 1. Classify with k=1 (nearest neighbor)
    print("\n1. Classification with k=1 (nearest neighbor):")
    test_labels_k1 = []
    for point in test_points:
        label = classify_knn(training_data, labels, point, k=1)
        test_labels_k1.append(label)
        pokemon = "Pikachu" if label == 1 else "Pichu"
        print(f"Point {tuple(point)} classified as {pokemon}")
    
    # Visualize results
    plot_data(widths, heights, labels, test_points, test_labels_k1, 
             "Classification with k=1 (Nearest Neighbor)")
    
    # 2. Classify with k=10 (10 nearest neighbors)
    print("\n2. Classification with k=10 (10 nearest neighbors):")
    test_labels_k10 = []
    for point in test_points:
        label = classify_knn(training_data, labels, point, k=10)
        test_labels_k10.append(label)
        pokemon = "Pikachu" if label == 1 else "Pichu"
        print(f"Point {tuple(point)} classified as {pokemon}")
    
    # Visualize results
    plot_data(widths, heights, labels, test_points, test_labels_k10, 
             "Classification with k=10 (10 Nearest Neighbors)")
    
    # 3. Split data into training and test sets (100 training, 50 test)
    print("\n3. Splitting data into training and test sets...")
    train_data, test_data, train_labels, test_labels = split_data(
        widths, heights, labels, test_size=25, random_state=42
    )
    
    print(f"Training set: {len(train_data)} samples")
    print(f"Test set: {len(test_data)} samples")
    
    # 4. Evaluate model performance
    print("\n4. Evaluating model performance on test set...")
    accuracy, confusion_matrix = evaluate_model(train_data, train_labels, test_data, test_labels, k=10)
    
    print(f"Accuracy: {accuracy:.3f}")
    print("Confusion Matrix:")
    print(f"True Positives (Pikachu correctly identified): {confusion_matrix['TP']}")
    print(f"True Negatives (Pichu correctly identified): {confusion_matrix['TN']}")
    print(f"False Positives (Pichu misclassified as Pikachu): {confusion_matrix['FP']}")
    print(f"False Negatives (Pikachu misclassified as Pichu): {confusion_matrix['FN']}")
    
    # Bonus: Repeated cross-validation
    print("\n5. Running repeated cross-validation (10 runs)...")
    accuracies, mean_accuracy = run_cross_validation(widths, heights, labels, k=10, n_runs=10, test_size=25)
    
    print("Accuracies per run:", [f"{acc:.3f}" for acc in accuracies])
    print(f"Mean accuracy: {mean_accuracy:.3f}")
    
    # Plot accuracy results
    plot_accuracy_results(accuracies, mean_accuracy)
    
    # Interactive classification
    interactive_classification(training_data, labels, k=10)
    
    print("\nProgram completed. Thank you for using the Pokémon Classifier!")

if __name__ == "__main__":
    main()