# Laboratory practice 2.2: KNN classification
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme()
import numpy as np  
import seaborn as sns
from collections import Counter
from sklearn.metrics import roc_curve


def minkowski_distance(a, b, p=2):
    """
    Compute the Minkowski distance between two arrays.

    Args:
        a (np.ndarray): First array.
        b (np.ndarray): Second array.
        p (int, optional): The degree of the Minkowski distance. Defaults to 2 (Euclidean distance).

    Returns:
        float: Minkowski distance between arrays a and b.
    """
    return np.sum(np.abs(a - b) ** p) ** (1 / p)



# k-Nearest Neighbors Model

# - [K-Nearest Neighbours](https://scikit-learn.org/stable/modules/neighbors.html#classification)
# - [KNeighborsClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)


class knn:
    def __init__(self):
        self.k = None
        self.p = None
        self.x_train = None
        self.y_train = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, k: int = 5, p: int = 2):
        """
        Fit the model using X as training data and y as target values.

        You should check that all the arguments shall have valid values:
            X and y have the same number of rows.
            k is a positive integer.
            p is a positive integer.

        Args:
            X_train (np.ndarray): Training data.
            y_train (np.ndarray): Target values.
            k (int, optional): Number of neighbors to use. Defaults to 5.
            p (int, optional): The degree of the Minkowski distance. Defaults to 2.
        """
        if not isinstance(k, int) or k <= 0 or not isinstance(p, int) or p <= 0:
            raise ValueError("k and p must be positive integers.")
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError("Length of X_train and y_train must be equal.")
    
        self.x_train = X_train
        self.y_train = y_train
        self.k = k
        self.p = p

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the class labels for the provided data.

        Args:
            X (np.ndarray): data samples to predict their labels.

        Returns:
            np.ndarray: Predicted class labels.
        """
        return np.array([self._predict(x) for x in X])
    
    def _predict(self, x):
        """Predicts the class of a single sample x."""
        distances = self.compute_distances(x)
        k_indices = self.get_k_nearest_neighbors(distances)
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        return self.most_common_label(k_nearest_labels)

    def predict_proba(self, X):
        """
        Predict the class probabilities for the provided data.

        Each class probability is the amount of each label from the k nearest neighbors
        divided by k.

        Args:
            X (np.ndarray): data samples to predict their labels.

        Returns:
            np.ndarray: Predicted class probabilities.
        """
        probabilities = []
        
        # Iterar sobre todas las muestras de datos
        for x in X:
            distances = [minkowski_distance(x, x_train, self.p) for x_train in self.x_train]
            k_indices = np.argsort(distances)[:self.k]
            k_nearest_labels = [self.y_train[i] for i in k_indices]
            label_counts = Counter(k_nearest_labels)
            total = sum(label_counts.values())
            
            # Las probabilidades deben ser para cada clase (en el caso binario, 2 clases)
            # Creamos un arreglo con las probabilidades para cada clase (suponiendo 2 clases)
            proba = np.zeros(2)
            for label, count in label_counts.items():
                proba[label] = count / total
            
            probabilities.append(proba)

        # Convertimos a un arreglo numpy y lo retornamos
        return np.array(probabilities)

    def compute_distances(self, point: np.ndarray) -> np.ndarray:
        """Compute distance from a point to every point in the training dataset

        Args:
            point (np.ndarray): data sample.

        Returns:
            np.ndarray: distance from point to each point in the training dataset.
        """
        return np.array([minkowski_distance(point, x_train, self.p) for x_train in self.x_train])


    def get_k_nearest_neighbors(self, distances: np.ndarray) -> np.ndarray:
        """Get the k nearest neighbors indices given the distances matrix from a point.

        Args:
            distances (np.ndarray): distances matrix from a point whose neighbors want to be identified.

        Returns:
            np.ndarray: row indices from the k nearest neighbors.

        Hint:
            You might want to check the np.argsort function.
        """
        return np.argsort(distances)[:self.k]

    def most_common_label(self, knn_labels: np.ndarray) -> int:
        """Obtain the most common label from the labels of the k nearest neighbors

        Args:
            knn_labels (np.ndarray): labels from the k nearest neighbors

        Returns:
            int: most common label
        """
        return Counter(knn_labels).most_common(1)[0][0]

    def __str__(self):
        """
        String representation of the kNN model.
        """
        return f"kNN model (k={self.k}, p={self.p})"



def plot_2Dmodel_predictions(X, y, model, grid_points_n):
    """
    Plot the classification results and predicted probabilities of a model on a 2D grid.

    This function creates two plots:
    1. A classification results plot showing True Positives, False Positives, False Negatives, and True Negatives.
    2. A predicted probabilities plot showing the probability predictions with level curves for each 0.1 increment.

    Args:
        X (np.ndarray): The input data, a 2D array of shape (n_samples, 2), where each row represents a sample and each column represents a feature.
        y (np.ndarray): The true labels, a 1D array of length n_samples.
        model (classifier): A trained classification model with 'predict' and 'predict_proba' methods. The model should be compatible with the input data 'X'.
        grid_points_n (int): The number of points in the grid along each axis. This determines the resolution of the plots.

    Returns:
        None: This function does not return any value. It displays two plots.

    Note:
        - This function assumes binary classification and that the model's 'predict_proba' method returns probabilities for the positive class in the second column.
    """
    # Map string labels to numeric
    unique_labels = np.unique(y)
    num_to_label = {i: label for i, label in enumerate(unique_labels)}

    # Predict on input data
    preds = model.predict(X)

    # Determine TP, FP, FN, TN
    tp = (y == unique_labels[1]) & (preds == unique_labels[1])
    fp = (y == unique_labels[0]) & (preds == unique_labels[1])
    fn = (y == unique_labels[1]) & (preds == unique_labels[0])
    tn = (y == unique_labels[0]) & (preds == unique_labels[0])

    # Plotting
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # Classification Results Plot
    ax[0].scatter(X[tp, 0], X[tp, 1], color="green", label=f"True {num_to_label[1]}")
    ax[0].scatter(X[fp, 0], X[fp, 1], color="red", label=f"False {num_to_label[1]}")
    ax[0].scatter(X[fn, 0], X[fn, 1], color="blue", label=f"False {num_to_label[0]}")
    ax[0].scatter(X[tn, 0], X[tn, 1], color="orange", label=f"True {num_to_label[0]}")
    ax[0].set_title("Classification Results")
    ax[0].legend()

    # Create a mesh grid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, grid_points_n),
        np.linspace(y_min, y_max, grid_points_n),
    )

    # # Predict on mesh grid
    grid = np.c_[xx.ravel(), yy.ravel()]
    probs = model.predict_proba(grid)[:, 1].reshape(xx.shape)

    # Use Seaborn for the scatter plot
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, palette="Set1", ax=ax[1])
    ax[1].set_title("Classes and Estimated Probability Contour Lines")

    # Plot contour lines for probabilities
    cnt = ax[1].contour(xx, yy, probs, levels=np.arange(0, 1.1, 0.1), colors="black")
    ax[1].clabel(cnt, inline=True, fontsize=8)

    # Show the plot
    plt.tight_layout()
    plt.show()



def evaluate_classification_metrics(y_true, y_pred, positive_label):
    """
    Calculate various evaluation metrics for a classification model.

    Args:
        y_true (array-like): True labels of the data.
        positive_label: The label considered as the positive class.
        y_pred (array-like): Predicted labels by the model.

    Returns:
        dict: A dictionary containing various evaluation metrics.

    Metrics Calculated:
        - Confusion Matrix: [TN, FP, FN, TP]
        - Accuracy: (TP + TN) / (TP + TN + FP + FN)
        - Precision: TP / (TP + FP)
        - Recall (Sensitivity): TP / (TP + FN)
        - Specificity: TN / (TN + FP)
        - F1 Score: 2 * (Precision * Recall) / (Precision + Recall)
    """
    # Map string labels to 0 or 1
    y_true_mapped = np.array([1 if label == positive_label else 0 for label in y_true])
    y_pred_mapped = np.array([1 if label == positive_label else 0 for label in y_pred])

    # Confusion Matrix
    tp = np.sum((y_true_mapped == 1) & (y_pred_mapped == 1))
    tn = np.sum((y_true_mapped == 0) & (y_pred_mapped == 0))
    fp = np.sum((y_true_mapped == 0) & (y_pred_mapped == 1))
    fn = np.sum((y_true_mapped == 1) & (y_pred_mapped == 0))

    # Accuracy
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    # Precision
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0

    # Recall (Sensitivity)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0


    # Specificity
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    # F1 Score
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "Confusion Matrix": [tn, fp, fn, tp],
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "Specificity": specificity,
        "F1 Score": f1,
    }



def plot_calibration_curve(y_true, y_probs, positive_label, n_bins=10):
    """
    Plot a calibration curve to evaluate the accuracy of predicted probabilities.

    This function creates a plot that compares the mean predicted probabilities
    in each bin with the fraction of positives (true outcomes) in that bin.
    This helps assess how well the probabilities are calibrated.

    Args:
        y_true (array-like): True labels of the data. Can be binary or categorical.
        y_probs (array-like): Predicted probabilities for the positive class (positive_label).
                            Expected values are in the range [0, 1].
        positive_label (int or str): The label that is considered the positive class.
                                    This is used to map categorical labels to binary outcomes.
        n_bins (int, optional): Number of bins to use for grouping predicted probabilities.
                                Defaults to 10. Bins are equally spaced in the range [0, 1].

    Returns:
        dict: A dictionary with the following keys:
            - "bin_centers": Array of the center values of each bin.
            - "true_proportions": Array of the fraction of positives in each bin

    """
    # Create bins equally spaced between 0 and 1
    bins = np.linspace(0, 1, n_bins + 1)
    
    # Initialize lists to store the mean predicted probabilities and the true positive fractions
    bin_centers = []
    true_proportions = []
    
    # Loop through the bins
    for i in range(n_bins):
        # Define the lower and upper edges of the current bin
        bin_lower = bins[i]
        bin_upper = bins[i + 1]
        
        # Select the predicted probabilities that fall within this bin
        bin_mask = (y_probs >= bin_lower) & (y_probs < bin_upper)
        
        # Get the true labels for these predictions
        y_true_bin = y_true[bin_mask]
        
        if len(y_true_bin) == 0:
            continue  # Skip if the bin is empty
        
        # Calculate the fraction of positives (true positive rate) in this bin
        true_positive_rate = np.mean(y_true_bin == positive_label)
        
        # Calculate the mean predicted probability in this bin
        mean_predicted_prob = np.mean(y_probs[bin_mask])
        
        # Append the results
        bin_centers.append(mean_predicted_prob)
        true_proportions.append(true_positive_rate)
    
    # Convert lists to numpy arrays for consistency
    bin_centers = np.array(bin_centers)
    true_proportions = np.array(true_proportions)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # Optionally, plot the calibration curve
    plt.figure(figsize=(8, 6))
    plt.plot(bin_centers, true_proportions, marker='o', label="Calibration Curve")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Perfectly calibrated", color='gray')
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title("Calibration Curve")
    plt.legend()
    plt.show()

    return {"bin_centers": bin_centers, "true_proportions": true_proportions}



def plot_probability_histograms(y_true, y_probs, positive_label, n_bins=10):
    """
    Plot probability histograms for the positive and negative classes separately.

    This function creates two histograms showing the distribution of predicted
    probabilities for each class. This helps in understanding how the model
    differentiates between the classes.

    Args:
        y_true (array-like): True labels of the data. Can be binary or categorical.
        y_probs (array-like): Predicted probabilities for the positive class. 
                            Expected values are in the range [0, 1].
        positive_label (int or str): The label considered as the positive class.
                                    Used to map categorical labels to binary outcomes.
        n_bins (int, optional): Number of bins for the histograms. Defaults to 10. 
                                Bins are equally spaced in the range [0, 1].

    Returns:
        dict: A dictionary with the following keys:
            - "array_passed_to_histogram_of_positive_class": 
                Array of predicted probabilities for the positive class.
            - "array_passed_to_histogram_of_negative_class": 
                Array of predicted probabilities for the negative class.

    """
    # Map y_true to binary labels (1 for positive class and 0 for negative class)
    y_true_mapped = (y_true == positive_label).astype(int)
    
    # Separate the probabilities for the positive and negative classes
    positive_class_probs = y_probs[y_true_mapped == 1]
    negative_class_probs = y_probs[y_true_mapped == 0]
    
    # Plot histograms for both classes
    plt.figure(figsize=(8, 6))
    
    # Plot for positive class
    plt.hist(positive_class_probs, bins=n_bins, alpha=0.5, label="Positive class", color='green', edgecolor='black')
    
    # Plot for negative class
    plt.hist(negative_class_probs, bins=n_bins, alpha=0.5, label="Negative class", color='red', edgecolor='black')
    
    # Add labels and title
    plt.xlabel("Predicted Probability")
    plt.ylabel("Frequency")
    plt.title("Probability Distributions for Positive and Negative Classes")
    plt.legend()
    
    # Show the plot
    plt.show()

    return {
        "array_passed_to_histogram_of_positive_class": y_probs[y_true_mapped == 1],
        "array_passed_to_histogram_of_negative_class": y_probs[y_true_mapped == 0],
    }



def plot_roc_curve(y_true, y_probs, positive_label):
    """
    Plot the Receiver Operating Characteristic (ROC) curve.

    The ROC curve is a graphical representation of the diagnostic ability of a binary
    classifier system as its discrimination threshold is varied. It plots the True Positive
    Rate (TPR) against the False Positive Rate (FPR) at various threshold settings.

    Args:
        y_true (array-like): True labels of the data. Can be binary or categorical.
        y_probs (array-like): Predicted probabilities for the positive class. 
                            Expected values are in the range [0, 1].
        positive_label (int or str): The label considered as the positive class.
                                    Used to map categorical labels to binary outcomes.

    Returns:
        dict: A dictionary containing the following:
            - "fpr": Array of False Positive Rates for each threshold.
            - "tpr": Array of True Positive Rates for each threshold.

    """
    # Convert y_true to binary labels (1 for positive class and 0 for negative class)
    y_true_mapped = (y_true == positive_label).astype(int)
    
    # Calculate FPR and TPR for various thresholds
    fpr, tpr, thresholds = roc_curve(y_true_mapped, y_probs)
    
    # Ensure exactly 11 thresholds by using np.linspace
    custom_thresholds = np.linspace(0, 1, 11)
    
    # Calculate FPR and TPR for the custom thresholds
    fpr_custom = []
    tpr_custom = []
    
    for thresh in custom_thresholds:
        # Get the predicted labels based on the threshold
        y_pred = (y_probs >= thresh).astype(int)
        
        # Calculate the confusion matrix components
        tp = np.sum((y_true_mapped == 1) & (y_pred == 1))
        fp = np.sum((y_true_mapped == 0) & (y_pred == 1))
        fn = np.sum((y_true_mapped == 1) & (y_pred == 0))
        tn = np.sum((y_true_mapped == 0) & (y_pred == 0))

        # Calculate TPR and FPR for this threshold
        tpr_custom.append(tp / (tp + fn) if tp + fn != 0 else 0)
        fpr_custom.append(fp / (fp + tn) if fp + tn != 0 else 0)

    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_custom, tpr_custom, color='blue', label="ROC Curve")
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label="Random classifier")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.show()
    
    return {"fpr": np.array(fpr_custom), "tpr": np.array(tpr_custom)}
