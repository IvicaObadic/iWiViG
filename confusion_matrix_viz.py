import os.path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

def load_class_accuracy_per_model(filepath):
    # Load the confusion matrix from the CSV file
    predictions = pd.read_csv(filepath, index_col=0)
    conf_matr = confusion_matrix(predictions["labels"], predictions["preds"])

    class_accuracy = conf_matr.diagonal() / conf_matr.sum(axis=1)
    return class_accuracy

def visualize_confusion_matrix(results_base_dir, conf_matrix_file):
    """Visualizes a large confusion matrix from a CSV file and optionally saves it as a plot.

    Args:
        confusion_matrix_file (str): Path to the CSV file containing the confusion matrix.
        save_path (str, optional): Path to save the plot. If None, the plot is displayed but not saved.
    """
    confusion_matrix_file = os.path.join(results_base_dir, conf_matrix_file)
    # Load the confusion matrix from the CSV file
    predictions = pd.read_csv(confusion_matrix_file, index_col=0)
    conf_matr = confusion_matrix(predictions["labels"], predictions["preds"])

    class_accuracy = conf_matr.diagonal()/conf_matr.sum(axis=1)
    class_accuracy_df = pd.DataFrame(class_accuracy)
    print(class_accuracy_df)

    # Create a heatmap using seaborn
    plt.figure(figsize=(20, 20))
    sns.heatmap(class_accuracy_df, annot=True, fmt=".2f", cmap="Blues")

    plt.title(model)

    # Save the plot if a save path is provided
    plt.savefig(os.path.join(results_base_dir, "{}_accuracy.png".format(model)))


if __name__ == "__main__":
    model = "vig_stem_VIG"
    results_base_file = os.path.join("/home/results/graph_image_understanding/resisc45/")
    conf_matrix_file = "predictions_test_100.csv"

    vig_accuracy = load_class_accuracy_per_model(os.path.join(results_base_file, model, conf_matrix_file))
    model2 = "bagnet_linear"
    conf_matrix_file2 = "predictions_test_100.csv"

    linearStemAccuracy = load_class_accuracy_per_model(os.path.join(results_base_file, model2, conf_matrix_file2))


    diff = vig_accuracy - linearStemAccuracy

    print(np.average(diff))

    most_influential_idx = np.argsort(diff)[::-1]

    print(np.sort(diff)[::-1])
    print(most_influential_idx)