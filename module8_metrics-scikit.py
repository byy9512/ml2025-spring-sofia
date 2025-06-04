import numpy as np
from sklearn.metrics import precision_score, recall_score

class BinaryClassificationMetrics:
    def __init__(self):
        self.data = np.empty((0, 2), dtype=np.int32)
    
    def load_data(self, point_list):
        self.data = point_list.astype(np.int32)
    
    def compute_metrics(self):
        # Set ground truth of the point
        true_labels = self.data[:, 0]

        #Set predicted class of the point
        predicted_labels = self.data[:, 1]

        # Calculate the precision and recall
        precision = precision_score(true_labels, predicted_labels, zero_division=0)
        recall = recall_score(true_labels, predicted_labels, zero_division=0)
        return precision, recall

def main():
    # Ask user to input N
    N = int(input("Please enter N (positive integer): "))
    
    # Read N points
    point_list = np.empty((N, 2), dtype=np.int32)
    for i in range(N):
        x = int(input(f"Enter x value (0 or 1) for point {i+1}: "))
        y = int(input(f"Enter y value (0 or 1) for point {i+1}: "))
        point_list[i, 0] = x
        point_list[i, 1] = y
    
    metrics = BinaryClassificationMetrics()

    # Load data into the NP array
    metrics.load_data(point_list)
    
    # Calculate the precision and recall and print the results
    precision, recall = metrics.compute_metrics()
    print(f"Precision: {precision:.2f}, Recall: {recall:.2f}")

if __name__ == "__main__":
    main()


