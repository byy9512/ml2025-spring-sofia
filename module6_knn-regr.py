import numpy as np

class KNNRegressor:
    def __init__(self):
        self.data = np.empty((0, 2), dtype=np.float64)
    
    def add_point(self, x, y):
        # Read the user's input data point of x and y
        new_point = np.array([[x, y]], dtype=np.float64)
        self.data = np.vstack((self.data, new_point))
    
    def knn_regression(self, X, k):
        # Check if k > N
        if k > self.data.shape[0]:
            raise ValueError("k must be ≤ N.")
        
        # Compute Manhattan distance
        distances = np.abs(self.data[:, 0] - X)
        sorted_indices = np.argsort(distances)

        # Locate k nearst neighbors
        k_nearest_indices = sorted_indices[:k]
        return np.mean(self.data[k_nearest_indices, 1])

def main():
    # Ask user to input N and k
    N = int(input("Please enter N (positive integer): "))
    k = int(input("Please enter k (positive integer): "))
    
    if k > N:
        print("Error: k cannot be greater than N.")
        return
    
    regressor = KNNRegressor()
    
    # Read N points
    for i in range(N):
        x = float(input(f"Please enter the x value for point {i+1}: "))
        y = float(input(f"Please enter the y value for point {i+1}: "))
        regressor.add_point(x, y)
    
    # Ask for test data and predict
    X = float(input("Please enter X: "))
    Y = regressor.knn_regression(X, k)
    print(f"The result (Y) of k-NN Regression is: {Y:.2f}")

if __name__ == "__main__":
    main()

