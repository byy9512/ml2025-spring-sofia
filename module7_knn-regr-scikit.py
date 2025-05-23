import numpy as np
from sklearn.neighbors import KNeighborsRegressor

class KNNRegressor:
    def __init__(self):
        self.data = np.empty((0, 2), dtype=np.float64)
    
    def load_data(self, point_list):
        self.data = np.array(point_list, dtype=np.float64)
    
    def knn_regression(self, X, k):
        # Check if k > N
        if k > self.data.shape[0]:
            raise ValueError("k must be ≤ N.")
        
        # KNN regression with Scikit-learn package
        X_train = self.data[:, 0].reshape(-1, 1)
        y_train = self.data[:, 1]
        model = KNeighborsRegressor(n_neighbors=k, metric='manhattan')
        model.fit(X_train, y_train)
        return model.predict([[X]])[0]

def main():
    # Ask user to input N and k
    N = int(input("Please enter N (positive integer): "))
    k = int(input("Please enter k (positive integer): "))

    if k > N:
        print("Error: k cannot be greater than N.")
        return
    
    # Read N points
    point_list = []
    for i in range(N):
        x = float(input(f"Please enter the x value for point {i+1}: "))
        y = float(input(f"Please enter the y value for point {i+1}: "))
        point_list.append((x, y))
    
    regressor = KNNRegressor()

    # Load data into the NP array
    regressor.load_data(point_list)
    
    # Calculate variance
    variance = np.var(regressor.data[:, 1])
    
    # Ask for test data and predict
    X = float(input("Please enter X: "))
    Y = regressor.knn_regression(X, k)
    print(f"The result (Y) of k-NN Regression is: {Y:.2f}, Variance of training data is: {variance:.2f}")

if __name__ == "__main__":
    main()


