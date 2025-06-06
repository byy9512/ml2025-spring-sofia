import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, LeaveOneOut

class KNNTuner:
    def __init__(self):
        self.train_X = None
        self.train_Y = None
        self.test_X = None
        self.test_Y = None
    
    def load_train_data(self, train_X, train_Y):
        self.train_X = train_X.astype(np.float64)
        self.train_Y = train_Y.astype(np.int64)
    
    def load_test_data(self, test_X, test_Y):
        self.test_X = test_X.astype(np.float64)
        self.test_Y = test_Y.astype(np.int64)
    
    def find_best_k(self):
        # Set the range for k in exhaustive search
        # param_grid = {'n_neighbors': list(range(1, 11))}

        # Set the range for k in exhaustive search, if k>len(samples), it will return warnings
        max_k = min(10, len(self.train_X) - 1)
        param_grid = {'n_neighbors': list(range(1, max_k + 1))}

        # Adjust cross-validation based on training size
        n_samples = self.train_X.shape[0]
        cv_selection = 5 if n_samples >= 5 else LeaveOneOut()
        
        # GridSearch initialization with five fold cross validation
        knn = KNeighborsClassifier()
        grid_search = GridSearchCV(
            estimator=knn,
            param_grid=param_grid,
            scoring='accuracy',
            cv=cv_selection 
        )
        
        # GridSearch on training data
        grid_search.fit(self.train_X, self.train_Y)
        
        # Get the best k
        best_k = grid_search.best_params_['n_neighbors']
        best_model = grid_search.best_estimator_
        
        # Get the accuracy on test data
        y_pred = best_model.predict(self.test_X)
        best_accuracy = accuracy_score(self.test_Y, y_pred)
        
        return best_k, best_accuracy

def main():
    # Ask user to input N
    N = int(input("Please enter N (positive integer): "))

    # Read N points of training data
    train_X = np.empty((N, 1), dtype=np.float64)
    train_Y = np.empty(N, dtype=np.int64)
    for i in range(N):
        x = float(input(f"Enter x value (real number) for training point {i+1}: "))
        y = int(input(f"Enter y value (non-negative integer) for training point {i+1}: "))
        train_X[i, 0] = x
        train_Y[i] = y
    
    # Ask user to input M
    M = int(input("Please enter M (positive integer): "))

    # Read M points of test data
    test_X = np.empty((M, 1), dtype=np.float64)
    test_Y = np.empty(M, dtype=np.int64)
    for i in range(M):
        x = float(input(f"Enter x value (real number) for test point {i+1}: "))
        y = int(input(f"Enter y value (non-negative integer) for test point {i+1}: "))
        test_X[i, 0] = x
        test_Y[i] = y
    
    # Load data
    tuner = KNNTuner()
    tuner.load_train_data(train_X, train_Y)
    tuner.load_test_data(test_X, test_Y)
    
    # GridSearch tunning for the best k and corresponding accuracy
    best_k, best_accuracy = tuner.find_best_k()
    
    # Return the best k and accuracy on test data
    print(f"Best k: {best_k}, Corresponding test accuracy: {best_accuracy:.2f}")

if __name__ == "__main__":
    main()


