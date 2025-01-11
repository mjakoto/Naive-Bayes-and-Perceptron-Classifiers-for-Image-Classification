import numpy as np

class Perceptron:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.lr = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.classes = None

    def fit(self, features, labels):
        self.classes = np.unique(labels)
        total_classes = len(self.classes)
        total_samples, total_features = features.shape
        
        self.weights = np.zeros((total_classes, total_features))
        self.bias = np.zeros(total_classes)

       
        modified_labels = np.zeros((total_samples, total_classes))
        for class_index, class_label in enumerate(self.classes):
            modified_labels[:, class_index] = np.where(labels == class_label, 1, -1)

       
        for class_idx in range(total_classes):
            current_weights = self.weights[class_idx]
            current_bias = self.bias[class_idx]
            for _ in range(self.n_iterations):
                for sample_idx, sample_features in enumerate(features):
                    activation = np.dot(sample_features, current_weights) + current_bias
                    predicted_label = 1 if activation >= 0 else -1
                    true_label = modified_labels[sample_idx, class_idx]
                    if predicted_label != true_label:
                        current_weights += self.lr * true_label * sample_features
                        current_bias += self.lr * true_label
            self.weights[class_idx] = current_weights
            self.bias[class_idx] = current_bias

    def predict(self, features):
        activations = np.dot(features, self.weights.T) + self.bias
        predicted_class_indices = np.argmax(activations, axis=1)
        return self.classes[predicted_class_indices]
