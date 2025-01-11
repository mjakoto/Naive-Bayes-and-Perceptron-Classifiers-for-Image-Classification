import numpy as np

class BernoulliNaiveBayes:
    def __init__(self, alpha=1.0):
        self.alpha = alpha  
        self.unique_classes = None
        self.prior_probabilities = {}
        self.conditional_probs = {} 

    def fit(self, features, labels):
        self.unique_classes = np.unique(labels)
        total_samples, total_features = features.shape
        
        for cls in self.unique_classes:
            class_subset = features[labels == cls]
            self.prior_probabilities[cls] = class_subset.shape[0] / total_samples
            feature_ones_count = np.sum(class_subset, axis=0)
            self.conditional_probs[cls] = (feature_ones_count + self.alpha) / (class_subset.shape[0] + 2 * self.alpha)

    def predict(self, features):
        predictions = []
        for instance in features:
            class_scores = []
            for cls in self.unique_classes:
                log_prior = np.log(self.prior_probabilities[cls])
                feature_prob = self.conditional_probs[cls]
                log_likelihood = np.sum(instance * np.log(feature_prob) + (1 - instance) * np.log(1 - feature_prob))
                posterior_log_prob = log_prior + log_likelihood
                class_scores.append((cls, posterior_log_prob))
            predicted_class = max(class_scores, key=lambda item: item[1])[0]
            predictions.append(predicted_class)
        return np.array(predictions)
