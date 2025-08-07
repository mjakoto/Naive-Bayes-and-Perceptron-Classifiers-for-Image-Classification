import numpy as np
import time
from classifiers.perceptron import Perceptron
from classifiers.naive_bayes import BernoulliNaiveBayes
from util import load_data, sample_training_data, compute_accuracy

if __name__ == "__main__":
    digit_image_height = 28
    digit_image_width = 28
    face_image_height = 70
    face_image_width = 60

    face_training_images = "data/facedata/facedatatrain"
    face_training_labels = "data/facedata/facedatatrainlabels"
    face_test_images = "data/facedata/facedatatest"
    face_test_labels = "data/facedata/facedatatestlabels"

    digit_training_images = "data/digitdata/trainingimages"
    digit_training_labels = "data/digitdata/traininglabels"
    digit_test_images = "data/digitdata/testimages"
    digit_test_labels = "data/digitdata/testlabels"


    X_train_digits, y_train_digits = load_data(digit_training_images, digit_training_labels, digit_image_height, digit_image_width)
    X_test_digits, y_test_digits = load_data(digit_test_images, digit_test_labels, digit_image_height, digit_image_width)

    X_train_faces, y_train_faces = load_data(face_training_images, face_training_labels, face_image_height, face_image_width)
    X_test_faces, y_test_faces = load_data(face_test_images, face_test_labels, face_image_height, face_image_width)

    training_fractions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    n_repeats = 5  

    def run_experiment(X_train, y_train, X_test, y_test, clf_class, fractions, repeats):
        acc_means = []
        acc_stds = []
        time_means = []
        time_stds = []
        for frac in fractions:
            accs = []
            times = []
            for _ in range(repeats):
                X_sub, y_sub = sample_training_data(X_train, y_train, fraction=frac)
                clf = clf_class()
                start = time.time()
                clf.fit(X_sub, y_sub)
                end = time.time()
                y_pred = clf.predict(X_test)
                accs.append(compute_accuracy(y_pred, y_test))
                times.append(end - start)
            acc_means.append(np.mean(accs))
            acc_stds.append(np.std(accs))
            time_means.append(np.mean(times))
            time_stds.append(np.std(times))
        return acc_means, acc_stds, time_means, time_stds

    digit_perceptron_acc_means, digit_perceptron_acc_stds, digit_perceptron_time_means, digit_perceptron_time_stds = \
        run_experiment(X_train_digits, y_train_digits, X_test_digits, y_test_digits, 
                       lambda: Perceptron(learning_rate=0.1, n_iterations=10), 
                       training_fractions, n_repeats)

    digit_nb_acc_means, digit_nb_acc_stds, digit_nb_time_means, digit_nb_time_stds = \
        run_experiment(X_train_digits, y_train_digits, X_test_digits, y_test_digits, 
                       BernoulliNaiveBayes, 
                       training_fractions, n_repeats)

    face_perceptron_acc_means, face_perceptron_acc_stds, face_perceptron_time_means, face_perceptron_time_stds = \
        run_experiment(X_train_faces, y_train_faces, X_test_faces, y_test_faces, 
                       lambda: Perceptron(learning_rate=0.1, n_iterations=10), 
                       training_fractions, n_repeats)

    face_nb_acc_means, face_nb_acc_stds, face_nb_time_means, face_nb_time_stds = \
        run_experiment(X_train_faces, y_train_faces, X_test_faces, y_test_faces, 
                       BernoulliNaiveBayes, 
                       training_fractions, n_repeats)

    print("Digits - Perceptron")
    for f, am, asd, tm, tsd in zip(training_fractions, digit_perceptron_acc_means, digit_perceptron_acc_stds, digit_perceptron_time_means, digit_perceptron_time_stds):
        print(f"Train fraction: {f*100}%, Acc: {am:.3f} ± {asd:.3f}, Time: {tm:.3f} ± {tsd:.3f}")

    print("Digits - Naive Bayes")
    for f, am, asd, tm, tsd in zip(training_fractions, digit_nb_acc_means, digit_nb_acc_stds, digit_nb_time_means, digit_nb_time_stds):
        print(f"Train fraction: {f*100}%, Acc: {am:.3f} ± {asd:.3f}, Time: {tm:.3f} ± {tsd:.3f}")

    print("Faces - Perceptron")
    for f, am, asd, tm, tsd in zip(training_fractions, face_perceptron_acc_means, face_perceptron_acc_stds, face_perceptron_time_means, face_perceptron_time_stds):
        print(f"Train fraction: {f*100}%, Acc: {am:.3f} ± {asd:.3f}, Time: {tm:.3f} ± {tsd:.3f}")

    print("Faces - Naive Bayes")
    for f, am, asd, tm, tsd in zip(training_fractions, face_nb_acc_means, face_nb_acc_stds, face_nb_time_means, face_nb_time_stds):
        print(f"Train fraction: {f*100}%, Acc: {am:.3f} ± {asd:.3f}, Time: {tm:.3f} ± {tsd:.3f}")
