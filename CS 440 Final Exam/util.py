import numpy as np

def load_data(image_file, label_file, image_height, image_width):
    """
    Load data from given ASCII image file and label file.
    Each image is image_height lines of text, each line is image_width chars.
    Labels file has one label per line.
    """
    with open(image_file, 'r') as f_img:
        image_lines = f_img.read().splitlines()
    with open(label_file, 'r') as f_lbl:
        labels = f_lbl.read().splitlines()
    n_images = len(labels)
    assert len(image_lines) == n_images * image_height, "Image file does not match expected dimensions"

    X = []
    y = []

    for i in range(n_images):
        start = i * image_height
        end = start + image_height
        img_block = image_lines[start:end]
        
        features = []
        for line in img_block:
            for char in line:
                features.append(0 if char == ' ' else 1)
        
        X.append(features)
        y.append(labels[i].strip())

    return np.array(X), np.array(y)

def sample_training_data(X, y, fraction=0.1):
    n_samples = int(len(X) * fraction)
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    selected = indices[:n_samples]
    return X[selected], y[selected]

def compute_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)
