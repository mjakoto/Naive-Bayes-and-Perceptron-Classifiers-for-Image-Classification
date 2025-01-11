def image_to_features(image_lines):
    features = []
    for line in image_lines:
        for char in line:
            if char == ' ':
                features.append(0)
            else:
                features.append(1)
    return features