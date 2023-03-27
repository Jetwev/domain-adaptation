import matplotlib.pyplot as plt
import numpy as np


def get_image(data_loader, check):
    if check is True:
        train_features, train_labels = next(iter(data_loader))
        print(f"Feature batch shape: {train_features.size()}")
        print(f"Labels batch shape: {train_labels.size()}")
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = train_features[0]
        label = train_labels[0]
        img = std * img.permute(1, 2, 0).numpy() + mean
        print(f"Label: {label}")
        plt.imshow(img)
        plt.show()
    pass
