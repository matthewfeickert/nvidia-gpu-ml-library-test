# c.f. https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-model-for-inference

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import datasets, transforms


# Use the CNN architecture defined in torch_MNIST.py
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def preprocess(image):
    image = image.resize((28, 28))
    image_array = np.array(image).astype(np.float32)
    return image_array.reshape(1, 28, 28)


def predict_image(image, model):
    x = image.unsqueeze(0)
    y = model(x)
    _, prediction = torch.max(y, dim=1)
    return prediction[0].item()


def main():
    parser = argparse.ArgumentParser(description="PyTorch MNIST inference example")
    parser.add_argument(
        "--model-path",
        default=Path.cwd() / "mnist_cnn.pt",
        help="Path to the trained model archive (default: ./mnist_cnn.pt)",
    )
    parser.add_argument(
        "--image-path",
        default=Path.cwd() / "test_image.png",
        help="Path to the image to perform inference on (default: ./test_image.png)",
    )
    parser.add_argument(
        "--data-dir",
        default=Path.cwd() / "data",
        help="Path to the directory with the MNIST training data (default: ./data)",
    )
    parser.add_argument(
        "--viz",
        default=False,
        help="Show the image inference is being run on (default: False)",
    )
    args = parser.parse_args()
    image_path = Path(args.image_path).resolve()

    if image_path.exists():
        # "L" is grayscale mode
        # c.f. https://pillow.readthedocs.io/en/latest/reference/Image.html
        with Image.open(image_path) as read_image:
            read_image.load()
            grayscale_image = read_image.convert("L")
            test_image = torch.as_tensor(preprocess(grayscale_image))
            label = None
    else:
        # Fallback to using images from the MNIST dataset
        data_dir = Path(args.data_dir).resolve()

        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        test_dataset = datasets.MNIST(data_dir, train=False, transform=transform)

        # Hardcoding an index as an example
        test_image, label = test_dataset[250]
        # save the image for reference for future use
        plt.imsave(image_path, test_image[0], cmap="gray")

    # Model class must be defined somewhere
    model_path = Path(args.model_path).resolve()
    state_dict = torch.load(model_path, weights_only=True)

    # Remember that you must call model.eval() to set dropout and batch normalization
    # layers to evaluation mode before running inference.
    # Failing to do this will yield inconsistent inference results.
    model = Net()
    model.load_state_dict(state_dict)
    model.eval()

    if label:
        print(f"Label: {label}, Prediction: {predict_image(test_image, model)}")
    else:
        print(f"Prediction: {predict_image(test_image, model)}")

    if args.viz:
        plt.imshow(test_image[0], cmap="gray")
        plt.show()


if __name__ == "__main__":
    main()
