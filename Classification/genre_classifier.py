"""
The aim of this script is to be able to predict the genre of the wav file
that is passed into it.

2 possible approaches to this problem are contrasted, as well as an ensemble of both models.

1. Use the spectogram of the wav file and use a CNN with 2DConv to classify the genre.
2. Use the raw wav file and use a CNN with 1DConv to classify the genre.
3. Ensemble model.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import random
from PIL import Image
import torchvision.transforms as transforms
import torch.utils.data as Data
import os
from PIL import ImageOps

from torch.utils.data import SubsetRandomSampler

# Directory of dataset used.
GTZAN_WAV = "../GTZAN/Data/genres_original/"
GTZAN_MEL = "../GTZAN/Data/images_original/"

PREPROCESS_CROP = (54, 35, 42, 35)

IMAGE_INPUT_DIMENSIONS = [432, 288]
GENRES = {'blues': 0, 'classical': 1, 'country': 2, 'disco': 3,
          'hiphop': 4, 'jazz': 5, 'metal': 6, 'pop': 7, 'reggae': 8,
          'rock': 9}


class Classifier:
    def __init__(self):
        self.X_train, self.X_test, self.y_train, self.y_test = 0, 0, 0, 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print("Using device", self.device)

    def test_train_split(self):
        raise NotImplementedError

    def train_model(self):
        raise NotImplementedError


# Use the raw wav file to train a model for classification of genre of the wav file
class RawApproachClassifier(Classifier):
    def __init__(self):
        super().__init__()

    def test_train_splitter(self):
        ...


# Use the mel spectrogram to train a model for classification of genre of the wav file
class MelSpecApproachClassifier(Classifier):
    class MelSpecTrainer(nn.Module):
        def __init__(self):
            super().__init__()

            self.current_dimensions = IMAGE_INPUT_DIMENSIONS

            self.conv_layer_1 = nn.Sequential(nn.Conv2d(4, 32, 3),
                                              nn.ReLU(),
                                              nn.MaxPool2d(kernel_size=2)
                                              )

            self.conv_layer_2 = nn.Sequential(nn.Conv2d(32, 16, 3),
                                              nn.ReLU(),
                                              nn.MaxPool2d(kernel_size=2)
                                              )

            self.flatten_layer = nn.Flatten()

            self.linear_layer = nn.Sequential(nn.Linear(69536, 512),
                                              nn.ReLU())

            self.classifier = nn.Linear(512, 10)

        def forward(self, x):
            # First 2D convolution layer
            x = self.conv_layer_1(x)
            # Second 2D convolution layer
            x = self.conv_layer_2(x)

            # Linear layer and classifier
            x = self.flatten_layer(x)
            x = self.linear_layer(x)
            x = self.classifier(x)

            return x

    class ImageDataset(Data.Dataset):
        def __init__(self):
            self.images = []
            self.labels = []

            # Go through all songs and tag X (tensor of image), Y as genre.
            for genre in os.listdir(GTZAN_MEL):
                for song in os.listdir(os.path.join(GTZAN_MEL, genre)):
                    abs_path = os.path.join(GTZAN_MEL, genre, song)
                    image = Image.open(abs_path)

                    # The images have been obtained in the dataset by using the mel spectogram (librosa)
                    # Cropping the image to only contain the spectogram to pass into CNN
                    image_cropped = ImageOps.crop(image, PREPROCESS_CROP)

                    transform = transforms.Compose([transforms.ToTensor()])
                    # Convert PIL Image to tensor
                    self.images.append(transform(image_cropped))
                    # Convert genre tag to associated digit
                    self.labels.append(GENRES[genre])

        def __len__(self):
            return len(self.images)

        def __getitem__(self, idx):
            return self.images[idx], self.labels[idx]

    def __init__(self):
        super().__init__()
        image_dataset = self.ImageDataset()
        self.model = self.MelSpecTrainer()
        self.model.to(self.device)
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        # shuffle dataset before splitting into test/train datasets
        indices = list(range(len(image_dataset)))
        random.seed(42)
        random.shuffle(indices)

        # 80% dataset for training, 10% for validation, and 10% for testing.
        num_train = int(len(image_dataset) * 0.8)
        num_validation = int(len(image_dataset) * 0.1)

        train_indices = indices[:num_train]
        test_and_validation = indices[num_train:]
        validation_indices = test_and_validation[:num_validation]
        test_indices = test_and_validation[num_validation:]

        # Create test and train datasets
        train_sampler = SubsetRandomSampler(train_indices)
        validation_sampler = SubsetRandomSampler(validation_indices)
        test_sampler = SubsetRandomSampler(test_indices)
        self.train_dataset = Data.DataLoader(image_dataset, batch_size=5, sampler=train_sampler)
        self.validation_dataset = Data.DataLoader(image_dataset, sampler=validation_sampler)
        self.test_dataset = Data.DataLoader(image_dataset, sampler=test_sampler)

    def train_model(self):
        for epoch in range(50):
            for batch_id, curr_batch in enumerate(self.train_dataset):
                # Predict and get loss
                images, labels = curr_batch[0].to(self.device), curr_batch[1].to(self.device)
                pred = self.model(images)
                loss = self.loss_fn(pred, labels)

                # backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                print(f"epoch: {epoch}, batch_id: {batch_id}, loss: {loss}")

    def test_model(self, type_dataset):
        # evaluation mode
        self.model.eval()
        correct_cnt = 0

        if type_dataset.lower() == "test":
            dataset = self.test_dataset
        elif type_dataset.lower() == "validation":
            dataset = self.validation_dataset
        else:
            dataset = self.train_dataset

        with torch.no_grad():
            for images, labels in dataset:

                images, labels = images.to(self.device), labels.to(self.device)
                pred = self.model(images)

                # Correctly classified genre of song snippet
                _, predicted = torch.max(pred, 1)
                correct_cnt += (predicted == labels).sum().item()

        print(f"{type_dataset} Accuracy: {correct_cnt / len(dataset)}")


if __name__ == '__main__':
    raw_approach_classifier = RawApproachClassifier()
    mel_spec_approach_classifier = MelSpecApproachClassifier()
    mel_spec_approach_classifier.train_model()
    mel_spec_approach_classifier.test_model(type_dataset="Train")
    mel_spec_approach_classifier.test_model(type_dataset="Validation")
    mel_spec_approach_classifier.test_model(type_dataset="Test")
