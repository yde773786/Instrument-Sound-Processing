"""
The aim of this script is to be able to predict the genre of the wav file
that is passed into it.

2 possible approaches to this problem are contrasted, as well as an ensemble of both models.

1. Use the spectogram of the wav file and use a CNN with 2DConv to classify the genre.
2. Use the raw wav file and use a CNN with 1DConv to classify the genre.
3. Ensemble model.
"""
import torch.nn as nn
from sklearn.model_selection import train_test_split
from PIL import Image
import torchvision.transforms as transforms

import os

# Directory of dataset used.
GTZAN_WAV = "../GTZAN/Data/genres_original/"
GTZAN_MEL = "../GTZAN/Data/images_original/"


class Classifier:
    def __init__(self):
        self.X_train, self.X_test, self.y_train, self.y_test = 0, 0, 0, 0

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
        pass

    def __init__(self):
        super().__init__()

    def test_train_splitter(self):
        X, Y = [], []

        # Go through all songs and tag X (tensor of image), Y as genre.
        for genre in os.listdir(GTZAN_MEL):
            for song in os.listdir(os.path.join(GTZAN_MEL, genre)):
                abs_path = os.path.join(GTZAN_MEL, genre, song)
                image = Image.open(abs_path)
                transform = transforms.Compose([transforms.PILToTensor()])
                # Convert PIL Image to tensor
                X.append(transform(image))
                Y.append(genre)

        # Obtain train/test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    def train_model(self):
        pass


if __name__ == '__main__':
    raw_approach_classifier = RawApproachClassifier()
    mel_spec_approach_classifier = MelSpecApproachClassifier()
    mel_spec_approach_classifier.test_train_splitter()
