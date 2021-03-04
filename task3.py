import pathlib
import matplotlib.pyplot as plt
import utils
import torch
from torch import nn
from dataloaders import load_cifar10
from trainer import Trainer, compute_loss_and_accuracy


class Model1(nn.Module):

    def __init__(self,
                 image_channels,
                 num_classes):
        """
            Is called when model is initialized.
            Args:
                image_channels. Number of color channels in image (3)
                num_classes: Number of classes we want to predict (10)
        """
        super().__init__()

        self.num_classes = num_classes

        # Define the convolutional layers
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=image_channels, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # the feature_extractor outputs [num_classes, 128, 4, 4]

        # Define the fully-connected layers
        self.classifier = nn.Sequential(
            nn.Linear(2048, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        """
        Performs a forward pass through the model
        Args:
            x: Input image, shape: [batch_size, 3, 32, 32]
        """
        batch_size = x.shape[0]
        expected_shape = (batch_size, self.num_classes)

        out = self.feature_extractor(x)
        out = torch.flatten(out, 1)
        out = self.classifier(out)

        assert out.shape == (batch_size, self.num_classes),\
            f"Expected output of forward pass to be: {expected_shape}, but got: {out.shape}"
        return out

class Model2(nn.Module):

    def __init__(self,
                 image_channels,
                 num_classes,
                 kernel_size,
                 padding,
                 n_filters):
        """
            Is called when model is initialized.
            Args:
                image_channels. Number of color channels in image (3)
                num_classes: Number of classes we want to predict (10)
        """
        super().__init__()

        self.num_classes = num_classes
        
        # new parameters
        self.kernel_size = kernel_size
        self.padding = padding
        self.n_filters = n_filters

        # Define the convolutional layers
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=image_channels, out_channels=self.n_filters[0], kernel_size=self.kernel_size, stride=1, padding=self.padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=self.n_filters[0], out_channels=self.n_filters[1], kernel_size=self.kernel_size, stride=1, padding=self.padding),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(in_channels=self.n_filters[1], out_channels=self.n_filters[2], kernel_size=self.kernel_size, stride=1, padding=self.padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=self.n_filters[2], out_channels=self.n_filters[3], kernel_size=self.kernel_size, stride=1, padding=self.padding),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # the feature_extractor outputs [num_classes, n_filters[-1], 4, 4]
        self.num_output_features = self.n_filters[3]*8*8

        # Define the fully-connected layers
        self.classifier = nn.Sequential(
            nn.Linear(self.num_output_features, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        """
        Performs a forward pass through the model
        Args:
            x: Input image, shape: [batch_size, 3, 32, 32]
        """
        batch_size = x.shape[0]
        expected_shape = (batch_size, self.num_classes)

        out = self.feature_extractor(x)
        out = torch.flatten(out, 1)
        out = self.classifier(out)

        assert out.shape == (batch_size, self.num_classes),\
            f"Expected output of forward pass to be: {expected_shape}, but got: {out.shape}"
        return out




def create_plots(trainer: Trainer, name: str):
    plot_path = pathlib.Path("plots")
    plot_path.mkdir(exist_ok=True)
    # Save plots and show them
    plt.figure(figsize=(20, 8))
    plt.subplot(1, 2, 1)
    plt.title("Cross Entropy Loss")
    utils.plot_loss(trainer.train_history["loss"], label="Training loss", npoints_to_average=10)
    utils.plot_loss(trainer.validation_history["loss"], label="Validation loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.title("Accuracy")
    utils.plot_loss(trainer.validation_history["accuracy"], label="Validation Accuracy")
    plt.legend()
    plt.savefig(plot_path.joinpath(f"{name}_plot.png"))
    plt.show()


if __name__ == "__main__":
    # Set the random generator seed (parameters, shuffling etc).
    # You can try to change this and check if you still get the same result! 
    utils.set_seed(0)
    epochs = 10
    batch_size = 64
    learning_rate = 5e-2
    early_stop_count = 4
    dataloaders = load_cifar10(batch_size)
    
    #parameters to experiment with
    F = 5
    p = 2
    n_filters = [64, 128, 256, 512]
    
    # training model2
    model2 = Model2(image_channels=3, 
                    num_classes=10, 
                    kernel_size=F, 
                    padding=p,
                    n_filters=n_filters)
    trainer = Trainer(
        batch_size,
        learning_rate,
        early_stop_count,
        epochs,
        model2,
        dataloaders
    )
    trainer.train()
    create_plots(trainer, f"task3_model1_F{F}_p{p}_n{n_filters[0]}-{n_filters[1]}-{n_filters[2]}-{n_filters[3]}_new_structure")
    #comment
