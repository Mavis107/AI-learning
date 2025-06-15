"""SegmentationNN"""
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


class ConvLayer(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.activation = nn.ReLU()
        self.norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x


class SegmentationNN(nn.Module):

    def __init__(self, num_classes=23, hp=None):
        super().__init__()
        self.hp = hp
        # Load pre-trained MobileNetV2 model
        mobilenet = models.mobilenet_v2(pretrained=True)
        # Extract features from MobileNetV2
        self.features = mobilenet.features
        # Replace the last classifier layer with a new segmentation head
        self.segmentation_head = nn.Sequential(
            nn.Conv2d(self.hp["n_hidden"], self.hp["n_filters"], kernel_size=self.hp["kernel_size"], padding=self.hp["padding"]),
            nn.BatchNorm2d(self.hp["n_filters"]),
            nn.ReLU(),
            nn.Conv2d(self.hp["n_filters"], num_classes, kernel_size=self.hp["kernel_size_2"]) # Final output layer for segmentation
        )



    def forward(self, x):
        """
        Forward pass of the convolutional neural network. Should not be called
        manually but by calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """
        # Extract features using MobileNetV2 backbone
        x = self.features(x)
        # Apply segmentation head
        x = self.segmentation_head(x)
        # Resize to match target size
        x = F.interpolate(x, size=(240, 240), mode='bilinear', align_corners=True)

        return x

    @property
    def is_cuda(self):
        """
        Check if model parameters are allocated on the GPU.
        """
        return next(self.parameters()).is_cuda

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)

        
class DummySegmentationModel(nn.Module):

    def __init__(self, target_image):
        super().__init__()
        def _to_one_hot(y, num_classes):
            scatter_dim = len(y.size())
            y_tensor = y.view(*y.size(), -1)
            zeros = torch.zeros(*y.size(), num_classes, dtype=y.dtype)

            return zeros.scatter(scatter_dim, y_tensor, 1)

        target_image[target_image == -1] = 1

        self.prediction = _to_one_hot(target_image, 23).permute(2, 0, 1).unsqueeze(0)

    def forward(self, x):
        return self.prediction.float()

if __name__ == "__main__":
    from torchinfo import summary
    summary(SegmentationNN(), (1, 3, 240, 240), device="cpu")