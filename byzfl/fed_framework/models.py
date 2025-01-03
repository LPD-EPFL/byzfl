import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Models Module
=============
This module contains a collection of models designed for various datasets such as MNIST and CIFAR. 
These models include fully connected networks, convolutional neural networks, logistic regression, 
and ResNet architectures.

Available Models
----------------

### MNIST Models

1. **`fc_mnist`**
   - **Type**: Fully connected neural network.
   - **Description**: A simple two-layer fully connected network designed for the MNIST dataset.
   - **Input Shape**: `28 x 28` (flattened into a vector).
   - **Output Classes**: 10.

2. **`cnn_mnist`**
   - **Type**: Convolutional neural network.
   - **Description**: A small convolutional neural network tailored for MNIST.
   - **Input Shape**: `1 x 28 x 28`.
   - **Output Classes**: 10.

3. **`logreg_mnist`**
   - **Type**: Logistic regression model.
   - **Description**: A simple logistic regression model for MNIST.
   - **Input Shape**: `28 x 28` (flattened into a vector).
   - **Output Classes**: 10.

### CIFAR Models

1. **`cnn_cifar_old`**
   - **Type**: Convolutional neural network.
   - **Description**: A small convolutional network for CIFAR datasets.
   - **Input Shape**: `3 x 32 x 32`.
   - **Output Classes**: 10.

2. **`cnn_cifar`**
   - **Type**: Convolutional neural network.
   - **Description**: An updated and efficient convolutional network for CIFAR datasets.
   - **Input Shape**: `3 x 32 x 32`.
   - **Output Classes**: 10.

3. **`cifar_Net`**
   - **Type**: Convolutional neural network.
   - **Description**: Another small convolutional network for CIFAR datasets.
   - **Input Shape**: `3 x 32 x 32`.
   - **Output Classes**: 10.

### ResNet Models

ResNet models are general-purpose convolutional neural networks capable of handling datasets like CIFAR-10 and CIFAR-100.

1. **`ResNet18`**
   - **Description**: ResNet with 18 layers.

2. **`ResNet34`**
   - **Description**: ResNet with 34 layers.

3. **`ResNet50`**
   - **Description**: ResNet with 50 layers.

4. **`ResNet101`**
   - **Description**: ResNet with 101 layers.

5. **`ResNet152`**
   - **Description**: ResNet with 152 layers.

Notes
-----
- All models are subclasses of `torch.nn.Module` and are compatible with PyTorch training pipelines.
- The ResNet implementations support custom class numbers via the `num_classes` parameter.
"""

class fc_mnist(nn.Module):
    """
    Fully Connected Network for MNIST.

    Description:
    ------------
    A simple fully connected neural network for the MNIST dataset, consisting of 
    two fully connected layers with ReLU activation and softmax output.

    Examples:
    ---------
    >>> model = fc_mnist()
    >>> x = torch.randn(16, 28*28)  # Batch of 16 MNIST images
    >>> output = model(x)
    >>> print(output.shape)
    torch.Size([16, 10])
    """
    def __init__(self):
        """Initialize the model parameters."""
        super().__init__()
        self._f1 = nn.Linear(28 * 28, 100)
        self._f2 = nn.Linear(100, 10)

    def forward(self, x):
        """Perform a forward pass through the model."""
        x = F.relu(self._f1(x.view(-1, 28 * 28)))
        x = F.log_softmax(self._f2(x), dim=1)
        return x


class cnn_mnist(nn.Module):
    """
    Convolutional Neural Network for MNIST.

    Description:
    ------------
    A simple convolutional neural network designed for the MNIST dataset. It 
    consists of two convolutional layers, ReLU activation, max pooling, and 
    fully connected layers.

    Examples:
    ---------
    >>> model = cnn_mnist()
    >>> x = torch.randn(16, 1, 28, 28)  # Batch of 16 grayscale MNIST images
    >>> output = model(x)
    >>> print(output.shape)
    torch.Size([16, 10])
    """
    def __init__(self):
        """Initialize the model parameters."""
        super().__init__()
        self._c1 = nn.Conv2d(1, 20, 5, 1)
        self._c2 = nn.Conv2d(20, 50, 5, 1)
        self._f1 = nn.Linear(800, 500)
        self._f2 = nn.Linear(500, 10)

    def forward(self, x):
        """Perform a forward pass through the model."""
        x = F.relu(self._c1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self._c2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self._f1(x.view(-1, 800)))
        x = F.log_softmax(self._f2(x), dim=1)
        return x


class logreg_mnist(nn.Module):
    """
    Logistic Regression Model for MNIST.

    Description:
    ------------
    A simple logistic regression model for the MNIST dataset. It consists of 
    a single linear layer.

    Examples:
    ---------
    >>> model = logreg_mnist()
    >>> x = torch.randn(16, 28*28)  # Batch of 16 MNIST images
    >>> output = model(x)
    >>> print(output.shape)
    torch.Size([16, 10])
    """
    def __init__(self):
        """Initialize the model parameters."""
        super().__init__()
        self._linear = nn.Linear(784, 10)

    def forward(self, x):
        """Perform a forward pass through the model."""
        return torch.sigmoid(self._linear(x.view(-1, 784)))

# ---------------------------------------------------------------------------- #

class cnn_cifar(nn.Module):
    """
    Convolutional Neural Network for CIFAR.

    Description:
    ------------
    A convolutional neural network designed for the CIFAR-10 and CIFAR-100 
    datasets. It consists of three convolutional layers, max pooling, and 
    fully connected layers.

    Examples:
    ---------
    >>> model = cnn_cifar()
    >>> x = torch.randn(16, 3, 32, 32)  # Batch of 16 CIFAR images
    >>> output = model(x)
    >>> print(output.shape)
    torch.Size([16, 10])
    """
    def __init__(self):
        """Initialize the model parameters."""
        super().__init__()
        self.conv1 = nn.Conv2d(3, 20, 5, padding=2)
        self.conv2 = nn.Conv2d(self.conv1.out_channels, 100, 5, padding=2)
        self.conv3 = nn.Conv2d(self.conv2.out_channels, 200, 5, padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(self.conv3.out_channels * 4 * 4, 512)
        self.fc2 = nn.Linear(self.fc1.out_features, 256)
        self.fc3 = nn.Linear(self.fc2.out_features, 10)

    def forward(self, x):
        """Perform a forward pass through the model."""
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        return F.relu(out)


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, out_feature=False):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.avg_pool2d(x, 4)
        feature = x.view(x.size(0), -1)
        out = self.linear(feature)
        return (out, feature) if out_feature else out


class ResNet18(nn.Module):
    """
    Description:
    ------------
    ResNet18 is a convolutional neural network architecture with 18 layers. 
    It is designed for image classification tasks and includes skip 
    connections for efficient gradient flow.

    Parameters:
    ------------
    num_classes : int
        The number of output classes for classification (default is 10).

    Examples:
    ---------
    >>> model = ResNet18(num_classes=10)
    >>> x = torch.randn(16, 3, 32, 32)  # Batch of 16 CIFAR images
    >>> output = model(x)
    >>> print(output.shape)
    torch.Size([16, 10])
    """
    def __init__(self, num_classes=10):
        super().__init__()
        self.model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes)

    def forward(self, x):
        """Perform a forward pass through the model."""
        return self.model(x)


class ResNet34(nn.Module):
    """
    Description:
    ------------
    ResNet34 is a deep convolutional neural network with 34 layers, designed for image classification tasks.
    It uses residual connections to improve gradient flow and enable training of very deep networks.

    Parameters:
    ------------
    num_classes : int
        The number of output classes for classification (default is 10).

    Examples:
    ---------
    >>> model = ResNet34(num_classes=100)
    >>> x = torch.randn(8, 3, 32, 32)  # Batch of 8 images with CIFAR-like dimensions
    >>> output = model(x)
    >>> print(output.shape)
    torch.Size([8, 100])
    """
    def __init__(self, num_classes=10):
        super().__init__()
        self.model = ResNet(BasicBlock, [3, 4, 6, 3], num_classes)

    def forward(self, x):
        """Perform a forward pass through the model."""
        return self.model(x)


class ResNet50(nn.Module):
    """
    Description:
    ------------
    ResNet50 is a deeper ResNet variant with 50 layers. It employs the Bottleneck block to reduce
    computational complexity while maintaining accuracy, making it suitable for larger-scale datasets
    and more complex tasks.

    Parameters:
    ------------
    num_classes : int
        The number of output classes for classification (default is 10).

    Examples:
    ---------
    >>> model = ResNet50(num_classes=1000)
    >>> x = torch.randn(16, 3, 224, 224)  # Batch of 16 images with ImageNet-like dimensions
    >>> output = model(x)
    >>> print(output.shape)
    torch.Size([16, 1000])
    """
    def __init__(self, num_classes=10):
        super().__init__()
        self.model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes)

    def forward(self, x):
        """Perform a forward pass through the model."""
        return self.model(x)


class ResNet101(nn.Module):
    """
    Description:
    ------------
    ResNet101 is a deeper ResNet variant with 101 layers, designed for highly complex tasks.
    It leverages Bottleneck blocks to maintain performance while keeping computational costs manageable.

    Parameters:
    ------------
    num_classes : int
        The number of output classes for classification (default is 10).

    Examples:
    ---------
    >>> model = ResNet101(num_classes=100)
    >>> x = torch.randn(4, 3, 64, 64)  # Batch of 4 images
    >>> output = model(x)
    >>> print(output.shape)
    torch.Size([4, 100])
    """
    def __init__(self, num_classes=10):
        super().__init__()
        self.model = ResNet(Bottleneck, [3, 4, 23, 3], num_classes)

    def forward(self, x):
        """Perform a forward pass through the model."""
        return self.model(x)


class ResNet152(nn.Module):
    """
    Description:
    ------------
    ResNet152 is the deepest ResNet variant among the standard configurations. With 152 layers,
    it is highly effective for complex tasks, including image classification, segmentation, and detection.
    The model achieves a balance between depth and computational feasibility using Bottleneck blocks.

    Parameters:
    ------------
    num_classes : int
        The number of output classes for classification (default is 10).

    Examples:
    ---------
    >>> model = ResNet152(num_classes=10)
    >>> x = torch.randn(2, 3, 128, 128)  # Batch of 2 high-resolution images
    >>> output = model(x)
    >>> print(output.shape)
    torch.Size([2, 10])
    """
    def __init__(self, num_classes=10):
        super().__init__()
        self.model = ResNet(Bottleneck, [3, 8, 36, 3], num_classes)

    def forward(self, x):
        """Perform a forward pass through the model."""
        return self.model(x)