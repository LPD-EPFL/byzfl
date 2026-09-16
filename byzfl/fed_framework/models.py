import torch
import torch.nn as nn
import torch.nn.functional as F
import snntorch as snn

from .surrogates import get_spike_grad

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


def _validate_time_steps(time_steps):
    if isinstance(time_steps, bool) or not isinstance(time_steps, int) or time_steps <= 0:
        raise ValueError("time_steps must be a positive integer.")
    return time_steps


def expand_temporal_dimension(model, x):
    """
    Repeat static inputs across time on their current device without copying.

    Static inputs have shape (batch, features) or (batch, channels, height,
    width). Temporal inputs have shape (batch, time, features) or (batch,
    time, channels, height, width) and retain their supplied sequence length.
    """
    if x.dim() == 4:
        time_steps = model.time_steps
        x = x.unsqueeze(1).expand(-1, time_steps, -1, -1, -1)
    elif x.dim() == 2:
        time_steps = model.time_steps
        x = x.unsqueeze(1).expand(-1, time_steps, -1)
    if x.dim() not in (3, 5):
        raise ValueError("SNN inputs must be batched static or temporal vectors or images.")
    if x.size(0) == 0 or x.size(1) == 0:
        raise ValueError("SNN inputs must contain at least one sample and one time step.")
    return x


class fc_snn(nn.Module):
    """
    Fully Connected Spiking Neural Network.

    Description:
    ------------
    A configurable fully connected spiking neural network designed for
    temporal classification tasks. This model accepts temporal
    inputs of shape ``(batch_size, time_steps, ...)``, where each time
    step is flattened and fed through the network sequentially.

    The model returns a tuple ``(spk_rec, mem_rec)`` containing the
    spike records and membrane potential records of the output layer,
    each of shape ``(time_steps, batch_size, output_dim)``.

    Static vectors or images are repeated across ``time_steps``. Encoded
    inputs retain their supplied time dimension.

    Parameters:
    -----------
    input_dim : int
        Dimensionality of the flattened input at each time step
        (default is 784 for 28x28 images).
    hidden_dim : int
        Number of neurons in the hidden layer (default is 100).
    output_dim : int
        Number of output classes (default is 10).
    beta : float
        Membrane potential decay rate for the leaky integrate-and-fire
        neurons (default is 0.95).
    surrogate_gradient : str
        Name of the surrogate gradient function from
        ``snntorch.surrogate`` (default is ``"atan"``).

    Examples:
    ---------
    >>> model = fc_snn(input_dim=784, hidden_dim=100, output_dim=10,
    ...               beta=0.95, surrogate_gradient="atan")
    >>> x = torch.randn(16, 25, 1, 28, 28)  # Batch of 16, 25 time steps
    >>> spk_rec, mem_rec = model(x)
    >>> print(spk_rec.shape)
    torch.Size([25, 16, 10])
    """

    is_snn = True

    def __init__(self, input_dim=784, hidden_dim=100, output_dim=10,
                 beta=0.95, surrogate_gradient="atan", time_steps=25, surrogate_params=None):
        super().__init__()
        self.time_steps = _validate_time_steps(time_steps)

        # Resolve the surrogate gradient before constructing the neuron layers.
        spike_grad = get_spike_grad(surrogate_gradient, surrogate_params)

        # Network layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)

    def forward(self, x):
        """
        Perform a forward pass through the spiking network.

        Parameters
        ----------
        x : torch.Tensor
            Temporal input of shape ``(batch_size, time_steps, ...)``.

        Returns
        -------
        tuple of torch.Tensor
            ``(spk_rec, mem_rec)`` where each has shape
            ``(time_steps, batch_size, output_dim)``.
        """
        x = expand_temporal_dimension(self, x)

        # Initialize membrane potentials
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        spk_rec = []
        mem_rec = []

        time_steps = x.size(1)
        for step in range(time_steps):
            # Extract time step and flatten spatial dimensions
            x_t = x[:, step].reshape(x.size(0), -1)

            cur1 = self.fc1(x_t)
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            spk_rec.append(spk2)
            mem_rec.append(mem2)

        return torch.stack(spk_rec), torch.stack(mem_rec)


class cnn_mnist_snn(nn.Module):
    """
    Spiking CNN matching the cnn_mnist architecture.

    Description:
    ------------
    A spiking convolutional neural network with 2 convolutional layers (20 and 50 filters)
    and 2 fully connected layers (500, num_classes) matching the cnn_mnist setup.
    LIF neurons are used for all spiking layers.

    The model returns a tuple ``(spk_rec, mem_rec)`` containing the
    spike records and membrane potential records of the output layer,
    each of shape ``(time_steps, batch_size, output_dim)``.
    """

    is_snn = True

    def __init__(self, in_channels=1, input_height=28, input_width=28, output_dim=10,
                 beta=0.95, surrogate_gradient="atan", threshold=1.0, learn_threshold=False,
                 time_steps=25, surrogate_params=None):
        super().__init__()
        self.time_steps = _validate_time_steps(time_steps)

        spike_grad = get_spike_grad(surrogate_gradient, surrogate_params)

        # Layer 1: Conv (in_channels -> 20 filters, 5x5 kernel) -> LIF -> MaxPool (2x2)
        self.conv1 = nn.Conv2d(in_channels, 20, kernel_size=5)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad, threshold=threshold, learn_threshold=learn_threshold)
        self.pool1 = nn.MaxPool2d(2)

        # Layer 2: Conv (20 -> 50 filters, 5x5 kernel) -> LIF -> MaxPool (2x2)
        self.conv2 = nn.Conv2d(20, 50, kernel_size=5)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad, threshold=threshold, learn_threshold=learn_threshold)
        self.pool2 = nn.MaxPool2d(2)

        # Calculate flattened feature count dynamically using dummy run
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, input_height, input_width)
            dummy_out = self.pool1(self.conv1(dummy))
            dummy_out = self.pool2(self.conv2(dummy_out))
            flat_features = dummy_out.numel()

        # Fully connected layers (500, output_dim classes)
        self.fc1 = nn.Linear(flat_features, 500)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad, threshold=threshold, learn_threshold=learn_threshold)

        self.fc2 = nn.Linear(500, output_dim)
        self.lif4 = snn.Leaky(beta=beta, spike_grad=spike_grad, threshold=threshold, learn_threshold=learn_threshold)

    def forward(self, x):
        x = expand_temporal_dimension(self, x)

        # Initialize membrane potentials
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem4 = self.lif4.init_leaky()

        spk_rec = []
        mem_rec = []

        time_steps = x.size(1)
        for step in range(time_steps):
            x_t = x[:, step] # Shape: (batch_size, channels, H, W)

            # Conv block 1
            cur1 = self.conv1(x_t)
            spk1, mem1 = self.lif1(cur1, mem1)
            spk1_pooled = self.pool1(spk1)

            # Conv block 2
            cur2 = self.conv2(spk1_pooled)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk2_pooled = self.pool2(spk2)

            # FC 1
            spk2_flat = spk2_pooled.reshape(spk2_pooled.size(0), -1)
            cur3 = self.fc1(spk2_flat)
            spk3, mem3 = self.lif3(cur3, mem3)

            # FC 2 (Output Layer)
            cur4 = self.fc2(spk3)
            spk4, mem4 = self.lif4(cur4, mem4)

            spk_rec.append(spk4)
            mem_rec.append(mem4)

        return torch.stack(spk_rec), torch.stack(mem_rec)


class cnn_cifar_snn(nn.Module):
    """
    Spiking CNN matching the cnn_cifar architecture.
    Uses LIF neurons for all spiking layers.
    """

    is_snn = True

    def __init__(self, in_channels=3, input_height=32, input_width=32, output_dim=10,
                 beta=0.95, surrogate_gradient="atan", threshold=1.0, learn_threshold=False,
                 time_steps=25, surrogate_params=None):
        super().__init__()
        self.time_steps = _validate_time_steps(time_steps)

        spike_grad = get_spike_grad(surrogate_gradient, surrogate_params)

        self.conv1 = nn.Conv2d(in_channels, 20, kernel_size=5, padding=2)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad, threshold=threshold, learn_threshold=learn_threshold)

        self.conv2 = nn.Conv2d(20, 100, kernel_size=5, padding=2)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad, threshold=threshold, learn_threshold=learn_threshold)

        self.conv3 = nn.Conv2d(100, 200, kernel_size=5, padding=2)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad, threshold=threshold, learn_threshold=learn_threshold)

        self.pool = nn.MaxPool2d(2, 2)

        # Calculate flattened feature count dynamically using dummy run
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, input_height, input_width)
            dummy_out = self.pool(self.conv1(dummy))
            dummy_out = self.pool(self.conv2(dummy_out))
            dummy_out = self.pool(self.conv3(dummy_out))
            flat_features = dummy_out.numel()

        self.fc1 = nn.Linear(flat_features, 512)
        self.lif4 = snn.Leaky(beta=beta, spike_grad=spike_grad, threshold=threshold, learn_threshold=learn_threshold)

        self.fc2 = nn.Linear(512, 256)
        self.lif5 = snn.Leaky(beta=beta, spike_grad=spike_grad, threshold=threshold, learn_threshold=learn_threshold)

        self.fc3 = nn.Linear(256, output_dim)
        self.lif6 = snn.Leaky(beta=beta, spike_grad=spike_grad, threshold=threshold, learn_threshold=learn_threshold)

    def forward(self, x):
        x = expand_temporal_dimension(self, x)

        # Initialize membrane potentials
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem4 = self.lif4.init_leaky()
        mem5 = self.lif5.init_leaky()
        mem6 = self.lif6.init_leaky()

        spk_rec = []
        mem_rec = []

        time_steps = x.size(1)
        for step in range(time_steps):
            x_t = x[:, step] # Shape: (batch_size, channels, H, W)

            # Conv block 1
            cur1 = self.conv1(x_t)
            spk1, mem1 = self.lif1(cur1, mem1)
            spk1_pooled = self.pool(spk1)

            # Conv block 2
            cur2 = self.conv2(spk1_pooled)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk2_pooled = self.pool(spk2)

            # Conv block 3
            cur3 = self.conv3(spk2_pooled)
            spk3, mem3 = self.lif3(cur3, mem3)
            spk3_pooled = self.pool(spk3)

            # FC 1
            spk3_flat = spk3_pooled.reshape(spk3_pooled.size(0), -1)
            cur4 = self.fc1(spk3_flat)
            spk4, mem4 = self.lif4(cur4, mem4)

            # FC 2
            cur5 = self.fc2(spk4)
            spk5, mem5 = self.lif5(cur5, mem5)

            # FC 3 (Output Layer)
            cur6 = self.fc3(spk5)
            spk6, mem6 = self.lif6(cur6, mem6)

            spk_rec.append(spk6)
            mem_rec.append(mem6)

        return torch.stack(spk_rec), torch.stack(mem_rec)
