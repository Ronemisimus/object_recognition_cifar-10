import torch
import torch.nn as nn


class ConvClassifier(nn.Module):
    """
    A convolutional classifier model based on PyTorch nn.Modules.

    The architecture is:
    [(Conv -> ReLU)*P -> MaxPool]*(N/P) -> (Linear -> ReLU)*M -> Linear
    """
    def __init__(self, in_size, out_classes, filters, pool_every, hidden_dims):
        """
        :param in_size: Size of input images, e.g. (C,H,W).
        :param out_classes: Number of classes to output in the final layer.
        :param filters: A list of length N containing the number of
            filters in each conv layer.
        :param pool_every: P, the number of conv layers before each max-pool.
        :param hidden_dims: List of of length M containing hidden dimensions of
            each Linear layer (not including the output layer).
        """
        super().__init__()
        self.in_size = in_size
        self.out_classes = out_classes
        self.filters = filters
        self.pool_every = pool_every
        self.hidden_dims = hidden_dims

        self.feature_extractor = self._make_feature_extractor()
        self.classifier = self._make_classifier()
        
    def _make_feature_extractor(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        # TODO: Create the feature extractor part of the model:
        # [(Conv -> ReLU)*P -> MaxPool]*(N/P)
        # Use only dimension-preserving 3x3 convolutions (you will need to add padding). Apply 2x2 Max
        # Pooling to reduce dimensions.
        # If P>N you should implement:
        # (Conv -> ReLU)*N
        # Hint: use loop for len(self.filters) and append the layers you need to the list named 'layers'.
        # Use :
        # if <layer index>%self.pool_every==0:
        #     ...
        # in order to append maxpooling layer in the right places.
        # ====== YOUR CODE: ======
        filters = [in_channels] + self.filters 

        for layer_idx in range(len(filters)-1):
          in_f = filters[layer_idx]
          out_f = filters[layer_idx+1]
          layer = torch.nn.Conv2d(in_f, out_f, (3,3), stride=1, padding="same")
          layers.append(layer)
          activation = torch.nn.ReLU()
          layers.append(activation)
          if layer_idx % self.pool_every == self.pool_every - 1:
            pol = torch.nn.MaxPool2d((2,2))
            layers.append(pol)
        # ========================
        seq = nn.Sequential(*layers)
        return seq

    def _make_classifier(self):
        in_channels, in_h, in_w = tuple(self.in_size)

        layers = []
        # TODO: Create the classifier part of the model:
        # (Linear -> ReLU)*M -> Linear
        # You'll need to calculate the number of features first.
        # The last Linear layer should have an output dimension of out_classes.
        # Hint: use loop for len(self.hidden_dims) and append the layers you need to list named layers.
        # ====== YOUR CODE: ======
        N = len(self.filters)
        P = self.pool_every
        decrease = N//P

        for _ in range(decrease):
          in_h = in_h//2
          in_w = in_w//2
        in_channels = self.filters[-1]

        features = self.hidden_dims
        features = [in_channels*in_h*in_w] + features + [self.out_classes]
        for layer_idx in range(len(features)-1):
          layer = torch.nn.Linear(features[layer_idx],features[layer_idx+1])
          layers.append(layer)
          activation = torch.nn.ReLU()
          if layer_idx+2 < len(features):
            layers.append(activation)
        # ========================
        seq = nn.Sequential(*layers)
        return seq

    def forward(self, x):
        # TODO: Implement the forward pass.
        # Extract features from the input (using self.feature_extractor), flatten your result (using torch.flatten),
        # run the classifier on them (using self.classifier) and return class scores.
        # ====== YOUR CODE: ======
        extractor = self.feature_extractor
        classifier = self.classifier

        if len(x.shape) == 4: # batch
          out = classifier(torch.flatten(extractor(x), start_dim=1))
        else: # single image
          out = classifier(torch.flatten(extractor(x)))
        # ========================
        return out


class YourCodeNet(ConvClassifier):
    def __init__(self, in_size, out_classes, filters, pool_every, hidden_dims, filter_size = (3,3), pooling_size = (2,2)):
        self.filter_size = filter_size
        self.pooling_size = pooling_size
        super().__init__(in_size, out_classes, filters, pool_every, hidden_dims)

    # TODO: Change whatever you want about the ConvClassifier to try to
    # improve it's results on CIFAR-10.
    # For example, add batchnorm, dropout, skip connections, change conv
    # filter sizes etc.
    # ====== YOUR CODE: ======
    def _make_feature_extractor(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        # Implement this function with the fixes you suggested question 1.1. Extra points.
        # ====== YOUR CODE: ======
        filters = [in_channels] + self.filters 

        # builds the layers according to parameters
        for layer_idx in range(len(filters)-1):
          in_f = filters[layer_idx]
          out_f = filters[layer_idx+1]
          # residual block
          layer = SkipConnectionLayer(in_f,out_f)
          if layer_idx>0 and layer_idx<len(filters)-1:
            # dropout
            layers.append(torch.nn.Dropout())
          layers.append(layer)
          if layer_idx % self.pool_every == self.pool_every - 1:
            # max pool
            pol = torch.nn.MaxPool2d(self.pooling_size)
            layers.append(pol)
        # ========================
        seq = nn.Sequential(*layers)
        return seq
    # ========================

# residual block
class SkipConnectionLayer(nn.Module):
    def __init__(self, in_channels, out_channels,filter_size = (3,3), pooling_size = (2,2)):
        super().__init__()
        self.in_ch = in_channels
        self.out_ch = out_channels
        # identety layer
        self.connection = torch.nn.Conv2d(self.in_ch,self.out_ch,(1,1),padding="same")
        
        # convolution block
        self.layer = torch.nn.Sequential(
          torch.nn.LeakyReLU(),
          torch.nn.BatchNorm2d(in_channels),
          torch.nn.Conv2d(in_channels,out_channels,filter_size,padding="same"),
          torch.nn.LeakyReLU(),
          torch.nn.Conv2d(out_channels,out_channels,filter_size,padding="same")
        )

    def forward(self,X):
      return self.connection(X)+self.layer(X)



