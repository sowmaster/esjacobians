
# Simple adaptation of the Resnet12 from learn2learn to output features vector

import torch
import torch.nn as nn
import torch.nn.functional as F
import learn2learn as l2l


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        drop_rate=0.0,
        drop_block=False,
        block_size=1,
    ):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.LeakyReLU(0.1)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv3x3(planes, planes)
        self.bn3 = nn.BatchNorm2d(planes)
        self.maxpool = nn.MaxPool2d(stride)
        self.downsample = downsample
        self.stride = stride
        self.drop_rate = drop_rate
        self.num_batches_tracked = 0
        self.drop_block = drop_block
        self.block_size = block_size
        self.DropBlock = DropBlock(block_size=self.block_size)

    def forward(self, x):
        self.num_batches_tracked += 1

        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        out = self.maxpool(out)

        if self.drop_rate > 0:
            if self.drop_block:
                feat_size = out.size()[2]
                keep_rate = max(
                    1.0 - self.drop_rate / 40000 * self.num_batches_tracked,
                    1.0 - self.drop_rate
                )
                gamma = (
                    (1 - keep_rate)
                    / self.block_size**2 * feat_size**2
                    / (feat_size - self.block_size + 1)**2
                )
                out = self.DropBlock(out, gamma=gamma)
            else:
                out = F.dropout(
                    out,
                    p=self.drop_rate,
                    training=self.training,
                    inplace=True,
                )
        return out


class DropBlock(nn.Module):
    def __init__(self, block_size):
        super(DropBlock, self).__init__()
        self.block_size = block_size

    def forward(self, x, gamma):

        if self.training:
            batch_size, channels, height, width = x.shape

            bernoulli = torch.distributions.Bernoulli(gamma)
            mask = bernoulli.sample((
                batch_size,
                channels,
                height - (self.block_size - 1),
                width - (self.block_size - 1),
            )).to(x.device)
            block_mask = self._compute_block_mask(mask)
            countM = (
                block_mask.size(0)
                * block_mask.size(1)
                * block_mask.size(2)
                * block_mask.size(3)
            )
            count_ones = block_mask.sum()
            return block_mask * x * (countM / count_ones)
        else:
            return x

    def _compute_block_mask(self, mask):
        left_padding = int((self.block_size-1) / 2)
        right_padding = int(self.block_size / 2)

        batch_size, channels, height, width = mask.shape
        non_zero_idxs = mask.nonzero(as_tuple=False)
        nr_blocks = non_zero_idxs.shape[0]

        offsets = torch.stack(
            [
                torch.arange(self.block_size).view(-1, 1).expand(
                    self.block_size,
                    self.block_size).reshape(-1),
                torch.arange(self.block_size).repeat(self.block_size),
            ]
        ).t()
        offsets = torch.cat(
            (torch.zeros(self.block_size**2, 2).long(), offsets.long()),
            dim=1,
        ).to(mask.device)

        if nr_blocks > 0:
            non_zero_idxs = non_zero_idxs.repeat(self.block_size ** 2, 1)
            offsets = offsets.repeat(nr_blocks, 1).view(-1, 4)
            offsets = offsets.long()

            block_idxs = non_zero_idxs + offsets
            padded_mask = F.pad(
                mask,
                (left_padding, right_padding, left_padding, right_padding)
            )
            padded_mask[
                block_idxs[:, 0],
                block_idxs[:, 1],
                block_idxs[:, 2],
                block_idxs[:, 3]] = 1.0
        else:
            padded_mask = F.pad(
                mask,
                (left_padding, right_padding, left_padding, right_padding)
            )

        block_mask = 1 - padded_mask
        return block_mask


class ResNet12(nn.Module):

    """
    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/vision/models/resnet12.py)
    **Description**
    The 12-layer residual network from Mishra et al, 2017.
    The code is adapted from [Lee et al, 2019](https://github.com/kjunelee/MetaOptNet/)
    who share it under the Apache 2 license.
    List of changes:
    * Rename ResNet to ResNet12.
    * Small API modifications.
    * Fix code style to be compatible with PEP8.
    * Support multiple devices in DropBlock
    **References**
    1. Mishra et al. 2017. “A Simple Neural Attentive Meta-Learner.” ICLR 18.
    2. Lee et al. 2019. “Meta-Learning with Differentiable Convex Optimization.” CVPR 19.
    3. Lee et al's code: [https://github.com/kjunelee/MetaOptNet/](https://github.com/kjunelee/MetaOptNet/)
    4. Oreshkin et al. 2018. “TADAM: Task Dependent Adaptive Metric for Improved Few-Shot Learning.” NeurIPS 18.
    **Arguments**
    * **output_size** (int) - The dimensionality of the output.
    * **hidden_size** (list, *optional*, default=640) - Size of the embedding once features are extracted.
        (640 is for mini-ImageNet; used for the classifier layer)
    * **keep_prob** (float, *optional*, default=1.0) - Dropout rate on the embedding layer.
    * **avg_pool** (bool, *optional*, default=True) - Set to False for the 16k-dim embeddings of Lee et al, 2019.
    * **drop_rate** (float, *optional*, default=0.1) - Dropout rate for the residual layers.
    * **dropblock_size** (int, *optional*, default=5) - Size of drop blocks.
    **Example**
    ~~~python
    model = ResNet12(output_size=ways, hidden_size=1600, avg_pool=False)
    ~~~
    """

    def __init__(
        self,
        keep_prob=1.0,  # dropout for embedding
        avg_pool=True,  # Set to False for 16000-dim embeddings
        drop_rate=0.0,  # dropout for residual layers
        dropblock_size=5,
    ):
        super(ResNet12, self).__init__()
        self.inplanes = 3
        #self.output_size = output_size
        block = BasicBlock

        self.layer1 = self._make_layer(
            block,
            64,
            stride=2,
            drop_rate=drop_rate,
        )
        self.layer2 = self._make_layer(
            block,
            160,
            stride=2,
            drop_rate=drop_rate,
        )
        self.layer3 = self._make_layer(
            block,
            320,
            stride=2,
            drop_rate=drop_rate,
            drop_block=True,
            block_size=dropblock_size,
        )
        self.layer4 = self._make_layer(
            block,
            640,
            stride=2,
            drop_rate=drop_rate,
            drop_block=True,
            block_size=dropblock_size,
        )
        if avg_pool:
            self.avgpool = nn.AvgPool2d(5, stride=1)
        else:
            self.avgpool = l2l.nn.Lambda(lambda x: x)
        self.keep_prob = keep_prob
        self.keep_avg_pool = avg_pool
        self.dropout = nn.Dropout(p=1.0 - self.keep_prob, inplace=False)
        self.drop_rate = drop_rate

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight,
                    mode='fan_out',
                    nonlinearity='leaky_relu',
                )
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        self.features = torch.nn.Sequential(
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4,
            self.avgpool,
            l2l.nn.Flatten(),
            self.dropout,
        )
        #self.classifier = torch.nn.Linear(hidden_size, output_size)

    def _make_layer(
        self,
        block,
        planes,
        stride=1,
        drop_rate=0.0,
        drop_block=False,
        block_size=1,
    ):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(
            self.inplanes,
            planes,
            stride,
            downsample,
            drop_rate,
            drop_block,
            block_size)
        )
        self.inplanes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        #x = self.classifier(x)
        return x


if __name__ == '__main__':
    model = ResNet12(avg_pool=True, drop_rate=0.0, dropblock_size=5)
    img = torch.randn(5, 3, 84, 84)
    #model = model.to('cuda')
    #img = img.to('cuda')

    print('model is : ', model.__class__.__name__)

    out = model(img)
    print(out.shape)
    numparams = sum([torch.numel(param) for param in model.parameters()])
    print('num of hyperparameters to be learned: ', numparams)
    __import__('pdb').set_trace()