import torch
import torch.nn as nn

class depthwise_separable_conv(nn.Module):
    def __init__(self,nin,nout,kernelsize,stride_,padding_):
        super(depthwise_separable_conv,self).__init__()
        self.depthwise = nn.Conv1d(nin,nin,kernel_size=kernelsize,stride=stride_,padding=padding_,groups=nin)
        self.pointwise = nn.Conv1d(nin,nout,kernel_size=1)
    def forward(self,x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class inception_block(nn.Module):
    def __init__(self,nin,nout,pool_size,reduce=True):
        super(inception_block,self).__init__()
        self.reduce = reduce
        self.swish = nn.Hardswish()

        #block1
        self.block1_1 = depthwise_separable_conv(nin,nout,11,1,5)
        self.block1_batch = nn.BatchNorm1d(nout)

        #block2
        self.block2_1 = depthwise_separable_conv(nin,nout,19,1,9)
        self.block2_batch = nn.BatchNorm1d(nout)

        #block3
        self.block3_1 = depthwise_separable_conv(nin,nout,27,1,13)
        self.block3_batch = nn.BatchNorm1d(nout)

        self.maxpool = nn.MaxPool1d(pool_size)

        #block4: Skip Connection
        if self.reduce:
            self.block4_1 = nn.Conv1d(nin,nout,1,pool_size)
            self.block4_batch = nn.BatchNorm1d(nout)

    def forward(self,x):
        block1 = self.block1_1(x)
        block1 = self.block1_batch(block1)

        block2 = self.block2_1(x)
        block2 = self.block2_batch(block2)

        block3 = self.block3_1(x)
        block3 = self.block3_batch(block3)

        z = block1 + block2
        z = z + block3

        if self.reduce:
            block4 = self.block4_1(x)
            block4 = self.block4_batch(block4)
            z = self.maxpool(z)
            z = z + block4
        else:
            z = z + x

        return self.swish(z)

class DSCNN(nn.Module):
    def __init__(self,classes,pool_size):
        super(DSCNN,self).__init__()
        self.classes = classes

        self.embedding = nn.Embedding(256,32)

        self.conv1 = nn.Conv1d(32,32,19,2,9)
        self.swish = nn.Hardswish()

        self.inception_block1 = inception_block(32,64,pool_size)
        self.inception_block2 = inception_block(64,64,pool_size,False)
        self.inception_block3 = inception_block(64,128,pool_size)

        self.adaptivepool = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Conv1d(128,classes,1)
        self.logsoftmax = nn.LogSoftmax(1)

    def forward(self,x):
        #Embedding
        x = self.embedding(x)
        x = x.permute(0,2,1)

        # 1: normal conv
        x = self.conv1(x)
        x = self.swish(x)

        #2: Inception Block 1
        x = self.inception_block1(x)

        #3: Inception Block 2
        x = self.inception_block2(x)

        #4: Inception Block 3
        x = self.inception_block3(x)

        x = self.adaptivepool(x)
        x = self.classifier(x)
        x = self.logsoftmax(x)

        return x.view(-1,self.classes)