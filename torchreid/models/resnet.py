"""
Code source: https://github.com/pytorch/vision
"""
from __future__ import division, absolute_import
import torch.utils.model_zoo as model_zoo
from torch import nn
import copy
#from .unet_var import UNetWithVarEncoder
from torch import nn
import torch.nn.functional as F
__all__ = ['resnet50', 'unet50', 'unetvar50','resnetvae','pcb_p6','multigrain']
import torchvision.models as models
model_urls = {'resnet50':'https://download.pytorch.org/models/resnet50-19c8e357.pth'}

import torch
from torch.nn import init
import torch.backends.cudnn as cudnn
from torchvision.models.resnet import resnet50 as tvresnet50, Bottleneck
from torchvision.models.resnet import  Bottleneck
cudnn.benchmark = False
class Interpolate(nn.Module):
    def __init__(self, size, mode):
        super(Interpolate, self).__init__()
        self.interp = F.upsample
        self.size = size
        self.mode = mode
        
    def forward(self, x):
        x = self.interp(x, size=self.size, mode=self.mode)
        return x
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)

### Squeeze and Excitation Class definition
class SENet(nn.Module):
    def __init__(self, channel, reduction_ratio =16):
        super(SENet, self).__init__()
        ### Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        ### Fully Connected Multi-Layer Perceptron (FC-MLP)
        self.mlp = nn.Sequential(
            nn.Linear(channel, channel // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction_ratio, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c = x.size()
        y = x.view(b, c)
        y = self.mlp(y).view(b, c, 1, 1)
        return x * y.view(b,c)


class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, dropout=True, relu=True, num_bottleneck=512):
        super(ClassBlock, self).__init__()
        self.dropout=nn.Dropout(0.5)
        add_block = []
        add_block += [nn.Linear(input_dim, num_bottleneck)] 
        add_block += [nn.BatchNorm1d(num_bottleneck)]
        #add_block += [nn.LayerNorm(num_bottleneck)]

        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if dropout:
            add_block += [nn.Dropout(p=0.5)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier
    def forward(self, x):
        x = self.dropout(x)
        x = self.add_block(x)
        x = self.classifier(x)
        return x

class ResNet_VAE(nn.Module):
    def __init__(self,
        num_classes,
        loss,
        block,
        layers,
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        norm_layer=None,
        last_stride=2,  # was 2 initially
        fc_dims=None,
        dropout_p=None,
        **kwargs
    ):
        super(ResNet_VAE, self).__init__()
        fc_hidden1=1024
        fc_hidden2=768
        drop_p=0.3
        CNN_embed_dim=256
        self.feature_dim=2048
        self.fc_hidden1, self.fc_hidden2, self.CNN_embed_dim = fc_hidden1, fc_hidden2, CNN_embed_dim

        # CNN architechtures
        self.ch1, self.ch2, self.ch3, self.ch4 = 16, 32, 64, 128
        self.k1, self.k2, self.k3, self.k4 = (5, 5), (3, 3), (3, 3), (3, 3)      # 2d kernal size
        self.s1, self.s2, self.s3, self.s4 = (2, 2), (2, 2), (2, 2), (2, 2)      # 2d strides
        self.pd1, self.pd2, self.pd3, self.pd4 = (0, 0), (0, 0), (0, 0), (0, 0)  # 2d padding

        # encoding components
        resnet = models.resnet50(pretrained=True)
      
        resnet.layer4[0].conv2.stride = (1,1)
        resnet.layer4[0].downsample[0].stride = (1,1)

        modules = list(resnet.children())[:-2]      # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        self.fc1 = nn.Linear(resnet.fc.in_features, self.fc_hidden1)
        self.bn1 = nn.BatchNorm1d(self.fc_hidden1, momentum=0.01)
        self.fc2 = nn.Linear(self.fc_hidden1, self.fc_hidden2)
        self.bn2 = nn.BatchNorm1d(self.fc_hidden2, momentum=0.01)
        # Latent vectors mu and sigma
        self.fc3_mu = nn.Linear(self.fc_hidden2, self.CNN_embed_dim)      # output = CNN embedding latent variables
        self.fc3_logvar = nn.Linear(self.fc_hidden2, self.CNN_embed_dim)  # output = CNN embedding latent variables

        # Sampling vector
        self.fc4 = nn.Linear(self.CNN_embed_dim, self.fc_hidden2)
        self.fc_bn4 = nn.BatchNorm1d(self.fc_hidden2)
        self.fc5 = nn.Linear(self.fc_hidden2, 64 * 16 * 4)
        self.fc_bn5 = nn.BatchNorm1d(64 * 16 * 4)
        self.relu = nn.ReLU(inplace=True)

        # Decoder
        self.convTrans6 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=self.k4, stride=self.s4,
                               padding=self.pd4),
            nn.BatchNorm2d(32, momentum=0.01),
            nn.ReLU(inplace=True),
        )
        self.convTrans7 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32, out_channels=8, kernel_size=self.k3, stride=self.s3,
                               padding=self.pd3),
            nn.BatchNorm2d(8, momentum=0.01),
            nn.ReLU(inplace=True),
        )

        self.convTrans8 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=8, out_channels=3, kernel_size=self.k2, stride=self.s2,
                               padding=self.pd2),
            nn.BatchNorm2d(3, momentum=0.01),
            nn.Sigmoid()    # y = (y1, y2, y3) \in [0 ,1]^3
        )
        planes = 2048
        local_conv_out_channels=256
        self.global_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier =ClassBlock(self.feature_dim, num_classes)# nn.Linear(self.feature_dim, num_classes)
        self.local_conv = nn.Conv2d(planes, local_conv_out_channels, 1)
        self.local_conv3 = nn.Conv2d(planes, 1, 1)
       
        #self.local_conv2 = nn.Conv2d(planes, local_conv_out_channels, 1)
        self.local_bn = nn.BatchNorm2d(local_conv_out_channels)
        #self.local_bn2 = nn.BatchNorm2d(local_conv_out_channels)
        self.local_relu = nn.ReLU(inplace=True)
        self.interpol = Interpolate(size=(16, 8), mode='bilinear')
        self.tanh = nn.Tanh()
        self.probs =  nn.Softmax(dim=1)
        #self.parts_classifier = {}
        self.num_parts =32
        self.num_classes = num_classes
        #for i in range(self.num_parts):
        self.parts_classifier = nn.ModuleDict()
        
        for i in range(self.num_parts):
           self.parts_classifier[str(i)] =ClassBlock(local_conv_out_channels, num_classes,num_bottleneck=128)# nn.Linear(local_conv_out_channels, num_classes)
       

    def encode(self, x):
        x = self.resnet(x)  # ResNet
        
        
        orig_feat = x
        x = self.global_avgpool(orig_feat)
        x = x.view(x.size(0), -1)  # flatten output of conv

        # FC layers
        x = self.bn1(self.fc1(x))
        x = self.relu(x)
        x = self.bn2(self.fc2(x))
        x = self.relu(x)
        # x = F.dropout(x, p=self.drop_p, training=self.training)
        mu, logvar = self.fc3_mu(x), self.fc3_logvar(x)
        return mu, logvar, orig_feat

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = std.data.new(std.size()).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        x = self.relu(self.fc_bn4(self.fc4(z)))
        x = self.relu(self.fc_bn5(self.fc5(x))).view(-1, 64, 16, 4)
        x = self.convTrans6(x)
        x = self.convTrans7(x)
        x = self.convTrans8(x)
        x = F.interpolate(x, size=(384, 128), mode='bilinear')
        return x

    def forward(self, x,return_featuremaps=False):
       
       
        mu, logvar,orig_feat = self.encode(x)
        mask_feat =self.local_conv3(orig_feat)
        mask_feat = self.tanh(mask_feat)
       
        orig_feat = orig_feat*mask_feat
      	
       
        

        v = self.global_avgpool(orig_feat)

        
        v = v.reshape(v.size(0),-1)
       

        z = self.reparameterize(mu, logvar)
       

       
        
       
        #local_feat = torch.mean(orig_feat, -1, keepdim=True) 
        #local_feat2 = torch.mean(orig_feat, 2, keepdim=True) 
        #local_feat2 = local_feat2.reshape(local_feat2.size(0),local_feat2.size(1),local_feat2.size(3),1) 
        #local_feat2 = self.interpol(local_feat2)
       
      
        local_feat = self.local_relu(self.local_bn(self.local_conv(orig_feat)))  
       
        
        #local_feat2 = self.local_relu(self.local_bn2(self.local_conv2(local_feat2)))  
         
        # shape [N, H, c]
        #local_feat = local_feat.squeeze(-1).permute(0, 2, 1)
     
       
        feats =  nn.AdaptiveAvgPool2d((8,1))(local_feat)
        
        
        y = self.classifier(v)
       # local_feat2 = local_feat2.squeeze(-1).permute(0, 2, 1)


       
        local_feat = feats.reshape(feats.size(0),feats.size(1),-1)
        reshaped_local = local_feat.mean(1).reshape(local_feat.size(0), local_feat.size(2))
        #probabilities = self.probs(reshaped_local)
        atten = F.sigmoid(reshaped_local)
        mask = F.normalize(atten, p=1, dim=1)
        #mask_zeros =  torch.zeros_like(probabilities)
        #mask_ones =  torch.ones_like(probabilities)
        
        #mask = torch.where(probabilities > 0.01, mask_ones,mask_zeros) 
      
        mask = mask.reshape(mask.size(0),1, mask.size(1))
        
        local_feat = local_feat*mask
       
        if return_featuremaps==True:
           mask=mask.reshape(mask.size(0),1,8,4)
           mask = self.interpol(mask)
           return orig_feat,mask
        parts_out = {}
        if not self.training and return_featuremaps==False:
            mask = mask.reshape(mask.size(0),-1)
            
            return local_feat,mask,v  
       
        for i in range(local_feat.size(2)):
          
           feat = local_feat[:,:,i]
          
           #feat =feat.squeeze()
          
           a = self.parts_classifier[str(i)](feat)
          
           parts_out[i] = a
         
         
           #part_out.append(self.parts_classifier[i](feat))"""
   
        
        x_reconst = self.decode(z)
      
        
        return y,parts_out,v, x_reconst, z, mu, logvar
        






















def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=1, stride=stride, bias=False
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None
    ):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError(
                'BasicBlock only supports groups=1 and base_width=64'
            )
        if dilation > 1:
            raise NotImplementedError(
                "Dilation > 1 not supported in BasicBlock"
            )
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None
    ):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width/64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """Residual network.
    
    Reference:
        - He et al. Deep Residual Learning for Image Recognition. CVPR 2016.
        - Xie et al. Aggregated Residual Transformations for Deep Neural Networks. CVPR 2017.

    Public keys:
        - ``resnet18``: ResNet18.
        - ``resnet34``: ResNet34.
        - ``resnet50``: ResNet50.
        - ``resnet101``: ResNet101.
        - ``resnet152``: ResNet152.
        - ``resnext50_32x4d``: ResNeXt50.
        - ``resnext101_32x8d``: ResNeXt101.
        - ``resnet50_fc512``: ResNet50 + FC.
    """

    def __init__(
        self,
        num_classes,
        loss,
        block,
        layers,
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        norm_layer=None,
        last_stride=2,  # was 2 initially
        fc_dims=None,
        dropout_p=None,
        **kwargs
    ):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.loss = loss
        self.feature_dim = 512 * block.expansion
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".
                format(replace_stride_with_dilation)
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(
            3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(
            block,
            128,
            layers[1],
            stride=2,
            dilate=replace_stride_with_dilation[0]
        )
        self.layer3 = self._make_layer(
            block,
            256,
            layers[2],
            stride=2,
            dilate=replace_stride_with_dilation[1]
        )
        self.layer4 = self._make_layer(
            block,
            512,
            layers[3],
            stride=last_stride,
            dilate=replace_stride_with_dilation[2]
        )
        self.global_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = self._construct_fc_layer(
            fc_dims, 512 * block.expansion, dropout_p
        )
        self.classifier = nn.Linear(self.feature_dim, num_classes)

        self._init_params()

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)
    
    
    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups,
                self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer
                )
            )

        return nn.Sequential(*layers)

    def _construct_fc_layer(self, fc_dims, input_dim, dropout_p=None):
        """Constructs fully connected layer

        Args:
            fc_dims (list or tuple): dimensions of fc layers, if None, no fc layers are constructed
            input_dim (int): input dimension
            dropout_p (float): dropout probability, if None, dropout is unused
        """
        if fc_dims is None:
            self.feature_dim = input_dim
            return None

        assert isinstance(
            fc_dims, (list, tuple)
        ), 'fc_dims must be either list or tuple, but got {}'.format(
            type(fc_dims)
        )

        layers = []
        for dim in fc_dims:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout_p is not None:
                layers.append(nn.Dropout(p=dropout_p))
            input_dim = dim

        self.feature_dim = fc_dims[-1]

        return nn.Sequential(*layers)

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu'
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def featuremaps(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def forward(self, x,return_featuremaps=False):
        f = self.featuremaps(x)
        if return_featuremaps:
           return f
        v = self.global_avgpool(f)
        v = v.view(v.size(0), -1)

        if self.fc is not None:
            v = self.fc(v)

        if not self.training:
            return v

        y = self.classifier(v)

        if self.loss == 'softmax':
            return y
        elif self.loss == 'triplet':
            return y, v
        elif self.loss == 'mmd':
            return y, v
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))

"""
class Unet50(nn.Module):
   

    def __init__(
        self,
        num_classes,
        loss,
        block,
        layers,
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        norm_layer=None,
        last_stride=2,  # was 2 initially
        fc_dims=None,
        dropout_p=None,
        **kwargs
    ):
        super(Unet50, self).__init__()
        self.loss = loss
        self.unet = UNetWithResnet50Encoder()
        #for mo in resnet50.modules():
        #    if isinstance(mo, nn.Conv2d):
        #        mo.stride = (1,1)

       


        self.fc = self._construct_fc_layer(
            fc_dims, 2048, dropout_p
        )
        self.feature_dim = 2048
        self.feat_dim = 2048
        self.classifier = nn.Linear(self.feature_dim, num_classes) #BNClassifier(2048, num_classes , initialization=True)
        self.global_avgpool = nn.AdaptiveAvgPool2d((1, 1))
    
    def _construct_fc_layer(self, fc_dims, input_dim, dropout_p=None):
       
        if fc_dims is None:
            self.feature_dim = input_dim
            return None

        assert isinstance(
            fc_dims, (list, tuple)
        ), 'fc_dims must be either list or tuple, but got {}'.format(
            type(fc_dims)
        )

        layers = []
        for dim in fc_dims:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout_p is not None:
                layers.append(nn.Dropout(p=dropout_p))
            input_dim = dim

        self.feature_dim = fc_dims[-1]

        return nn.Sequential(*layers)

    def featuremaps(self, x):
        recons,x = self.unet(x)
        
        return recons,x

    def forward(self, x):
        recons,f = self.featuremaps(x)
        v = self.global_avgpool(f)
        v = v.view(v.size(0), -1)

        if self.fc is not None:
            v = self.fc(v)

        if not self.training:
            return v

        y = self.classifier(v)

        if self.loss == 'softmax':
            return y
        elif self.loss == 'triplet':
            return y, v
        elif self.loss == 'mmd':
            return y, v
        elif self.loss == 'recons':
            return y, v , recons
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))

class UnetVar50(nn.Module):
    
    
   


    def __init__(
        self,
        num_classes,
        loss,
        block,
        layers,
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        norm_layer=None,
        last_stride=2,  # was 2 initially
        fc_dims=None,
        dropout_p=None,
        **kwargs
    ):
        super(UnetVar50, self).__init__()
        self.loss = loss
        self.unet = UNetWithVarEncoder()
        #for mo in resnet50.modules():
        #    if isinstance(mo, nn.Conv2d):
        #        mo.stride = (1,1)

       


        self.fc = self._construct_fc_layer(
            fc_dims, 2048, dropout_p
        )
        self.feature_dim = 2048
        self.feat_dim = 2048
        self.classifier = nn.Linear(self.feature_dim, num_classes) #BNClassifier(2048, num_classes , initialization=True)
        self.global_avgpool = nn.AdaptiveAvgPool2d((1, 1))
    
    def _construct_fc_layer(self, fc_dims, input_dim, dropout_p=None):
       
        if fc_dims is None:
            self.feature_dim = input_dim
            return None

        assert isinstance(
            fc_dims, (list, tuple)
        ), 'fc_dims must be either list or tuple, but got {}'.format(
            type(fc_dims)
        )

        layers = []
        for dim in fc_dims:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout_p is not None:
                layers.append(nn.Dropout(p=dropout_p))
            input_dim = dim

        self.feature_dim = fc_dims[-1]

        return nn.Sequential(*layers)

    def featuremaps(self, x):
        recons,x,mean_f, var_f,resnet_out = self.unet(x)
        
        return recons,x,mean_f, var_f,resnet_out

    def forward(self, x,return_featuremaps=False):
        recons,f,mean_f, var_f,resnet_out = self.featuremaps(x)
        v = self.global_avgpool(f)
        v = v.view(v.size(0), -1)

        if return_featuremaps==True:
           return resnet_out

        if self.fc is not None:
            v = self.fc(v)

        if not self.training:
            return v

        y = self.classifier(v)

      
        return y, v , recons, mean_f, var_f
      
def init_pretrained_weights(model, model_url):
   
    model_dict = model.state_dict()
    pretrain_dict = {
        k: v
        for k, v in pretrain_dict.items()
        if k in model_dict and model_dict[k].size() == v.size()
    }
    model_dict.update(pretrain_dict)
    model.load_state_dict(model_dict)

"""
def resnet50(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = ResNet(
        num_classes=num_classes,
        loss=loss,
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        last_stride=2,
        fc_dims=None,
        dropout_p=None,
        **kwargs
    )
    if pretrained:
        init_pretrained_weights(model, model_urls['resnet50'])
    return model



def unet50(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = Unet50(
        num_classes=num_classes,
        loss=loss,
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        last_stride=2,
        fc_dims=None,
        dropout_p=None,
        **kwargs
    )
  
    return model
def unetvar50(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = UnetVar50(
        num_classes=num_classes,
        loss=loss,
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        last_stride=2,
        fc_dims=None,
        dropout_p=None,
        **kwargs
    )
  
    return model

def unetvar50(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = UnetVar50(
        num_classes=num_classes,
        loss=loss,
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        last_stride=2,
        fc_dims=None,
        dropout_p=None,
        **kwargs
    )
  
    return model

def resnetvae(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = ResNet_VAE(
        num_classes=num_classes,
        loss=loss,
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        last_stride=2,
        fc_dims=None,
        dropout_p=None,
        **kwargs
    )
  
    return model


class PCB(nn.Module):
    """Part-based Convolutional Baseline.
    Reference:
        Sun et al. Beyond Part Models: Person Retrieval with Refined
        Part Pooling (and A Strong Convolutional Baseline). ECCV 2018.
    Public keys:
        - ``pcb_p4``: PCB with 4-part strips.
        - ``pcb_p6``: PCB with 6-part strips.
    """

    def __init__(self, num_classes, loss, block, layers,
                 parts=6,
                 reduced_dim=256,
                 nonlinear='relu',
                 **kwargs):
        self.inplanes = 64
        super(PCB, self).__init__()
        self.loss = loss
        self.parts = parts
        self.feature_dim = 512 * block.expansion

        # backbone network
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1)
       
        self.ch1, self.ch2, self.ch3, self.ch4 = 16, 32, 64, 128
        self.k1, self.k2, self.k3, self.k4 = (5, 5), (3, 3), (3, 3), (3, 3)      # 2d kernal size
        self.s1, self.s2, self.s3, self.s4 = (2, 2), (2, 2), (2, 2), (2, 2)      # 2d strides
        self.pd1, self.pd2, self.pd3, self.pd4 = (0, 0), (0, 0), (0, 0), (0, 0)  # 2d padding

        # pcb layers
        self.parts_avgpool = nn.AdaptiveAvgPool2d((self.parts, 1))
        self.global_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.5)
        self.em = nn.ModuleList(
            [self._construct_em_layer(reduced_dim, 512 * block.expansion) for _ in range(self.parts)])
        self.feature_dim = reduced_dim
        self.classifier = nn.ModuleList(
            [nn.Linear(self.feature_dim, num_classes, bias=False) for _ in range(self.parts)])

        self._init_params()
        self.max_p = nn.MaxPool1d(256)
        self.CNN_embed_dim =2048
        self.fc_hidden2 = 1024
        self.fc4 = nn.Linear(self.CNN_embed_dim, self.fc_hidden2)
        self.fc_bn4 = nn.BatchNorm1d(self.fc_hidden2)
        self.fc5 = nn.Linear(self.fc_hidden2, 64 * 24 * 8)
        self.fc_bn5 = nn.BatchNorm1d(64 * 24 * 8)
        self.relu = nn.ReLU(inplace=True)
        self.mlp1_feat = nn.ModuleList([self._construct_weight_layers(2048) for _ in range(self.parts)])
        self.probs =  nn.Softmax(dim=1)
        
        # Decoder
        self.convTrans6 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=self.k4, stride=self.s4,
                               padding=self.pd4),
            nn.BatchNorm2d(32, momentum=0.01),
            nn.ReLU(inplace=True),
        )
        self.convTrans7 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32, out_channels=8, kernel_size=self.k3, stride=self.s3,
                               padding=self.pd3),
            nn.BatchNorm2d(8, momentum=0.01),
            nn.ReLU(inplace=True),
        )

        self.convTrans8 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=8, out_channels=3, kernel_size=self.k2, stride=self.s2,
                               padding=self.pd2),
            nn.BatchNorm2d(3, momentum=0.01),
            nn.Sigmoid()    # y = (y1, y2, y3) \in [0 ,1]^3
        )

    def decode(self, z):
        
        x = self.relu(self.fc_bn4(self.fc4(z)))
        x = self.relu(self.fc_bn5(self.fc5(x))).view(-1, 64, 24, 8)
        
        x = self.convTrans6(x)
        x = self.convTrans7(x)
        x = self.convTrans8(x)
        x = F.interpolate(x, size=(384, 128), mode='bilinear')
        return x

    def _construct_weight_layers(self, input_dim, dropout_p=0.5):
        layers = []
       
        # layers.append(nn.Linear(input_dim, fc_dims))
        layers.append(nn.Conv2d(2048, 2048, 1, stride=1, padding=0))
        layers.append(nn.BatchNorm2d(2048, momentum=0.01))
        layers.append( nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(2048, 2048, 1, stride=1, padding=0))
        layers.append(nn.Sigmoid())
        return nn.Sequential(*layers)

    def _construct_em_layer(self, fc_dims, input_dim, dropout_p=0.5):
        """
        Construct fully connected layer
        - fc_dims (list or tuple): dimensions of fc layers, if None,
                                   no fc layers are constructed
        - input_dim (int): input dimension
        - dropout_p (float): dropout probability, if None, dropout is unused
        """
        
        if fc_dims is None:
            self.feature_dim = input_dim
            return None

        layers = []
      
        # layers.append(nn.Linear(input_dim, fc_dims))
        layers.append(nn.Conv2d(input_dim, fc_dims, 1, stride=1, padding=0))
        layers.append(nn.BatchNorm2d(fc_dims))
        layers.append(nn.ReLU(inplace=True))
        # layers.append(nn.Dropout(p=dropout_p))

        # self.feature_dim = fc_dims[-1]

        return nn.Sequential(*layers)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def featuremaps(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def forward(self, x,return_featuremaps=False):
        f = self.featuremaps(x)
        if return_featuremaps:
           return f
        v = self.global_avgpool(f)
        v = v.view(v.size(0), -1)
        recons = self.decode(v)

        # vis_featmat_Kmeans(f)
        # vis_featmat_DBSCAN(f)
        v_g = self.parts_avgpool(f)
        
        

        # v_g = self.dropout(v_g)
        # v_h = self.conv5(v_g)

        y = []
        v = []
        #print(v_g.shape)
        #print(v_g.shape)
        # v_g = v_g.reshape(v_g.size(0), v_g.size(1),-1)
        #v_gg = self.max_p(v_g)
        #print(v_gg.shape)
        #v_gg = torch.max(v_g, 3, keepdim=True)
        
        #v_gg = v_gg.reshape(v_gg.size(0),v_gg.size(1),v_gg.size(2),1)
       
        
        mask_da = torch.ones(v_g.size(0),6,2048).cuda()
        mask = torch.ones(v_g.size(0),6).cuda()
        for i in range(self.parts):
            v_g_i = v_g[:, :, i, :].reshape(v_g.size(0), -1, 1, 1)
           
            a = self.mlp1_feat[i](v_g_i)
            
            
            a = a.squeeze()
            
            mask_da[:,i,:] = a
            
        mask_da = mask_da.reshape(mask_da.size(0),1,-1)
        mask_da = mask_da.reshape(mask_da.size(0), 2048,6)
        v_gg = v_g.reshape(v_g.size(0), 2048,6)
        v_gg = v_gg*mask_da
        v_g = v_gg
        v_g = v_g.reshape(v_g.size(0),2048,6,1)
        for i in range(self.parts):
            v_g_i = v_g[:, :, i, :].reshape(v_g.size(0), -1, 1, 1)
            
            v_g_i = self.em[i](v_g_i)
            v_h_i = v_g_i.view(v_g_i.size(0), -1)
          
            mask[:,i] = v_h_i.mean(dim=1)#v_h_i.max(1)
            y_i = self.classifier[i](v_h_i)
            y.append(y_i)
            v.append(v_g_i)
        mask = torch.ones((v_g.size(0),6)).cuda()
        mask=mask.reshape(mask.size(0),6)
        #atten = self.probs(mask)
        #atten = F.sigmoid(mask)
        #mask = F.normalize(atten, p=1, dim=1)
        if not self.training:
            v_g = v_g.reshape(v_g.size(0),2048,6)
            v_gg = F.normalize(v_g, p=2, dim=1)
            
            #mask = torch.ones((v_gg.size(0),6)).cuda()
            return v_gg, mask
      
        if self.loss == 'softmax':
            return y
        elif self.loss == 'triplet':
            #mask_da = mask_da.reshape(mask_da.size(0), 2048,6,1)
            
            #v_ggda = v_g * mask_da
            v_g = F.normalize(v_gg, p=2, dim=1)
            return y, v_g.view(v_g.size(0), -1), recons,v_g #, v_ggda.view(v_ggda.size(0), -1)v_g.view(v_g.size(0), -1)
        else:
            raise KeyError('Unsupported loss: {}'.format(self.loss))


def init_pretrained_weights(model, model_url):
    """Initializes model with pretrained weights.
    Layers that don't match with pretrained layers in name or size are kept unchanged.
    """
    pretrain_dict = model_zoo.load_url(model_url)
    model_dict = model.state_dict()
    pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
    model_dict.update(pretrain_dict)
    model.load_state_dict(model_dict)



class DimReduceLayer(nn.Module):

    def __init__(self, in_channels, out_channels, nonlinear):
        super(DimReduceLayer, self).__init__()
        layers = []
        layers.append(
            nn.Conv2d(
                in_channels, out_channels, 1, stride=1, padding=0, bias=False
            )
        )
        layers.append(nn.BatchNorm2d(out_channels))

        if nonlinear == 'relu':
            layers.append(nn.ReLU(inplace=True))
        elif nonlinear == 'leakyrelu':
            layers.append(nn.LeakyReLU(0.1))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class PCBT(nn.Module):
    """Part-based Convolutional Baseline.

    Reference:
        Sun et al. Beyond Part Models: Person Retrieval with Refined
        Part Pooling (and A Strong Convolutional Baseline). ECCV 2018.

    Public keys:
        - ``pcb_p4``: PCB with 4-part strips.
        - ``pcb_p6``: PCB with 6-part strips.
    """

    def __init__(
        self,
        num_classes,
        loss,
        block,
        layers,
        parts=6,
        reduced_dim=256,
        nonlinear='relu',
        **kwargs
    ):
        self.inplanes = 64
        super(PCBT, self).__init__()
        self.loss = loss
        self.parts = parts
        self.feature_dim = 512 * block.expansion
        
        # backbone network
        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1)

        # pcb layers
        self.parts_avgpool = nn.AdaptiveAvgPool2d((self.parts, 1))
        self.dropout = nn.Dropout(p=0.5)
        self.conv5 = DimReduceLayer(
            512 * block.expansion, reduced_dim, nonlinear=nonlinear
        )
        self.feature_dim = reduced_dim
        self.classifier = nn.ModuleList(
            [
                nn.Linear(self.feature_dim, num_classes)
                for _ in range(self.parts)
            ]
        )

        self._init_params()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu'
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def featuremaps(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def forward(self, x):
        f = self.featuremaps(x)
        v_g = self.parts_avgpool(f)

        if not self.training:
            v_g = F.normalize(v_g, p=2, dim=1)
            return v_g.view(v_g.size(0), -1)

        v_g = self.dropout(v_g)
        v_h = self.conv5(v_g)

        y = []
        for i in range(self.parts):
            v_h_i = v_h[:, :, i, :]
            v_h_i = v_h_i.view(v_h_i.size(0), -1)
            y_i = self.classifier[i](v_h_i)
            y.append(y_i)

        if self.loss == 'softmax':
            return y
        elif self.loss == 'triplet':
            v_g = F.normalize(v_g, p=2, dim=1)
            return y, v_g.view(v_g.size(0), -1), v_h.view(v_h.size(0), -1)
        else:
            raise KeyError('Unsupported loss: {}'.format(self.loss))



def init_pretrained_weights(model, model_url):
    """Initializes model with pretrained weights.
    
    Layers that don't match with pretrained layers in name or size are kept unchanged.
    """
    pretrain_dict = model_zoo.load_url(model_url)
    model_dict = model.state_dict()
    pretrain_dict = {
        k: v
        for k, v in pretrain_dict.items()
        if k in model_dict and model_dict[k].size() == v.size()
    }
    model_dict.update(pretrain_dict)
    model.load_state_dict(model_dict)

class FC_Model_Binary(nn.Module):
    def __init__(self):
        super(FC_Model_Binary, self).__init__()
        feats = 256
        student_classes=1
        self.fc_binary1 = nn.Linear(feats, student_classes)
        self.fc_mlp = nn.Linear(feats, feats)
        self.relu = nn.ReLU()
      
        self._init_fc(self.fc_binary1)
        

    @staticmethod
    def _init_fc(fc):
        nn.init.kaiming_normal_(fc.weight, mode='fan_out')
        # nn.init.normal_(fc.weight, std=0.001)
        nn.init.constant_(fc.bias, 0.)
    
    def forward(self, inp):
        intermediateA = self.fc_mlp(inp)
        mlpout=self.relu(intermediateA)
        ret = self.fc_binary1(mlpout)
    
        return ret,intermediateA


class MGN(nn.Module):
    def __init__(self, num_classes, **kwargs):
        super(MGN, self).__init__()

        
        
        feats = 256
        resnet = tvresnet50(pretrained=True) #torchvision resnet50

        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3[0],
        )
        self.sig = nn.Sigmoid()

        res_conv4 = nn.Sequential(*resnet.layer3[1:])

        res_g_conv5 = resnet.layer4

        res_p_conv5 = nn.Sequential(
            Bottleneck(1024, 512, downsample=nn.Sequential(nn.Conv2d(1024, 2048, 1, bias=False), nn.BatchNorm2d(2048))),
            Bottleneck(2048, 512),
            Bottleneck(2048, 512))
        res_p_conv5.load_state_dict(resnet.layer4.state_dict())
        
       
        self.pae = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_g_conv5))
        self.pmmd = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_g_conv5))
        self.p1 = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_g_conv5))
        self.p2 = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_p_conv5))
        self.p3 = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_p_conv5))

        self.maxpool_zg_p1 = nn.MaxPool2d(kernel_size=(12, 4))
        self.maxpool_zg_p2 = nn.MaxPool2d(kernel_size=(24, 8))
        self.maxpool_zg_p3 = nn.MaxPool2d(kernel_size=(24, 8))
        self.maxpool_zp2 = nn.MaxPool2d(kernel_size=(12, 8))
        self.maxpool_zp3 = nn.MaxPool2d(kernel_size=(8, 8))

        self.reduction = nn.Sequential(nn.Conv2d(2048, feats, 1, bias=False), nn.BatchNorm2d(feats), nn.ReLU())

        self._init_reduction(self.reduction)

        self.fc_id_2048_0 = nn.Linear(feats, num_classes)
        self.fc_id_2048_1 = nn.Linear(feats, num_classes)
        self.fc_id_2048_2 = nn.Linear(feats, num_classes)

        self.fc_id_256_1_0 = nn.Linear(feats, num_classes)
        self.fc_id_256_1_1 = nn.Linear(feats, num_classes)
        self.fc_id_256_2_0 = nn.Linear(feats, num_classes)
        self.fc_id_256_2_1 = nn.Linear(feats, num_classes)
        self.fc_id_256_2_2 = nn.Linear(feats, num_classes)

        self._init_fc(self.fc_id_2048_0)
        self._init_fc(self.fc_id_2048_1)
        self._init_fc(self.fc_id_2048_2)

        self._init_fc(self.fc_id_256_1_0)
        self._init_fc(self.fc_id_256_1_1)
        self._init_fc(self.fc_id_256_2_0)
        self._init_fc(self.fc_id_256_2_1)
        self._init_fc(self.fc_id_256_2_2)
        
        self.senet1 = SENet(channel=256)
        self.senet2 = SENet(channel=256)
        self.senet3 = SENet(channel=256)
        self.ch1, self.ch2, self.ch3, self.ch4 = 16, 32, 64, 128
        self.k1, self.k2, self.k3, self.k4 = (5, 5), (3, 3), (3, 3), (3, 3)      # 2d kernal size
        self.s1, self.s2, self.s3, self.s4 = (2, 2), (2, 2), (2, 2), (2, 2)      # 2d strides
        self.pd1, self.pd2, self.pd3, self.pd4 = (0, 0), (0, 0), (0, 0), (0, 0)  # 2d padding

        # pcb layers
        #self.parts_avgpool = nn.AdaptiveAvgPool2d((self.parts, 1))
        self.global_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.5)
        
        #self.feature_dim = reduced_dim
       
        self.max_p = nn.MaxPool1d(256)
        self.CNN_embed_dim =2048
        self.fc_hidden2 = 1024
        self.fc4 = nn.Linear(self.CNN_embed_dim, self.fc_hidden2)
        self.fc_bn4 = nn.BatchNorm1d(self.fc_hidden2)
        self.fc5 = nn.Linear(self.fc_hidden2, 64 * 24 * 8)
        self.fc_bn5 = nn.BatchNorm1d(64 * 24 * 8)
        self.relu = nn.ReLU(inplace=True)
       
        
        # Decoder
        self.convTrans6 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=self.k4, stride=self.s4,
                               padding=self.pd4),
            nn.BatchNorm2d(32, momentum=0.01),
            nn.ReLU(inplace=True),
        )
        self.convTrans7 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32, out_channels=8, kernel_size=self.k3, stride=self.s3,
                               padding=self.pd3),
            nn.BatchNorm2d(8, momentum=0.01),
            nn.ReLU(inplace=True),
        )

        self.convTrans8 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=8, out_channels=3, kernel_size=self.k2, stride=self.s2,
                               padding=self.pd2),
            nn.BatchNorm2d(3, momentum=0.01),
            nn.Sigmoid()    # y = (y1, y2, y3) \in [0 ,1]^3
        )
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.Binary_FC = FC_Model_Binary()

    def decode(self, z):
        
        x = self.relu(self.fc_bn4(self.fc4(z)))
        x = self.relu(self.fc_bn5(self.fc5(x))).view(-1, 64, 24, 8)
        
        x = self.convTrans6(x)
        x = self.convTrans7(x)
        x = self.convTrans8(x)
        x = F.interpolate(x, size=(384, 128), mode='bilinear')
        return x


    @staticmethod
    def _init_reduction(reduction):
        # conv
        nn.init.kaiming_normal_(reduction[0].weight, mode='fan_in')
        # nn.init.constant_(reduction[0].bias, 0.)

        # bn
        nn.init.normal_(reduction[1].weight, mean=1., std=0.02)
        nn.init.constant_(reduction[1].bias, 0.)

    @staticmethod
    def _init_fc(fc):
        nn.init.kaiming_normal_(fc.weight, mode='fan_out')
        # nn.init.normal_(fc.weight, std=0.001)
        nn.init.constant_(fc.bias, 0.)

    def forward(self, x,return_featuremaps=False):
        x = self.backbone(x)

        p1 = self.p1(x)
        p2 = self.p2(x)
        p3 = self.p3(x)
        pae   = self.pae(x)
        if return_featuremaps:
           return p2
        

        reconsfeat = self.gap(pae)

        zg_p1 = self.maxpool_zg_p1(p1)
        zg_p2 = self.maxpool_zg_p2(p2)
        zg_p3 = self.maxpool_zg_p3(p3)

        zp2 = self.maxpool_zp2(p2)
        z0_p2 = zp2[:, :, 0:1, :]
        z1_p2 = zp2[:, :, 1:2, :]

        zp3 = self.maxpool_zp3(p3)
        z0_p3 = zp3[:, :, 0:1, :]
        z1_p3 = zp3[:, :, 1:2, :]
        z2_p3 = zp3[:, :, 2:3, :]

        reconsfeat = reconsfeat.view(reconsfeat.size(0), -1)
        recons = self.decode(reconsfeat)
        reconsfeat=reconsfeat.view(reconsfeat.size(0), 2048,1,1)
        fg_recons = self.reduction(reconsfeat).squeeze(dim=3).squeeze(dim=2)
        fg_p1 = self.reduction(zg_p1).squeeze(dim=3).squeeze(dim=2)
        fg_p2 = self.reduction(zg_p2).squeeze(dim=3).squeeze(dim=2)
        fg_p3 = self.reduction(zg_p3).squeeze(dim=3).squeeze(dim=2)
        f0_p2 = self.reduction(z0_p2).squeeze(dim=3).squeeze(dim=2)
        f1_p2 = self.reduction(z1_p2).squeeze(dim=3).squeeze(dim=2)
        f0_p3 = self.reduction(z0_p3).squeeze(dim=3).squeeze(dim=2)
        f1_p3 = self.reduction(z1_p3).squeeze(dim=3).squeeze(dim=2)
        f2_p3 = self.reduction(z2_p3).squeeze(dim=3).squeeze(dim=2)
       
        bin_out1, intermediate1=self.Binary_FC(fg_p1)
        bin_out2, intermediate2=self.Binary_FC(fg_p2)
        bin_out3, intermediate3=self.Binary_FC(fg_p3)


        intermediate1 = self.sig(intermediate1)
        intermediate2 = self.sig(intermediate2)
        intermediate3 = self.sig(intermediate3)
        fg_p1 = fg_p1*intermediate1
        fg_p2 = fg_p2*intermediate2
        fg_p3 = fg_p3*intermediate3

        fg_p1 = self.senet1(fg_p1)
        fg_p2 = self.senet1(fg_p2)
        fg_p3 = self.senet1(fg_p3)
        

        if not self.training:
            feat =  torch.cat([fg_recons,fg_p1, fg_p2, fg_p3, f0_p2, f1_p2, f0_p3, f1_p3, f2_p3], dim=1)
            fnorm = torch.norm(feat, p=2, dim=1, keepdim=True)
            ff = feat.div(fnorm.expand_as(feat))
            return ff

        l_p1 = self.fc_id_2048_0(fg_p1)
        l_p2 = self.fc_id_2048_1(fg_p2)
        l_p3 = self.fc_id_2048_2(fg_p3)

        l0_p2 = self.fc_id_256_1_0(f0_p2)
        l1_p2 = self.fc_id_256_1_1(f1_p2)
        l0_p3 = self.fc_id_256_2_0(f0_p3)
        l1_p3 = self.fc_id_256_2_1(f1_p3)
        l2_p3 = self.fc_id_256_2_2(f2_p3)

        

        predict = torch.cat([fg_p1, fg_p2, fg_p3, f0_p2, f1_p2, f0_p3, f1_p3, f2_p3], dim=1)
         
        
        output1=(predict, fg_p1, fg_p2, fg_p3, l_p1, l_p2, l_p3, l0_p2, l1_p2, l0_p3, l1_p3, l2_p3)
        
        output2 = (fg_p1, fg_p2, fg_p3,f0_p2,f1_p2,f0_p3,f1_p3,f2_p3)

        return output1, output2, recons,bin_out1,bin_out2, bin_out3 

def pcb_p6(num_classes, loss='softmax', pretrained=True, **kwargs):
   
    model = PCBT(
        num_classes=num_classes,
        loss=loss,
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        last_stride=1,
        parts=6,
        reduced_dim=256,
        nonlinear='relu',
        **kwargs
    )
    if pretrained:
        init_pretrained_weights(model, model_urls['resnet50'])
    return model

def multigrain(num_classes, loss='softmax', pretrained=True, **kwargs):
   
    model = MGN( num_classes=num_classes,**kwargs)
    
    return model


def pcb_p4(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = PCB(
        num_classes=num_classes,
        loss=loss,
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        last_stride=1,
        parts=4,
        reduced_dim=256,
        nonlinear='relu',
        **kwargs
    )
    if pretrained:
        init_pretrained_weights(model, model_urls['resnet50'])
    return model

