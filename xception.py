import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import sys
from torchsummary import summary
import numpy as np

# os.environ['CUDA_VISIBLE_DEVICES'] =  '2'
def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = torch.cuda.FloatTensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = torch.autograd.Variable(torch.cuda.FloatTensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = torch.autograd.grad(
        outputs=d_interpolates,  # fack samples
        inputs=interpolates,   # real samples
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

class SeparableConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=1,stride=1,padding=0,dilation=1,bias=False):
        super(SeparableConv2d,self).__init__()

        self.conv1 = nn.Conv2d(in_channels,in_channels,kernel_size,stride,padding,dilation,groups=in_channels,bias=bias)
        self.pointwise = nn.Conv2d(in_channels,out_channels,1,1,0,1,1,bias=bias)
    
    def forward(self,x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class Block(nn.Module):
    def __init__(self,in_filters,out_filters,reps,strides=1,start_with_relu=True,grow_first=True):
        super(Block, self).__init__()

        if out_filters != in_filters or strides!=1:
            self.skip = nn.Conv2d(in_filters,out_filters,1,stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip=None
        
        self.relu = nn.ReLU(inplace=True)
        rep=[]

        filters=in_filters
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters,out_filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters

        for i in range(reps-1):
            rep.append(self.relu)
            rep.append(SeparableConv2d(filters,filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(filters))
        
        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters,out_filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(out_filters))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

        if strides != 1:
            rep.append(nn.MaxPool2d(3,strides,1))
        self.rep = nn.Sequential(*rep)

    def forward(self,inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x+=skip
        return x


class Xception_(nn.Module):
    """
    Xception optimized for the ImageNet dataset, as specified in
    https://arxiv.org/pdf/1610.02357.pdf
    """
    def __init__(self, maptype, num_classes=1000):
        """ Constructor
        Args:
            num_classes: number of classes
        """
        super(Xception_, self).__init__()

        self.maptype = maptype
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(3, 32, 3,2, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32,64,3,bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        #do relu here

        self.block1=Block(64,128,2,2,start_with_relu=False,grow_first=True)
        self.block2=Block(128,256,2,2,start_with_relu=True,grow_first=True)
        self.block3=Block(256,728,2,2,start_with_relu=True,grow_first=True)

        self.block4=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block5=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block6=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block7=Block(728,728,3,1,start_with_relu=True,grow_first=True)

        self.block8=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block9=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block10=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block11=Block(728,728,3,1,start_with_relu=True,grow_first=True)

        self.block12=Block(728,1024,2,2,start_with_relu=True,grow_first=False)

        self.conv3 = SeparableConv2d(1024,1536,3,1,1)
        self.bn3 = nn.BatchNorm2d(1536)

        #do relu here
        self.conv4 = SeparableConv2d(1536,2048,3,1,1)
        self.bn4 = nn.BatchNorm2d(2048)

        self.fc = nn.Linear(2048, num_classes)

        if maptype == 'none':
          self.map = [1, None]
        elif maptype == 'reg':
          self.map = RegressionMap(728)
        elif maptype == 'tmp':
          self.map = TemplateMap(728, templates)
        elif maptype == 'pca_tmp':
          self.map = PCATemplateMap(728)
        else:
          print('Unknown map type: `{0}`'.format(maptype))
          sys.exit()

        #------- init weights --------
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        #-----------------------------





    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)

        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class RegressionMap(nn.Module):
  def __init__(self, c_in):
    super(RegressionMap, self).__init__()
    self.c = SeparableConv2d(c_in, 1, 3, stride=1, padding=1, bias=False)
    self.s = nn.Sigmoid()

  def forward(self, x):
    mask = self.c(x)
    mask = self.s(mask)
    return mask, None


class Xception_Tail(nn.Module):
  """
  Xception optimized for the ImageNet dataset, as specified in
  https://arxiv.org/pdf/1610.02357.pdf
  """
  def __init__(self, isFC=False):
    super(Xception_Tail, self).__init__()

    self.block8=Block(728,728,3,1,start_with_relu=True,grow_first=True)
    self.block9=Block(728,728,3,1,start_with_relu=True,grow_first=True)
    self.block10=Block(728,728,3,1,start_with_relu=True,grow_first=True)
    self.block11=Block(728,728,3,1,start_with_relu=True,grow_first=True)
    self.block12=Block(728,1024,2,2,start_with_relu=True,grow_first=False)

    self.conv3 = SeparableConv2d(1024,1536,3,1,1)
    self.bn3 = nn.BatchNorm2d(1536)

    self.conv4 = SeparableConv2d(1536,2048,3,1,1)
    self.bn4 = nn.BatchNorm2d(2048)
    self.isFC = isFC
    if self.isFC:
      self.last_linear = nn.Linear(2048, 2)
    self.relu = nn.ReLU(inplace=True)

    for m in self.modules():
      if isinstance(m, nn.Conv2d):
          n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
          m.weight.data.normal_(0, math.sqrt(2. / n))
      elif isinstance(m, nn.BatchNorm2d):
          m.weight.data.fill_(1)
          m.bias.data.zero_()

  def forward(self, x):
    x = self.block8(x)
    x = self.block9(x)
    x = self.block10(x)
    x = self.block11(x)
    x = self.block12(x)
    x = self.conv3(x)
    x = self.bn3(x)
    x = self.relu(x)

    x = self.conv4(x)
    x = self.bn4(x)

    x = self.relu(x)
    x = F.adaptive_avg_pool2d(x, (1, 1))
    x = x.view(x.size(0), -1)
    if self.isFC:
      x = self.last_linear(x)
    return x


class Xception_Head(nn.Module):
  """
  Xception optimized for the ImageNet dataset, as specified in
  https://arxiv.org/pdf/1610.02357.pdf
  """
  def __init__(self, maptype = 'none'):
    super(Xception_Head, self).__init__()

    self.conv1 = nn.Conv2d(3, 32, 3,2, 0, bias=False)
    self.bn1 = nn.BatchNorm2d(32)
    self.relu = nn.ReLU(inplace=True)

    self.conv2 = nn.Conv2d(32,64,3,bias=False)
    self.bn2 = nn.BatchNorm2d(64)

    self.block1=Block(64,128,2,2,start_with_relu=False,grow_first=True)
    self.block2=Block(128,256,2,2,start_with_relu=True,grow_first=True)
    self.block3=Block(256,728,2,2,start_with_relu=True,grow_first=True)
    self.block4=Block(728,728,3,1,start_with_relu=True,grow_first=True)
    self.block5=Block(728,728,3,1,start_with_relu=True,grow_first=True)
    self.block6=Block(728,728,3,1,start_with_relu=True,grow_first=True)
    self.block7=Block(728,728,3,1,start_with_relu=True,grow_first=True)
    
    self.maptype = maptype
    if maptype == 'none':
      self.map = [1, None]
    elif maptype == 'reg':
      self.map = RegressionMap(728)
    else:
      print('Unknown map type: `{0}`'.format(maptype))
      sys.exit()
    
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
          n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
          m.weight.data.normal_(0, math.sqrt(2. / n))
      elif isinstance(m, nn.BatchNorm2d):
          m.weight.data.fill_(1)
          m.bias.data.zero_()

  def forward(self, input):
    x = self.conv1(input)
    x = self.bn1(x)
    x = self.relu(x)

    x = self.conv2(x)
    x = self.bn2(x)
    x = self.relu(x)

    x = self.block1(x)
    x = self.block2(x)
    x = self.block3(x)
    x = self.block4(x)
    x = self.block5(x)
    x = self.block6(x)
    f = self.block7(x)
    if self.maptype != 'none':
      mask, _ = self.map(f)
      x = f * mask
    else:
      mask = None
    return x, mask, f

class Tail_Model:
  def __init__(self, load_pretrain=True, isFC=False):
    model = Xception_Tail(isFC=isFC)
    if load_pretrain:
      state_dict = torch.load('/media/user/deepfake/FFD/FFD_Gan/xception-c0a72b38.pth.tar')
      
      # for name, weights in state_dict.items():
      #   if 'pointwise' in name:
      #     state_dict[name] = weights.unsqueeze(-1).unsqueeze(-1)
      # del state_dict['fc.weight']
      # del state_dict['fc.bias']
      model.load_state_dict(state_dict, False)
      print("----loading success-----")
    self.model = model

  def save(self, epoch, optim, model_dir, pre, use_gpu=False):
    if use_gpu:
      state = {'net': self.model.module.state_dict(), 'optim': optim.state_dict()}
    else:
      state = {'net': self.model.state_dict(), 'optim': optim.state_dict()}
    torch.save(state, ('{0}'+str(pre)+'_{1:06d}.tar').format(model_dir, epoch))
    print('Saved model `{0}`'.format(epoch))

  def load(self, epoch, model_dir, pre):
    filename = ('{0}'+str(pre)+'_{1:06d}.tar').format(model_dir, epoch)
    print('Loading model from {0}'.format(filename))
    if os.path.exists(filename):
      state = torch.load(filename)
      self.model.load_state_dict(state['net'])
    else:
      print('Failed to load model from {0}'.format(filename))

class Head_Model:
  def __init__(self, load_pretrain=True, maptype="none"):
    model = Xception_Head(maptype=maptype)
    if load_pretrain:
      state_dict = torch.load('/media/user/deepfake/FFD/FFD_Gan/xception-c0a72b38.pth.tar')
      # model.apply(init_weights)
      # for name, weights in state_dict.items():
      #   if 'pointwise' in name:
      #     state_dict[name] = weights.unsqueeze(-1).unsqueeze(-1)
      del state_dict['fc.weight']
      del state_dict['fc.bias']
      model.load_state_dict(state_dict, False)
      print("----loading success-----")
    self.model = model

  def save(self, epoch, optim, model_dir, pre, use_gpu=False):
    if use_gpu:
      state = {'net': self.model.module.state_dict(), 'optim': optim.state_dict()}
    else:
      state = {'net': self.model.state_dict(), 'optim': optim.state_dict()}
    torch.save(state, ('{0}'+str(pre)+'_{1:06d}.tar').format(model_dir, epoch))
    print('Saved model `{0}`'.format(epoch))

  def load(self, epoch, model_dir, pre):
    filename = ('{0}'+str(pre)+'_{1:06d}.tar').format(model_dir, epoch)
    print('Loading model from {0}'.format(filename))
    if os.path.exists(filename):
      state = torch.load(filename)
      self.model.load_state_dict(state['net'])
    else:
      print('Failed to load model from {0}'.format(filename))

class Xception(nn.Module):
  """
  Xception optimized for the ImageNet dataset, as specified in
  https://arxiv.org/pdf/1610.02357.pdf
  """
  def __init__(self, maptype, templates, isLogits=True, num_classes=2):
    super(Xception, self).__init__()
    self.num_classes = num_classes

    self.maptype = maptype
    self.isLogits = isLogits
    self.conv1 = nn.Conv2d(3, 32, 3,2, 0, bias=False)
    self.bn1 = nn.BatchNorm2d(32)
    self.relu = nn.ReLU(inplace=True)

    self.conv2 = nn.Conv2d(32,64,3,bias=False)
    self.bn2 = nn.BatchNorm2d(64)

    self.block1=Block(64,128,2,2,start_with_relu=False,grow_first=True)
    self.block2=Block(128,256,2,2,start_with_relu=True,grow_first=True)
    self.block3=Block(256,728,2,2,start_with_relu=True,grow_first=True)
    self.block4=Block(728,728,3,1,start_with_relu=True,grow_first=True)
    self.block5=Block(728,728,3,1,start_with_relu=True,grow_first=True)
    self.block6=Block(728,728,3,1,start_with_relu=True,grow_first=True)
    self.block7=Block(728,728,3,1,start_with_relu=True,grow_first=True)
    self.block8=Block(728,728,3,1,start_with_relu=True,grow_first=True)
    self.block9=Block(728,728,3,1,start_with_relu=True,grow_first=True)
    self.block10=Block(728,728,3,1,start_with_relu=True,grow_first=True)
    self.block11=Block(728,728,3,1,start_with_relu=True,grow_first=True)
    self.block12=Block(728,1024,2,2,start_with_relu=True,grow_first=False)

    self.conv3 = SeparableConv2d(1024,1536,3,1,1)
    self.bn3 = nn.BatchNorm2d(1536)

    self.conv4 = SeparableConv2d(1536,2048,3,1,1)
    self.bn4 = nn.BatchNorm2d(2048)
    
    self.last_linear = nn.Linear(2048, 1024)
    
    if maptype == 'none':
      self.map = [1, None]
    elif maptype == 'reg':
      self.map = RegressionMap(728)
    elif maptype == 'tmp':
      self.map = TemplateMap(728, templates)
    elif maptype == 'pca_tmp':
      self.map = PCATemplateMap(728)
    else:
      print('Unknown map type: `{0}`'.format(maptype))
      sys.exit()

    for m in self.modules():
      if isinstance(m, nn.Conv2d):
          n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
          m.weight.data.normal_(0, math.sqrt(2. / n))
      elif isinstance(m, nn.BatchNorm2d):
          m.weight.data.fill_(1)
          m.bias.data.zero_()

  def features(self, input):
    x = self.conv1(input)
    x = self.bn1(x)
    x = self.relu(x)

    x = self.conv2(x)
    x = self.bn2(x)
    x = self.relu(x)

    x = self.block1(x)
    x = self.block2(x)
    x = self.block3(x)
    x = self.block4(x)
    x = self.block5(x)
    x = self.block6(x)
    x = self.block7(x)
    if self.maptype != 'none':
      mask, vec = self.map(x)
      x = x * mask
    else:
      mask, vec = None, None
    x = self.block8(x)
    x = self.block9(x)
    x = self.block10(x)
    x = self.block11(x)
    x = self.block12(x)
    x = self.conv3(x)
    x = self.bn3(x)
    x = self.relu(x)

    x = self.conv4(x)
    x = self.bn4(x)
    return x, mask, vec

  def logits(self, features):
    x = self.relu(features)
    x = F.adaptive_avg_pool2d(x, (1, 1))
    x = x.view(x.size(0), -1)
    if self.isLogits:
      return x
    x = self.last_linear(x)
    return x

  def forward(self, input):
    x, mask, vec = self.features(input)
    x = self.logits(x)
    return x, mask, vec

class Discriminator(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        # Filters [256, 512, 1024]
        # Input_dim = channels (Cx64x64)
        # Output_dim = 1
        self.main_module = nn.Sequential(
            # Omitting batch normalization in critic because our new penalized training objective (WGAN with gradient penalty) is no longer valid
            # in this setting, since we penalize the norm of the critic's gradient with respect to each input independently and not the enitre batch.
            # There is not good & fast implementation of layer normalization --> using per instance normalization nn.InstanceNorm2d()
            # Image (Cx32x32)
            # State (2048x19x19)
            nn.Conv2d(in_channels=1456, out_channels=1456, kernel_size=3, stride=1, padding=1),
            # nn.InstanceNorm2d(1456, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # State (2048x9x9)
            nn.Conv2d(in_channels=1456, out_channels=2048, kernel_size=4, stride=2, padding=1),
            # nn.InstanceNorm2d(2048, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # State (4096x4x4)
            nn.Conv2d(in_channels=2048, out_channels=2048, kernel_size=4, stride=2, padding=1),
            # nn.InstanceNorm2d(2048, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # (4096x1x1)
            nn.Conv2d(in_channels=2048, out_channels=2048, kernel_size=4, stride=1, padding=0),
            # nn.InstanceNorm2d(2048, affine=True),
            nn.LeakyReLU(0.2, inplace=True))

        #(1, 4, 4)
        self.fc = nn.Sequential(
            # The output of D is no longer a probability, we do not apply sigmoid at the output of D.
            nn.Linear(2048, 1))
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)  
    def forward(self, x):
        batch_size = x.size()[0]
        x = self.main_module(x)
        return self.fc(x.view(batch_size, -1))

class D_Model:
  def __init__(self):
    model = Discriminator(728*2)
    self.model = model

  def save(self, epoch, optim, model_dir, pre, use_gpu=False):
    if use_gpu:
      state = {'net': self.model.module.state_dict(), 'optim': optim.state_dict()}
    else:
      state = {'net': self.model.state_dict(), 'optim': optim.state_dict()}
    torch.save(state, ('{0}'+str(pre)+'_{1:06d}.tar').format(model_dir, epoch))
    print('Saved model `{0}`'.format(epoch))

  def load(self, epoch, model_dir, pre):
    filename = ('{0}'+str(pre)+'_{1:06d}.tar').format(model_dir, epoch)
    print('Loading model from {0}'.format(filename))
    if os.path.exists(filename):
      state = torch.load(filename)
      self.model.load_state_dict(state['net'])
    else:
      print('Failed to load model from {0}'.format(filename))

class Model:
  def __init__(self, maptype='none', templates=None, num_classes=2, isLogits=False, load_pretrain=True):
    model = Xception(maptype, templates, isLogits=isLogits,  num_classes=num_classes)
    if load_pretrain:
      state_dict = torch.load('/media/user/deepfake/FFD/FFD_Gan/xception-c0a72b38.pth.tar')
      # for name, weights in state_dict.items():
      #   if 'pointwise' in name:
      #     state_dict[name] = weights.unsqueeze(-1).unsqueeze(-1)
      del state_dict['fc.weight']
      del state_dict['fc.bias']
      model.load_state_dict(state_dict, False)
      print("---load success--")
    self.model = model

  def save(self, epoch, optim, model_dir):
    state = {'net': self.model.state_dict(), 'optim': optim.state_dict()}
    torch.save(state, '{0}/{1:06d}.tar'.format(model_dir, epoch))
    print('Saved model `{0}`'.format(epoch))

  def load(self, epoch, model_dir):
    filename = '{0}{1:06d}.tar'.format(model_dir, epoch)
    print('Loading model from {0}'.format(filename))
    if os.path.exists(filename):
      state = torch.load(filename)
      self.model.load_state_dict(state['net'])
    else:
      print('Failed to load model from {0}'.format(filename))

if __name__ == "__main__":
  # tail1 = Tail_Model(isFC=False)
  # head1 = Head_Model("none")

  # tail2 = Tail_Model(isFC=False)
  # head2 = Head_Model("none")
  # full1 = Model(isLogits=True)
  # full2 = Model(isLogits=True)
  # img = torch.ones((1,3,299,299))

  # output1, _, _ = full1.model(img)
  # output2, _, _ = full2.model(img)
  # output1 = tail1.model( head1.model(img)[0] )
  # output2 = tail2.model( head2.model(img)[0] )

  # print(output1 == output2)
  model = Discriminator(728*2).cuda()
  summary(model, (1456, 19, 19))