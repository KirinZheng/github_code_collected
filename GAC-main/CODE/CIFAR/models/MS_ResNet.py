from models.layers import *
from models.GAU import TA,SCA

from spikingjelly.clock_driven.neuron import MultiStepParametricLIFNode, MultiStepLIFNode, LIFNode, MultiStepIFNode

import math

from models.embeddings import get_3d_sincos_pos_embed, get_sinusoid_spatial_temporal_encoding

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock_MS(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, use_spikingjelly_lif=False, first_block=False, use_spikingjelly_if=False):
        super(BasicBlock_MS, self).__init__()
        # changed on 2025-05-25
        self.use_spikingjelly_lif = use_spikingjelly_lif
        self.use_spikingjelly_if = use_spikingjelly_if
        self.first_block = first_block
        if norm_layer is None:
            norm_layer = tdBatchNorm
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

        self.conv1_s = tdLayer(self.conv1, self.bn1)
        self.conv2_s = tdLayer(self.conv2, self.bn2)
        if self.use_spikingjelly_lif:
            if self.first_block:
                #self.spike1 = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')        # changed on 2025-05-25
                self.spike2 = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')        # changed on 2025-05-25
            else:
                self.spike1 = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')        # changed on 2025-05-25
                self.spike2 = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')        # changed on 2025-05-25
        elif self.use_spikingjelly_if:
            if self.first_block:
                #self.spike1 = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')        # changed on 2025-05-25
                self.spike2 = MultiStepIFNode(tau=2.0, detach_reset=True, backend='cupy')        # changed on 2025-05-25
            else:
                self.spike1 = MultiStepIFNode(tau=2.0, detach_reset=True, backend='cupy')        # changed on 2025-05-25
                self.spike2 = MultiStepIFNode(tau=2.0, detach_reset=True, backend='cupy')        # changed on 2025-05-25
        else:
            self.spike = LIFSpike()

    def forward(self, x, first_identity=None):
        # changed on 2025-05-25, 用于区分recurrent_coding之后不需要进行spike操作了（提前）
        if not self.first_block:
            identity = x
            if self.use_spikingjelly_lif:
                out = x.permute(1, 0, 2, 3, 4).contiguous()    # T B C H W
                out = self.spike1(out)
                out = out.permute(1, 0, 2, 3, 4).contiguous()    # B T C H W
            else:
                out = self.spike(x)
        else:
            identity = first_identity
            out = x
        out = self.conv1_s(out)
        if self.use_spikingjelly_lif:
            out = out.permute(1, 0, 2, 3, 4).contiguous()    # T B C H W
            out = self.spike2(out)
            out = out.permute(1, 0, 2, 3, 4).contiguous()    # B T C H W
        else:
            out = self.spike(out)
        out = self.conv2_s(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        

        return out


class GAC(nn.Module):
    def __init__(self,T,out_channels):
        super().__init__()
        self.TA = TA(T = T)
        self.SCA = SCA(in_planes= out_channels,kerenel_size=4)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_seq, spikes):
        # x_seq B T inplanes H W
        # spikes B T inplanes H W

        TA = self.TA(x_seq)
        SCA = self.SCA(x_seq)
        out = self.sigmoid(TA * SCA)
        y_seq = out * spikes
        return y_seq


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10, using_GAC=True, time_step=4,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, use_spikingjelly_lif=False, recurrent_coding=False, recurrent_lif=None,
                 pe_type=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = tdBatchNorm
        self._norm_layer = norm_layer
        self.GAC = using_GAC
        self.inplanes = 64
        self.dilation = 1
        self.use_spikingjelly_lif = use_spikingjelly_lif
        self.recurrent_coding = recurrent_coding
        self.recurrent_lif = recurrent_lif
        self.pe_type = pe_type
    
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float32  # 默认为 float32，后续可能会使用混合精度

        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        if not self.recurrent_coding:
            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1,
                                bias=False)

            self.bn1 = norm_layer(self.inplanes)
            self.conv1_s = tdLayer(self.conv1, self.bn1)
        else:   # using recurrent coding
            self.re_conv = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1,
                                bias=False)
            self.re_bn = nn.BatchNorm2d(self.inplanes)
            self.re_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')  # changed on 2025-05-26

        ########################################################################################################################
        # changed on 2025-05-25
        assert not self.recurrent_coding or self.use_spikingjelly_lif, \
                            "If recurrent_coding is True, use_spikingjelly_lif must also be True"
        if self.pe_type is not None:
            if self.pe_type == "3d_pe_arch_1" or self.pe_type == "3d_pe_arch_4":
                pos_embed = get_3d_sincos_pos_embed(embed_dim=self.inplanes, spatial_size=32,
                                                    temporal_size=time_step, output_type="pt",
                                                    )  # T HW D
                T_pe, HW_pe, D_pe = pos_embed.shape
                # pos_embed = pos_embed.to(x.dtype)
                pos_embed = pos_embed.reshape(T_pe, int(math.sqrt(HW_pe)), int(math.sqrt(HW_pe)), D_pe).unsqueeze(dim=1).permute(0, 1, 4, 2, 3).contiguous()  # T 1 C H W
                self.learnable_pos_embed = nn.Parameter(pos_embed.to(self.device, self.dtype))
            
            elif self.pe_type == "3d_pe_arch_2" or self.pe_type == "3d_pe_arch_3":
                pos_embed = get_sinusoid_spatial_temporal_encoding(height=32, width=32,
                                                                   time_step=time_step, d_hid=3)
                
                self.learnable_pos_embed = nn.Parameter(pos_embed.to(self.device, self.dtype))  # T 1 C H W
            

        if recurrent_coding:
            if self.pe_type == "3d_pe_arch_1" or self.pe_type == "3d_pe_arch_2":
                self.proj_temporal_conv = nn.Conv2d(self.inplanes, 3, kernel_size=3, stride=1, padding=1, bias=False)
                self.proj_temporal_bn = nn.BatchNorm2d(3)
            elif self.pe_type == "3d_pe_arch_3" or self.pe_type == "3d_pe_arch_4" or self.pe_type == "3d_pe_arch_5":
                self.proj_temporal_conv = nn.Conv2d(self.inplanes, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
                self.proj_temporal_bn = nn.BatchNorm2d(self.inplanes)
            
            if recurrent_lif is not None:
                if recurrent_lif == 'lif':
                    self.proj_temporal_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
                elif recurrent_lif == 'plif':
                    self.proj_temporal_lif = MultiStepParametricLIFNode(init_tau=2.0,detach_reset=True, backend='cupy')

        ########################################################################################################################


        # changed on 2025-05-25
        if self.recurrent_coding:
            self.layer1 = self._make_layer(block, 128, layers[0], use_spikingjelly_lif=use_spikingjelly_lif, first_block=True)
        else:
            self.layer1 = self._make_layer(block, 128, layers[0], use_spikingjelly_lif=use_spikingjelly_lif)

        self.layer2 = self._make_layer(block, 256, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0], use_spikingjelly_lif=use_spikingjelly_lif)
        self.layer3 = self._make_layer(block, 512, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1], use_spikingjelly_lif=use_spikingjelly_lif)
        self.avgpool = tdLayer(nn.AdaptiveAvgPool2d((1, 1)))

        self.fc1 = nn.Linear(512 * block.expansion, num_classes)
        self.fc1_s = tdLayer(self.fc1)
        if self.use_spikingjelly_lif:
            self.GAC_spike = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')    # changed on 2025-05-25
            self.spike = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')        # changed on 2025-05-25
        else:
            self.spike = LIFSpike()

        self.T = time_step
        if using_GAC==True:
            self.encoding = GAC(T=self.T ,out_channels =64)
        



    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, use_spikingjelly_lif=False, first_block=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = tdLayer(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, use_spikingjelly_lif=use_spikingjelly_lif))
        self.inplanes = planes * block.expansion
        for idx in range(1, blocks):
            if first_block and idx == 0:
                layers.append(block(self.inplanes, planes, groups=self.groups,
                                    base_width=self.base_width, dilation=self.dilation,
                                    norm_layer=norm_layer, use_spikingjelly_lif=use_spikingjelly_lif, first_block=first_block))
            else:
                layers.append(block(self.inplanes, planes, groups=self.groups,
                                    base_width=self.base_width, dilation=self.dilation,
                                    norm_layer=norm_layer, use_spikingjelly_lif=use_spikingjelly_lif))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # B T C H W(注意这里的维度有点奇怪的)
        '''encoding'''
        # 这里是使用 recurrent_coding 同时 不使用GAC的情况 
        if self.recurrent_coding is True and self.GAC is False:
            B, T, C, H, W = x.shape
            x = x.permute(1, 0, 2, 3, 4).contiguous()  # T B C H W
            # changed on 2025-05-26
            if self.pe_type == "3d_pe_arch_2" or self.pe_type == "3d_pe_arch_3":
                x = x + self.learnable_pos_embed

            t_x = []

            if self.pe_type == "3d_pe_arch_1" or self.pe_type == "3d_pe_arch_2":
                non_spike_t_x = []
                for i in range(T):
                    if i == 0:
                        x_in = x[i] # B C H W
                    else:
                        x_in = x[i] + x_out
                        # x_in = x_out
                    x_out = self.re_conv(x_in)    # B C H W
                    x_out = self.re_bn(x_out)     # B C H W

                    # add 3d_pe_arch_1
                    if self.pe_type == "3d_pe_arch_1":
                        x_out = x_out + self.learnable_pos_embed[i] # B C H W
                    
                    non_spike_t_x.append(x_out)
                
                    x_out = self.re_lif(x_out.unsqueeze(0)).squeeze(0)  # B C H W
                
                    tmp = x_out
                    
                    x_out = self.proj_temporal_conv(x_out)    # B C H W
                    
                    x_out = self.proj_temporal_bn(x_out)     # B C H W
                    if self.recurrent_lif is not None:
                        x_out = self.proj_temporal_lif(x_out.unsqueeze(0)).squeeze(0) # B C H W
                
                    t_x.append(tmp)
            elif self.pe_type == "3d_pe_arch_3" or self.pe_type == "3d_pe_arch_4" or self.pe_type == "3d_pe_arch_5":
                x = self.re_conv(x.flatten(0, 1)) # TB C H W
                x = self.re_bn(x).reshape(T, B, -1, H, W) # T B C H W
                
                if self.pe_type == "3d_pe_arch_4":
                    x = x + self.learnable_pos_embed # T B C H W
                
                non_spike_t_x = x   # T B C H W

                for i in range(T):
                    if i == 0:
                        x_in = x[i] # B C H W
                    else:
                        x_in = x[i] + x_out
                        # x_in = x_out
                
                    x_out = self.re_lif(x_in.unsqueeze(0)).squeeze(0)  # B C H W
                
                    tmp = x_out
                    
                    x_out = self.proj_temporal_conv(x_out)    # B C H W
                    
                    x_out = self.proj_temporal_bn(x_out)     # B C H W
                    if self.recurrent_lif is not None:
                        x_out = self.proj_temporal_lif(x_out.unsqueeze(0)).squeeze(0) # B C H W
                
                    t_x.append(tmp)

            x = torch.stack(t_x, dim=0).permute(1, 0, 2, 3, 4).contiguous() # B T C H W
            if isinstance(non_spike_t_x, list):
                non_spike_x = torch.stack(non_spike_t_x, dim=0).permute(1, 0, 2, 3, 4).contiguous() # B T C H W
            else:
                non_spike_x = non_spike_t_x.permute(1, 0, 2, 3, 4).contiguous()  # B T C H W

            '''recurrent encoding'''
        # 其他除上述组合的唯一特殊情况
        else:
            if self.GAC==True:
                if self.recurrent_coding:
                    B, T, C, H, W = x.shape

                    x = x.permute(1, 0, 2, 3, 4).contiguous()  # T B C H W
                    # changed on 2025-05-26
                    if self.pe_type == "3d_pe_arch_2" or self.pe_type == "3d_pe_arch_3":
                        x = x + self.learnable_pos_embed

                    t_x = []
                    img_x = []

                    if self.pe_type == "3d_pe_arch_1" or self.pe_type == "3d_pe_arch_2":
                        for i in range(T):
                            if i == 0:
                                x_in = x[i] # B C H W
                            else:
                                x_in = x[i] + x_out
                                # x_in = x_out
                            x_out = self.re_conv(x_in)    # B C H W
                            x_out = self.re_bn(x_out)     # B C H W

                            img_x.append(x_out)

                            # add 3d_pe_arch_1
                            if self.pe_type == "3d_pe_arch_1":
                                x_out = x_out + self.learnable_pos_embed[i] # B C H W
                        
                            x_out = self.re_lif(x_out.unsqueeze(0)).squeeze(0)  # B C H W
                        
                            tmp = x_out
                            
                            x_out = self.proj_temporal_conv(x_out)    # B C H W
                            
                            x_out = self.proj_temporal_bn(x_out)     # B C H W
                            if self.recurrent_lif is not None:
                                x_out = self.proj_temporal_lif(x_out.unsqueeze(0)).squeeze(0) # B C H W
                        
                            t_x.append(tmp)
                            img = torch.stack(img_x, dim=0).permute(1, 0, 2, 3, 4).contiguous() # B T C H W

                    elif self.pe_type == "3d_pe_arch_3" or self.pe_type == "3d_pe_arch_4" or self.pe_type == "3d_pe_arch_5":
                        x = self.re_conv(x.flatten(0, 1)) # TB C H W
                        x = self.re_bn(x).reshape(T, B, -1, H, W) # T B C H W
                        img = x.permute(1, 0, 2, 3, 4).contiguous()  # B T C H W
                        
                        if self.pe_type == "3d_pe_arch_4":
                            x = x + self.learnable_pos_embed # T B C H W

                        for i in range(T):
                            if i == 0:
                                x_in = x[i] # B C H W
                            else:
                                x_in = x[i] + x_out
                                # x_in = x_out
                        
                            x_out = self.re_lif(x_in.unsqueeze(0)).squeeze(0)  # B C H W
                        
                            tmp = x_out
                            
                            x_out = self.proj_temporal_conv(x_out)    # B C H W
                            
                            x_out = self.proj_temporal_bn(x_out)      # B C H W
                            if self.recurrent_lif is not None:
                                x_out = self.proj_temporal_lif(x_out.unsqueeze(0)).squeeze(0) # B C H W
                        
                            t_x.append(tmp)

                    x = torch.stack(t_x, dim=0).permute(1, 0, 2, 3, 4).contiguous() # B T C H W
                    x = self.encoding(img,x)
                    '''recurrent encoding'''
                else:
                    x = self.conv1_s(x)
                    img = x
                    if self.use_spikingjelly_lif:
                        x = x.permute(1, 0, 2, 3, 4).contiguous()    # T B C H W
                        x = self.GAC_spike(x)
                        x = x.permute(1, 0, 2, 3, 4).contiguous()   # B T C H W
                    else:
                        x = self.spike(x)
                    x = self.encoding(img,x)
            else:
                x = self.conv1_s(x)
            '''encoding'''
        
        if self.recurrent_coding is True and self.GAC is False:
            for i, block in enumerate(self.layer1):
                if i == 0:
                    x = block(x, first_identity=non_spike_x)
                else:
                    x = block(x)
        else:
            x = self.layer1(x)

        x = self.layer2(x)
        x = self.layer3(x)
        # changed on 2025-05-25
        if self.use_spikingjelly_lif:
            x = x.permute(1, 0, 2, 3, 4).contiguous()    # T B C H W
            x = self.spike(x)
            x = x.permute(1, 0, 2, 3, 4).contiguous()    # B T C H W
        else:
            x = self.spike(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 2)
        x = self.fc1_s(x)
        return x

    def forward(self, x):
        x = add_dimention(x, self.T)
        return self._forward_impl(x)


def _resnet(arch, block, layers, **kwargs):
    model = ResNet(block, layers, **kwargs)
    print("Using GAC: {}".format(model.GAC))
    print("Using SpikingJelly LIF: {}".format(model.use_spikingjelly_lif))
    print("Using Recurrent LIF: {}".format(model.recurrent_lif))
    print("Using Recurrent Coding: {}".format(model.recurrent_coding))
    print("Using PE Type: {}".format(model.pe_type))
    return model
def msresnet18(pretrained=False, progress=True, **kwargs):
    return _resnet('resnet18', BasicBlock_MS, [3, 3, 2],
                   **kwargs)



if __name__ == '__main__':
    model = msresnet18(num_classes=10,using_GAC=False)
    model.T = 4
    x = torch.rand(2,3,32,32)
    y = model(x)
    print(y.shape)
    print("Parameter numbers: {}".format(
        sum(p.numel() for p in model.parameters())))
