"""
MSEG_e: 这个就是MSEG_e1_1文件
"""
from locale import DAY_3
import sys
sys.path.append("/code/EGVD/network/basic")
import torch
import torch.nn as nn
import torch.nn.init as init
from numpy import *
import torch.nn.functional as F
import commentjson as json
from ConvLSTM import ConvLSTM

def To3D(E1, group):
    [b, c, h, w] = E1.shape
    nf = int(c/group)

    E_list = []
    for i in range(0, group):
        tmp = E1[:, nf*i:nf*(i+1), :, :]
        tmp = tmp.view(b, nf, 1, h, w)
        E_list.append(tmp)
        
    E1_3d = torch.cat(E_list, 2)
    return E1_3d

def To2D(E1_3d):
    [b, c, g, h, w] = E1_3d.shape

    E_list = []
    for i in range(0, g):
        tmp = E1_3d[:, :, i, :, :]
        tmp = tmp.view(b, c, h, w)
        E_list.append(tmp)

    E1 = torch.cat(E_list, 1)
    return E1

def initialize_weights(self):
    for m in self.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n/2))
            if m.bias is not None:
                m.bias.data.zero_()

def conv(in_channels, out_channels, kernel_size, bias=False, stride = 1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride = stride)

def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

class CAB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act):
        super(CAB, self).__init__()
        modules_body = []
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
        modules_body.append(act)
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))

        self.CA = CALayer(n_feat, reduction, bias=bias)
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res = self.CA(res)
        res += x
        return res

"""
CBAM_BLOCK
"""

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = ConvLayer(2, 1, kernel_size, stride=1)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out) # broadcasting
        return x * scale

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()
    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out

class Par_CBAM_con(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(Par_CBAM_con,self).__init__()
        self.ChannelGate1 = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.SpatialGate1 = SpatialGate()
        self.ChannelGate2 = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.SpatialGate2 = SpatialGate()
        self.fus = ConvLayer(4*gate_channels, gate_channels, 3, 1)

    def forward(self, x):
        x1_u = self.ChannelGate1(x)
        x2_u = self.SpatialGate1(x1_u)
        x1_b = self.SpatialGate2(x)
        x2_b = self.ChannelGate2(x1_b)
        fus = self.fus(torch.cat((x1_u, x2_u, x1_b, x2_b), 1))
        return fus


class Par_CBAM_add(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(Par_CBAM_add,self).__init__()
        self.ChannelGate1 = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.SpatialGate1 = SpatialGate()
        self.ChannelGate2 = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.SpatialGate2 = SpatialGate()
        self.fus = ConvLayer(gate_channels, gate_channels, 3, 1)

    def forward(self, x):
        x1_u = self.ChannelGate1(x)
        x2_u = self.SpatialGate1(x1_u)
        x1_b = self.SpatialGate2(x)
        x2_b = self.ChannelGate2(x1_b)
        fus = self.fus(x1_u + x2_u + x1_b + x2_b)
        return fus


class Ser_CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(Ser_CBAM, self).__init__()
        self.ChannelGate1 = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.SpatialGate1 = SpatialGate()
        self.SpatialGate2 = SpatialGate()
        self.ChannelGate2 = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.fus = ConvLayer(gate_channels, gate_channels, 3, 1)
    def forward(self, x):
        x1 = self.ChannelGate1(x)
        x2 = self.SpatialGate1(x1)
        x3 = self.SpatialGate2(x2)
        x4 = self.ChannelGate2(x3)
        fus = self.fus(x4)
        return fus

class Sym_CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(Sym_CBAM, self).__init__()
        self.ChannelGate1 = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.SpatialGate1 = SpatialGate()
        self.SpatialGate2 = SpatialGate()
        self.ChannelGate2 = ChannelGate(gate_channels, reduction_ratio, pool_types)
    def forward(self, x):
        x1 = self.ChannelGate1(x)
        x2 = self.SpatialGate1(x1)
        x3 = self.SpatialGate2(x2)
        x4 = self.ChannelGate2(x3)
        return x4

class multi_Sym_CBAM(nn.Module):
    def __init__(self, gate_channels, n_blocks, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(multi_Sym_CBAM, self).__init__()
        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        modules_body = []
        for _ in range(n_blocks//2):
            modules_body.append(Sym_CBAM(gate_channels))
            modules_body.append(self.act)
            modules_body.append(Sym_CBAM(gate_channels))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        out = self.body(x)
        return out

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1, norm=None, bias=True, last_bias=0):
        super(ConvLayer, self).__init__()
        padding = kernel_size // 2
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=bias)

        if last_bias!=0:
            init.constant(self.conv2d.weight, 0)
            init.constant(self.conv2d.bias, last_bias)

    def forward(self, x):
        out = self.conv2d(x)

        return out

class ResidualBlock(nn.Module):
    
    def __init__(self, channels, groups=1, norm=None, bias=True):
        super(ResidualBlock, self).__init__()
        self.conv1  = ConvLayer(channels, channels, kernel_size=3, stride=1, groups=groups, bias=bias, norm=norm)
        self.conv2  = ConvLayer(channels, channels, kernel_size=3, stride=1, groups=groups, bias=bias, norm=norm)

        self.relu   = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        
    def forward(self, x):
        
        input = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)

        out = out + input

        return out

### Spatial Attention Layer
class Transit_mask(nn.Module):
    def __init__(self, in_c=64, out_c=64):
        super(Transit_mask,self).__init__()
        self.conv = nn.Sequential(
                nn.ReflectionPad2d((1,1,1,1)),
                nn.Conv2d(in_c,out_c,3,1,padding=0),
                nn.ReLU(inplace=True),
                nn.ReflectionPad2d((1,1,1,1)),
                nn.Conv2d(in_c,out_c,3,1,padding=0),
                nn.ReLU(inplace=True),
            )
        self.mask = nn.Sequential(
                #nn.ReflectionPad2d((1,1,1,1)),
                nn.Conv2d(in_c,out_c,1,1,padding=0),
                nn.Tanh(),
                nn.Conv2d(out_c,1,1,1,padding=0),
                nn.Sigmoid()
            )

    def forward(self,input):
        out = self.conv(input)
        mask = self.mask(out)
        out1 = out*mask
        # out2 = out*(1-mask)
        return out1

class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

class CALayer_weight(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, 4, 1, padding=0, bias=bias),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

class DownSample(nn.Module):
    def __init__(self, in_channels,s_factor):
        super(DownSample, self).__init__()
        self.down = nn.Sequential(nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False),
                                  nn.Conv2d(in_channels, in_channels+s_factor, 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x = self.down(x)
        return x

class UpSample(nn.Module):
    def __init__(self, in_channels, s_factor, scale_factor):
        super(UpSample, self).__init__()

        if scale_factor == 4:
            self.up = nn.Sequential(nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False),
                                    nn.Conv2d(in_channels+2*s_factor, in_channels, 1, stride=1, padding=0, bias=False))
        else:
            self.up = nn.Sequential(nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False),
                                    nn.Conv2d(in_channels+s_factor, in_channels, 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x = self.up(x)
        return x

class SkipUpSample(nn.Module):
    def __init__(self, in_channels,s_factor):
        super(SkipUpSample, self).__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                nn.Conv2d(in_channels+s_factor, in_channels, 1, stride=1, padding=0, bias=False))

    def forward(self, x, y):
        x = self.up(x)
        x = x + y
        return x

class Encoder(nn.Module):
    def __init__(self, n_feat, scale_unetfeats, kernel_size=3, reduction=4, bias=False, act=nn.PReLU()):
        super(Encoder, self).__init__()
        
        self.encoder_level1 = [CAB(n_feat,                     kernel_size, reduction, bias=bias, act=act) for _ in range(2)]
        self.encoder_level2 = [CAB(n_feat+scale_unetfeats,     kernel_size, reduction, bias=bias, act=act) for _ in range(2)]
        self.encoder_level3 = [CAB(n_feat+(scale_unetfeats*2), kernel_size, reduction, bias=bias, act=act) for _ in range(2)]

        self.encoder_level1 = nn.Sequential(*self.encoder_level1)
        self.encoder_level2 = nn.Sequential(*self.encoder_level2)
        self.encoder_level3 = nn.Sequential(*self.encoder_level3)

        self.down12  = DownSample(n_feat, scale_unetfeats)
        self.down23  = DownSample(n_feat+scale_unetfeats, scale_unetfeats)     

    def forward(self, x):

        enc1 = self.encoder_level1(x)
        x = self.down12(enc1)

        enc2 = self.encoder_level2(x)
        x = self.down23(enc2)

        enc3 = self.encoder_level3(x)
        
        return [enc1, enc2, enc3] #[64,128,128] [96, 64, 64] [128, 32, 32]

class Decoder(nn.Module):
    def __init__(self, n_feat, scale_unetfeats, kernel_size=3, reduction=4, bias=False, act=nn.PReLU()):
        super(Decoder, self).__init__()
        
        self.CSA1 = multi_Sym_CBAM(n_feat, n_blocks = 4)
        self.CSA2 = multi_Sym_CBAM((n_feat+(scale_unetfeats*1)), n_blocks = 4)
        self.CSA3 = multi_Sym_CBAM((n_feat+(scale_unetfeats*2)), n_blocks = 4)

        self.CMF1 = CMFB(n_feat)
        self.CMF2 = CMFB(n_feat+(scale_unetfeats*1))
        self.CMF3 = CMFB(n_feat+(scale_unetfeats*2))
        
        self.decoder_level1 = [CAB(n_feat,                     kernel_size, reduction, bias=bias, act=act) for _ in range(2)]
        self.decoder_level2 = [CAB(n_feat+scale_unetfeats,     kernel_size, reduction, bias=bias, act=act) for _ in range(2)]
        self.decoder_level3 = [CAB(n_feat+(scale_unetfeats*2), kernel_size, reduction, bias=bias, act=act) for _ in range(2)]
        
        self.decoder_level1 = nn.Sequential(*self.decoder_level1)
        self.decoder_level2 = nn.Sequential(*self.decoder_level2)
        self.decoder_level3 = nn.Sequential(*self.decoder_level3)

        self.skip_attn1 = CAB(n_feat,                 kernel_size, reduction, bias=bias, act=act)
        self.skip_attn2 = CAB(n_feat+scale_unetfeats, kernel_size, reduction, bias=bias, act=act)

        self.up21  = SkipUpSample(n_feat, scale_unetfeats)
        self.up32  = SkipUpSample(n_feat+scale_unetfeats, scale_unetfeats)
        
        self.sam3 = SAM(n_feat+(scale_unetfeats*2), 4, kernel_size=1, bias=bias)
        self.sam2 = SAM(n_feat+(scale_unetfeats), 2, kernel_size=1, bias=bias)

        self.tail = nn.Sequential(
                ConvLayer(n_feat, n_feat, kernel_size=3, stride=1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                ConvLayer(n_feat, 3, kernel_size=1, stride=1)
        )

    def forward(self, enc, enc_e, state, img):

        enc1, enc2, enc3 = enc
        enc1_e, enc2_e, enc3_e = enc_e

        ref_enc1_e = self.CSA1(enc1_e)
        ref_enc2_e = self.CSA2(enc2_e)
        ref_enc3_e = self.CSA3(enc3_e)

        fus3 = self.CMF3(enc3, ref_enc3_e, state)              
        dec3 = self.decoder_level3(fus3)

        ref_dec3, clean3 = self.sam3(dec3, img)

        fus2 = self.CMF2(enc2, ref_enc2_e)
        x = self.up32(ref_dec3, self.skip_attn2(fus2))
        dec2 = self.decoder_level2(x)

        ref_dec2, clean2 = self.sam2(dec2, img)

        fus1 = self.CMF1(enc1, ref_enc1_e)
        x = self.up21(ref_dec2, self.skip_attn1(fus1))
        dec1 = self.decoder_level1(x)

        clean = self.tail(dec1) + img

        return [clean, clean2, clean3]

class Mask_learning(nn.Module):
    def __init__(self,channels):
        super(Mask_learning,self).__init__()
        self.mask = nn.Sequential(
                nn.Conv2d(channels,channels,1,1,padding=0),
                nn.Tanh(),
                nn.Conv2d(channels,2,1,1,padding=0),
                nn.Sigmoid()
            )

    def forward(self,feature):
        return self.mask(feature)

class CSAB(nn.Module):
    def __init__(self, n_feat):
        super(CSAB, self).__init__()
        self.ca1 = CALayer(n_feat, reduction=4)
        self.sa1 = Transit_mask(in_c=n_feat, out_c=n_feat)
        self.sa2 = Transit_mask(in_c=n_feat, out_c=n_feat)
        self.ca2 = CALayer(n_feat, reduction=4)
        
        if False:
            self.fus = CALayer_weight(n_feat*4, reduction = n_feat)

            self.fus1 = ConvLayer(n_feat*4, n_feat*4, kernel_size=3, stride=1, groups=1, bias=True, norm=None)
            self.fus2 = ConvLayer(n_feat*4, n_feat*2, kernel_size=3, stride=1, groups=1, bias=True, norm=None)
            self.fus3 = ConvLayer(n_feat*2, n_feat*1, kernel_size=3, stride=1, groups=1, bias=True, norm=None)
            self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        
        if True:
            self.fus1 = ConvLayer(n_feat*4, n_feat*4, kernel_size=3, stride=1, groups=1, bias=True, norm=None)
            self.fus2 = ConvLayer(n_feat*4, n_feat, kernel_size=3, stride=1, groups=1, bias=True, norm=None)

            self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    # also directly map n_feat*4 into n_feat
    def forward(self, x):
        x1 = self.ca1(x)
        x2 = self.sa1(x1)
        du_x1 = self.sa2(x)
        du_x2 = self.ca2(du_x1)

        fus = torch.cat((x1, x2, du_x1, du_x2), 1)
        if False:
            channel_patch_weight = self.fus(fus)
            fus1 = torch.cat((x1*channel_patch_weight[:,0,:,:], x2*channel_patch_weight[:,1,:,:], du_x1*channel_patch_weight[:,2,:,:], du_x2*channel_patch_weight[:,3,:,:]), 1)
        if True:
            out = self.relu(self.fus2(self.relu(self.fus1(fus))))
    
        return out

class AMF(nn.Module):
    def __init__(self, n_feat):
        super(AMF, self).__init__()
        
        self.mask_atten = Mask_learning(n_feat)
        self.conv = ConvLayer(n_feat, n_feat, kernel_size=3, stride=1, groups=1, bias=True, norm=None)
        self.conv_e =  ConvLayer(n_feat, n_feat, kernel_size=3, stride=1, groups=1, bias=True, norm=None)
        
        self.conv1 = ConvLayer(n_feat, n_feat, kernel_size=3, stride=1, groups=1, bias=True, norm=None)
        self.res1 =  ResidualBlock(n_feat, groups=1, bias=True, norm=None)
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
    
    def forward(self, rgb, event):
        
        x = rgb+event
        mask = self.mask_atten(x)

        enhance_rgb = self.conv(mask[:,0,:,:].unsqueeze(1)*rgb)
        enhance_event = self.conv_e(mask[:,1,:,:].unsqueeze(1)*event)
        
        fus = enhance_rgb + enhance_event

        out = self.res1(self.relu(self.conv1(fus)))
        
        return out
        
class CMF(nn.Module):
    def __init__(self, n_feat):
        super(CMF, self).__init__()

        self.mask_atten = Mask_learning(n_feat*2)
        self.conv = ConvLayer(n_feat, n_feat, kernel_size=3, stride=1, groups=1, bias=True, norm=None)
        self.conv_e =  ConvLayer(n_feat, n_feat, kernel_size=3, stride=1, groups=1, bias=True, norm=None)
        
        self.conv1 = ConvLayer(2*n_feat, n_feat, kernel_size=3, stride=1, groups=1, bias=True, norm=None)
        self.res1 =  ResidualBlock(n_feat, groups=1, bias=True, norm=None)
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    
    def forward(self, rgb, event):
        
        x = torch.cat((rgb, event), 1)
        mask = self.mask_atten(x)
        
        enhance_rgb = self.conv(mask[:,0,:,:].unsqueeze(1)*rgb)
        enhance_event = self.conv_e(mask[:,1,:,:].unsqueeze(1)*event)

        fus = torch.cat((enhance_rgb, enhance_event),1)
        
        out = self.res1(self.relu(self.conv1(fus)))

        return out

class CMFB(nn.Module):
    def __init__(self, n_feat):
        super(CMFB, self).__init__()

        self.conv1 = ConvLayer(2*n_feat, n_feat, kernel_size=3, stride=1, groups=1, bias=True, norm=None)
        self.res1 =  ResidualBlock(n_feat, groups=1, bias=True, norm=None)
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    
    def forward(self, rgb, event, state=None):
        if state is not None:
            x = rgb + state
        else:
            x = rgb
        
        fus = torch.cat((x, -event), 1)
        out = self.res1(self.relu(self.conv1(fus)))

        return out                

class OutPut(nn.Module):
    def __init__(self, n_feat, scale_unetfeats):
        super(OutPut, self).__init__()

        self.up4 = UpSample(n_feat, scale_unetfeats, 4)
        self.up2 = UpSample(n_feat, scale_unetfeats, 2)

        self.output1 = ConvLayer(n_feat, n_feat, kernel_size=3, stride=1)
        self.output2 = ConvLayer(n_feat, 3, kernel_size=1, stride=1)

        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, out):
        out3, out2, out1 = out
        
        clean3 = self.output2(self.relu(self.output1(self.relu(self.up4(out3)))))
        clean2 = self.output2(self.relu(self.output1(self.relu(self.up2(out2)))))
        clean = self.output2(self.relu(self.output1(out1)))

        return [clean, clean2, clean3]

class Coor_M(nn.Module):
    def __init__(self, n_feat):
        super(Coor_M, self).__init__()
        
        self.conv_k7 = nn.Sequential(
            ConvLayer(n_feat, n_feat, kernel_size=7, stride=1),
            ConvLayer(n_feat, n_feat, kernel_size=1, stride=1))

        self.conv_k5 = nn.Sequential(
            ConvLayer(n_feat, n_feat, kernel_size=5, stride=1),
            ConvLayer(n_feat, n_feat, kernel_size=1, stride=1))

        self.conv_k3 = nn.Sequential(
            ConvLayer(n_feat, n_feat, kernel_size=3, stride=1),
            ConvLayer(n_feat, n_feat, kernel_size=1, stride=1))

        self.fus = ConvLayer(3*n_feat, n_feat, kernel_size=3, stride=1)
    
    def forward(self, x):
        x1 = self.conv_k7(x)
        x2 = self.conv_k5(x)
        x3 = self.conv_k3(x)

        x_fus = self.fus(torch.cat((x1, x2, x3),1))
        corr_prob = torch.sigmoid(x_fus)
        return corr_prob

##########################################################################
## Supervised Attention Module
class SAM(nn.Module):
    def __init__(self, n_feat, scale_factor, kernel_size, bias):
        super(SAM, self).__init__()
        
        self.up = nn.Sequential(nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False),
                                nn.Conv2d(n_feat, n_feat, 3, stride=1, padding=1, bias=False),
                                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                nn.Conv2d(n_feat, 3, 1, stride=1, padding=0, bias=False))
        
        self.down = nn.Sequential(nn.Upsample(scale_factor=(1/scale_factor), mode='bilinear', align_corners=False),
                                nn.Conv2d(n_feat, n_feat, 1, stride=1, padding=0, bias=False))

        self.conv1 = conv(n_feat, n_feat, kernel_size, bias=bias)
        self.conv3 = conv(3, n_feat, kernel_size, bias=bias)

    def forward(self, x, x_img):
        x1 = self.conv1(x)
        img = self.up(x) + x_img
        x2 = torch.sigmoid(self.down(self.conv3(img)))
        x1 = x1*x2
        x1 = x1+x
        return x1, img

class RMFD(nn.Module):
    def __init__(self, opts, num_bins):
        super(RMFD, self).__init__()

        nf = opts["nf"] 
        scale_unetfeats=opts["scale_unetfeats"]
        use_bias = True
        opts["norm"] = "None"

        self.conv1 = ConvLayer(9, nf*3, kernel_size=3, stride=1, groups=3, bias=use_bias, norm=opts["norm"])
        self.res1 = ResidualBlock(nf*3, groups=3, bias=use_bias, norm=opts["norm"])

        self.conv1_e = ConvLayer(num_bins*2, nf*2, kernel_size=3, stride=1, groups=2, bias=use_bias, norm=opts["norm"])
        self.res1_e = ResidualBlock(nf*2, groups=2, bias=use_bias, norm=opts["norm"])
        
        self.corr = Coor_M(nf*2)
        self.corr_conv = ConvLayer(nf*2, nf*2, kernel_size=1, stride=1)

        self.conv2 = nn.Conv3d(nf, nf, kernel_size=(3, 3, 3), stride=(1,1,1), padding=(0,1,1), bias=use_bias)
        self.res2 = ResidualBlock(nf, groups=1, bias=use_bias, norm=opts["norm"])
        
        self.conv2_e = nn.Conv3d(nf, nf, kernel_size=(2, 3, 3), stride=(1,1,1), padding=(0,1,1), bias=use_bias)
        self.res2_e = ResidualBlock(nf, groups=1, bias=use_bias, norm=opts["norm"])

        self.encoder = Encoder(nf, scale_unetfeats)
        self.encoder_e = Encoder(nf, scale_unetfeats)

        self.convlstm = ConvLSTM(input_size=(nf + scale_unetfeats*2), hidden_size = (nf + scale_unetfeats*2), kernel_size=3)

        self.decoder = Decoder(nf, scale_unetfeats)
        
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
    
        initialize_weights(self)

    def forward(self, image, prev_state, event):

        X1 = self.res1(self.relu(self.conv1(image)))
        X1_neigh = torch.cat((X1[:,:32,:,:], X1[:,64:96,:,:]),1) #now nf =32
        X1_tar = X1[:,32:64,:,:]

        X1_e = self.res1_e(self.relu(self.conv1_e(event)))

        coor_E = self.corr(X1_e)
        ref_X_neigh = self.corr_conv(coor_E * X1_neigh)
        ref_X1 = torch.cat((ref_X_neigh[:,:32,:,:], X1_tar, ref_X_neigh[:,32:64,:,:]), 1)

        X1_3d = To3D(ref_X1, 3)
        X2_3d = self.conv2(X1_3d)
        X2 = To2D(X2_3d) #B, 64, 128, 128
        X2 = self.res2(self.relu(X2))

        X1_e_3d = To3D(X1_e, 2)
        X2_e_3d = self.conv2_e(X1_e_3d)
        X2_e = To2D(X2_e_3d) #B, 64, 128, 128
        X2_e = self.res2_e(self.relu(X2_e))

        feat_enc = self.encoder(X2)
        feat_enc_e = self.encoder_e(X2_e)
        
        state = self.convlstm(feat_enc[-1], prev_state)
        
        out = self.decoder(feat_enc, feat_enc_e, state[0], image[:,3:6,:,:])

        return out, state