# a module to extract foreground feature and color from the input burst
# first we extract the foreground feature from the input burst
# then we do the alignment based on the first frame
# finally we extract the color from the aligned burst

import torch
import torch.nn as nn
from torchvision.ops import DeformConv2d

class RGCAB(nn.Module):
    def __init__(self, num_features, num_rcab, reduction):
        super(RGCAB, self).__init__()
        self.module = [RGCA(num_features, reduction) for _ in range(num_rcab)]
        self.module.append(nn.Conv2d(num_features, num_features, kernel_size=3, padding=1, bias=False))
        self.module = nn.Sequential(*self.module)

    def forward(self, x):
        return x + self.module(x)

class RGCA(nn.Module):
    def __init__(self, n_feat, reduction=8, bias=False, act=nn.LeakyReLU(negative_slope=0.2,inplace=True), groups =1):

        super(RGCA, self).__init__()

        self.n_feat = n_feat
        self.groups = groups
        self.reduction = reduction

        modules_body = [nn.Conv2d(n_feat, n_feat, 3,1,1 , bias=bias, groups=groups), act, nn.Conv2d(n_feat, n_feat, 3,1,1 , bias=bias, groups=groups)]
        self.body   = nn.Sequential(*modules_body)

        self.gcnet = nn.Sequential(GCA(n_feat, n_feat))
        self.conv1x1 = nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias)

    def forward(self, x):
        res = self.body(x)
        res = self.gcnet(res)
        res = self.conv1x1(res)
        res += x
        return res

######################### Global Context Attention ##########################################

class GCA(nn.Module):
    def __init__(self, inplanes, planes, act=nn.LeakyReLU(negative_slope=0.2,inplace=True), bias=False):
        super(GCA, self).__init__()

        self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1, bias=bias)
        self.softmax = nn.Softmax(dim=2)

        self.channel_add_conv = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=1, bias=bias),
            act,
            nn.Conv2d(planes, inplanes, kernel_size=1, bias=bias)
        )

    def spatial_pool(self, x):
        batch, channel, height, width = x.size()
        input_x = x
        # [N, C, H * W]
        input_x = input_x.view(batch, channel, height * width)
        # [N, 1, C, H * W]
        input_x = input_x.unsqueeze(1)
        # [N, 1, H, W]
        context_mask = self.conv_mask(x)
        # [N, 1, H * W]
        context_mask = context_mask.view(batch, 1, height * width)
        # [N, 1, H * W]
        context_mask = self.softmax(context_mask)
        # [N, 1, H * W, 1]
        context_mask = context_mask.unsqueeze(3)
        # [N, 1, C, 1]
        context = torch.matmul(input_x, context_mask)
        # [N, C, 1, 1]
        context = context.view(batch, channel, 1, 1)

        return context

    def forward(self, x):
        # [N, C, 1, 1]
        context = self.spatial_pool(x)

        # [N, C, 1, 1]
        channel_add_term = self.channel_add_conv(context)
        x = x + channel_add_term

        return x

class UpSample(nn.Module):

    def __init__(self, in_channels, chan_factor, bias=False):
        super(UpSample, self).__init__()

        self.up = nn.Sequential(nn.Conv2d(in_channels, int(in_channels/chan_factor), 1, stride=1, padding=0, bias=bias),
                                 nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))

    def forward(self, x):
        x = self.up(x)
        return x

class DownSample(nn.Module):
    def __init__(self, in_channels, chan_factor, bias=False):
        super(DownSample, self).__init__()

        self.down = nn.Sequential(nn.AvgPool2d(2, ceil_mode=True, count_include_pad=False),
                                nn.Conv2d(in_channels, int(in_channels*chan_factor), 1, stride=1, bias=bias))

    def forward(self, x):
        x = self.down(x)
        return x
    
class MSF(nn.Module):
    def __init__(self, in_channels=64, reduction=8, bias=False):
        super(MSF, self).__init__()
        
        self.feat_ext1 = nn.Sequential(*[RGCAB(in_channels, 2, reduction) for _ in range(2)])
        
        self.down1 = DownSample(in_channels, chan_factor=1.5)
        self.feat_ext2 = nn.Sequential(*[RGCAB(int(in_channels*1.5), 2, reduction) for _ in range(2)])
        
        self.down2 = DownSample(int(in_channels*1.5), chan_factor=1.5)
        self.feat_ext3 = nn.Sequential(*[RGCAB(int(in_channels*1.5*1.5), 2, reduction) for _ in range(1)])
               
        self.up2 = UpSample(int(in_channels*1.5*1.5), chan_factor=1.5)
        self.feat_ext5 = nn.Sequential(*[RGCAB(int(in_channels*1.5), 2, reduction) for _ in range(2)])
        
        self.up1 = UpSample(int(in_channels*1.5), chan_factor=1.5)
        self.feat_ext6 = nn.Sequential(*[RGCAB(in_channels, 2, reduction) for _ in range(2)])
        
    def forward(self, x):
        
        x = self.feat_ext1(x)
        
        enc_1 = self.down1(x)
        enc_1 = self.feat_ext2(enc_1)
        
        enc_2 = self.down2(enc_1)
        enc_2 = self.feat_ext3(enc_2)
        
        dec_2 = self.up2(enc_2)
        dec_2 = self.feat_ext5(dec_2 + enc_1)
        
        dec_1 = self.up1(dec_2)
        dec_2 = self.feat_ext6(dec_1 + x)
        
        return dec_2

class FusionBlock(nn.Module):

    '''
        # Compuate the attention map, highlight distinctions while keep similarities
        
        Input: Aligned frames, [B, T, C, H, W]
        Output: Fused frame, [B, C, H, W]
    '''

    def __init__(self, num_feat=64, num_frame=8, center_frame_idx=0) -> None:
        super(FusionBlock, self).__init__()

        self.center_frame_idx = center_frame_idx
        self.temporal_attn1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.temporal_attn2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.feat_fusion = nn.Conv2d(num_frame * num_feat, num_feat, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, aligned_feat):
        b, t, c, h, w = aligned_feat.size()

        # attention map, highlight distinctions while keep similarities
        embedding_ref = self.temporal_attn1(aligned_feat[:, self.center_frame_idx, :, :, :].clone()) # query
        embedding = self.temporal_attn2(aligned_feat.view(-1, c, h, w)) # key
        embedding = embedding.view(b, t, -1, h, w)  # [b,t,c,h,w]
        embedding_ref_expanded = embedding_ref.unsqueeze(1).expand(b, t, c, h, w) 
        attention_map = torch.sum(embedding * embedding_ref_expanded, dim=2, keepdim=True) # [b,t,1,h,w] # query * key, and sum over channel
        attention_map = torch.softmax(attention_map, dim=1) # softmax over time
        aligned_feat = aligned_feat * attention_map
        fused_feat = self.lrelu(self.feat_fusion(aligned_feat.view(b, -1, h, w)))
        return fused_feat

class AlignBlock(nn.Module):

    '''
        # Align the burst frames based on the first frame
        
        Input: Burst frames, [B, T, C, H, W]
        Output: Aligned frames, [B, T, C, H, W]
    '''

    def __init__(self, num_features=64, bias=False) -> None:
        super(AlignBlock, self).__init__()

        # Offset Setting
        kernel_size = 3
        deform_groups = 8
        out_channels = deform_groups * 3 * kernel_size**2
        
        self.bottleneck = nn.Conv2d(num_features*2, num_features, kernel_size=3, padding=1, bias=bias)

        # Offset Conv
        self.offset_conv1 = nn.Conv2d(num_features, out_channels, 3, stride=1, padding=1, bias=bias)
        self.offset_conv2 = nn.Conv2d(num_features, out_channels, 3, stride=1, padding=1, bias=bias)
        self.offset_conv3 = nn.Conv2d(num_features, out_channels, 3, stride=1, padding=1, bias=bias)
        self.offset_conv4 = nn.Conv2d(num_features, out_channels, 3, stride=1, padding=1, bias=bias)
        
        # Deform Conv
        self.deform1 = DeformConv2d(num_features, num_features, 3, padding=1, groups=deform_groups)
        self.deform2 = DeformConv2d(num_features, num_features, 3, padding=1, groups=deform_groups)
        self.deform3 = DeformConv2d(num_features, num_features, 3, padding=1, groups=deform_groups)
        self.deform4 = DeformConv2d(num_features, num_features, 3, padding=1, groups=deform_groups)
    
    def offset_gen(self, x):
        
        o1, o2, mask = torch.chunk(x, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        return offset, mask

    def forward(self, burst_feat):
        
        # b, f, H, W = burst_feat.size()
        B, N, F, H, W = burst_feat.size()
        ref = burst_feat[:, 0:1, ...]
        ref = torch.repeat_interleave(ref, N, dim=1)
        feat = torch.cat([burst_feat, ref], dim=2)
        feat = feat.view(-1, feat.size(2), H, W)
        # print(feat.size())
        feat = self.bottleneck(feat)
        offset1, mask1 = self.offset_gen(self.offset_conv1(feat))
        feat = self.deform1(feat, offset1, mask1)
        
        offset2, mask2 = self.offset_gen(self.offset_conv2(feat))
        feat = self.deform2(feat, offset2, mask2)
        
        offset3, mask3 = self.offset_gen(self.offset_conv3(feat))
        feat = self.deform3(feat, offset3, mask3)
        
        offset4, mask4 = self.offset_gen(self.offset_conv4(feat))
        aligned_feat = self.deform4(feat, offset4, mask4)        

        aligned_feat = aligned_feat.view(B, N, -1, H, W)

        return aligned_feat

class BGFusionBlock(nn.Module):
    
        '''
            # Compuate the attention map, highlight distinctions while keep similarities
            
            Input: Aligned frames, [B, T, C, H, W]
            Output: Fused frame, [B, C, H, W]
        '''
    
        def __init__(self, num_feat=64, num_frame=8, temperature = 0.5) -> None:
            super(BGFusionBlock, self).__init__()
    
            self.temporal_attn1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.temporal_attn2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.feat_fusion = nn.Conv2d(num_frame * num_feat, num_feat, 1, 1)
            self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
            self.temperature = temperature
    
        def forward(self, aligned_feat):
            b, t, c, h, w = aligned_feat.size()

            # calculate the similarity between the reference frame and the other frames, if the similarity is high, the attention map will be high

            # Temporal attention layers processing
            embedding = self.temporal_attn2(aligned_feat.view(-1, c, h, w)).view(b, t, c, h, w) # query
            embedding_ref = self.temporal_attn1(aligned_feat.view(b * t, c, h, w)).view(b, t, c, h, w) # key

            # Expanding embedding_ref to match the dimensions of embedding
            embedding_ref_expanded = embedding_ref.unsqueeze(2).expand(b, t, t, c, h, w)

            # Adjusting the embedding dimensions for multiplication
            embedding_expanded = embedding.unsqueeze(1).expand(b, t, t, c, h, w)

            # Element-wise multiplication and sum over the channel dimension (dim=3)
            attention_scores = (embedding_expanded * embedding_ref_expanded).sum(dim=3)

            # Summing over the new time dimension (t_prime, dim=2) and keep dimensions for broadcasting
            # print(attention_scores.size(),attention_scores.type())
            attention_scores = attention_scores.sum(dim=2, keepdim=True) 
            
            # Softmax over the time dimension to get the attention maps
            if self.temperature > 0:
                attention_map = torch.softmax(attention_scores / self.temperature, dim=1)
            else:
                attention_map = torch.argmax(attention_scores, dim=1, keepdim=True)

            # Apply the attention map
            aligned_feat = aligned_feat * attention_map

            fused_feat = self.lrelu(self.feat_fusion(aligned_feat.view(b, -1, h, w)))
            return fused_feat

    
        
class FGRModel(nn.Module):

    def __init__(self, mode='color', num_features=64, burst_size=8, reduction=8, bias=False, center_frame_idx=0):
        super(FGRModel, self).__init__()        


        if mode=="color":            
            inp_chn = 6 # comp + trimap
            out_chn = 3 # RGB_foreground
     

        self.conv1 = nn.Sequential(nn.Conv2d(inp_chn, num_features, kernel_size=3, padding=1, bias=bias))

        ####### Edge Boosting Feature Alignment
        
        ## Feature Processing Module
        self.encoder = nn.Sequential(*[RGCAB(num_features, 3, reduction) for _ in range(3)])

        ## Burst Feature Alignment
        self.alignment = AlignBlock(num_features)        
        ## Refined Aligned Feature
        self.feat_ext1 = nn.Sequential(*[RGCAB(num_features, 3, reduction) for _ in range(3)])
        self.cor_conv1 = nn.Sequential(nn.Conv2d(num_features, num_features, kernel_size=3, padding=1, bias=bias))

        self.fusion = FusionBlock(num_feat=num_features, num_frame=burst_size, center_frame_idx=center_frame_idx)
        ## Multi-scale Feature Extraction
        self.UNet = nn.Sequential(MSF(num_features))
        self.out_conv = nn.Conv2d(num_features, out_chn, kernel_size=3, padding=1, bias=bias)
    
    def forward(self, burst, trimap):
        #burst.shape = [B, 8, 3, H, W]
        #trimap.shape = [B, 8, 3, H, W]
        B, N, C, H, W = burst.size()
        burst = torch.cat([burst, trimap], dim=2)
        # make burst.shape to [B * 8, 6, H, W]
        burst = burst.view(-1, 6, burst.size(3), burst.size(4))
        burst_feat = self.conv1(burst)          
        burst_feat = burst_feat.view(B, N, -1, H, W)
        ##################################################
        ####### Edge Boosting Feature Alignment #################
        ##################################################
        # base_frame_feat shape = [B, 1, 64, H, W]
        base_frame_feat = burst_feat[:, 0:1, ...]
        base_frame_feat = torch.repeat_interleave(base_frame_feat, N, dim=1)
        base_frame_feat = base_frame_feat.view(-1, base_frame_feat.size(2), H, W)
        burst_feat = burst_feat.view(-1, burst_feat.size(2), H, W)
        burst_feat = self.encoder(burst_feat)    
        burst_feat = burst_feat.view(B, N, -1, H, W)           
        ## Burst Feature Alignment
        burst_feat = self.alignment(burst_feat)
        burst_feat = burst_feat.view(-1, burst_feat.size(2), H, W)
        ## Refined Aligned Feature
        burst_feat = self.feat_ext1(burst_feat)        
        Residual = burst_feat - base_frame_feat
        Residual = self.cor_conv1(Residual)
        burst_feat += Residual                   

        ##################################################
        ####### Pseudo Burst Feature Fusion ####################
        ##################################################

        burst_feat = burst_feat.view(B, N, -1, H, W)
        fgr_fused_feat = self.fusion(burst_feat)
        fgr_feat = self.UNet(fgr_fused_feat)
        fgr_output = self.out_conv(fgr_feat)
        return fgr_output, fgr_fused_feat

# output the background color
class BGRModel(nn.Module):
    
        def __init__(self, mode='color', num_features=64, burst_size=8, reduction=8, bias=False, temperature=0.5):
            super(BGRModel, self).__init__()        
    
    
            if mode=="color":            
                inp_chn = 6 # comp + trimap
                out_chn = 3 # RGB_background
        
    
            self.conv1 = nn.Sequential(nn.Conv2d(inp_chn, num_features, kernel_size=3, padding=1, bias=bias))
    
            ####### Edge Boosting Feature Alignment
            
            ## Feature Processing Module
            self.encoder = nn.Sequential(*[RGCAB(num_features, 3, reduction) for _ in range(3)])
    
            ## Burst Feature Alignment
            self.alignment = AlignBlock(num_features)        
            ## Refined Aligned Feature
            self.feat_ext1 = nn.Sequential(*[RGCAB(num_features, 3, reduction) for _ in range(3)])
            self.cor_conv1 = nn.Sequential(nn.Conv2d(num_features, num_features, kernel_size=3, padding=1, bias=bias))
    
            self.fusion = BGFusionBlock(num_feat=num_features, num_frame=burst_size, temperature=temperature)
            ## Multi-scale Feature Extraction
            self.UNet = nn.Sequential(MSF(num_features))
            self.out_conv = nn.Conv2d(num_features, out_chn, kernel_size=3, padding=1, bias=bias)
        
        def forward(self, burst, trimap):
            #burst.shape = [B, 8, 3, H, W]
            #trimap.shape = [B, 8, 3, H, W]
            B, N, C, H, W = burst.size()
            burst = torch.cat([burst, trimap], dim=2)
            # make burst.shape to [B * 8, 6, H, W]
            burst = burst.view(-1, 6, burst.size(3), burst.size(4))
            burst_feat = self.conv1(burst)          
            burst_feat = self.encoder(burst_feat)
            burst_feat = burst_feat.view(B, N, -1, H, W)
            ##################################################
            ####### Background Fusion ########################
            ##################################################
            bgr_fusion_feat = self.fusion(burst_feat)
            burst_feat_unet = self.UNet(bgr_fusion_feat)
            bgr_out = self.out_conv(burst_feat_unet)
            return bgr_out, bgr_fusion_feat


class MattingModel(nn.Module):
    def __init__(self, mode='color', num_features=64, burst_size=8, reduction=8, temperature = 1.0, bias=False, center_frame_idx = 0):
        super(MattingModel, self).__init__()

        self.center_frame_idx = center_frame_idx

        if mode=="color":
            inp_chn = 6 # comp + trimap
            out_chn = 1 # alpha

        self.conv1 = nn.Sequential(nn.Conv2d(inp_chn, num_features, kernel_size=3, padding=1, bias=bias))
        self.encoder = nn.Sequential(*[RGCAB(num_features, 3, reduction) for _ in range(3)])
        self.fg_model = FGRModel(mode, num_features, burst_size, reduction, bias)
        self.bg_model = BGRModel(mode, num_features, burst_size, reduction, bias, temperature)
        self.fusion = nn.Conv2d(num_features * 3, num_features, kernel_size=3, padding=1, bias=bias)
        self.UNet = nn.Sequential(MSF(num_features))
        self.out_conv = nn.Conv2d(num_features, out_chn, kernel_size=3, padding=1, bias=bias)

    def forward(self, burst, trimap):
        #burst.shape = [B, 8, 3, H, W]
        #trimap.shape = [B, 8, 3, H, W]
        B, N, C, H, W = burst.size()
        input = torch.cat([burst, trimap], dim=2)
        # make burst.shape to [B * 8, 6, H, W]
        center_frame = input[:, self.center_frame_idx, ...]
        center_frame_feat = self.conv1(center_frame)
        center_frame_feat = self.encoder(center_frame_feat)
        ##################################################
        ####### Matting Fusion ########################
        ##################################################
        fg_out, fg_feat = self.fg_model(burst, trimap)
        bg_out, bg_feat = self.bg_model(burst, trimap)
        # print(fg_feat.size(), bg_feat.size(), center_frame_feat.size())
        fused_feat = torch.cat([fg_feat, bg_feat, center_frame_feat], dim=1)
        
        fused_feat = self.fusion(fused_feat)
        fused_feat = self.UNet(fused_feat)
        alpha_out = self.out_conv(fused_feat)
        fg_out = fg_out.unsqueeze(1)
        bg_out = bg_out.unsqueeze(1)
        alpha_out = alpha_out.unsqueeze(1)
        return  fg_out, bg_out, alpha_out




if __name__ == "__main__":
    from time import sleep
    input = torch.randn(2, 8, 3, 256, 256).cuda()
    trimap = torch.randn(2, 8, 3, 256, 256).cuda()
    model = FGRModel().cuda()
    output , _ = model(input, trimap)
    print(output.size())
    model = BGRModel().cuda()
    output , _ = model(input, trimap)
    print(output.size())
    # model = MattingModel().cuda()
    # output = model(input, trimap)
    # print(output[0].size())
    # print(output[1].size())
    # print(output[2].size())
    # sleep(1000)
