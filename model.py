from torchvision import models
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import Swinv2Backbone
from transformers import logging
from torch.utils.checkpoint import checkpoint
logging.set_verbosity_error()

# ==========================================
# Weight Initialization
# ==========================================
def weights_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight, a=0.25, mode='fan_in', nonlinearity='leaky_relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        torch.nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

# ==========================================
# 1. The Clean TUNA Parallel Adapter
# ==========================================
class CleanTunaAdapter(nn.Module):
    def __init__(self, dim, hidden_dim=64, conv_size=3):
        super().__init__()
        self.down_proj = nn.Linear(dim, hidden_dim)
        
        self.norm = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.dw_conv = nn.Conv2d(
            hidden_dim, hidden_dim, 
            kernel_size=conv_size, 
            padding=conv_size // 2, 
            groups=hidden_dim, 
            bias=False
        )
        self.inner_proj = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1, bias=False)
        
        self.up_proj = nn.Linear(hidden_dim, dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(p=0.1)
        
        # Zero-init so network starts mathematically identical to the frozen backbone
        nn.init.constant_(self.up_proj.weight, 0)
        nn.init.constant_(self.up_proj.bias, 0)

    def inner_conv_block(self, x, hw_shape):
        identity = x
        B, L, C = x.shape
        H, W = hw_shape
        
        x = self.norm(x)
        x_spatial = x.transpose(1, 2).reshape(B, C, H, W)
        x_spatial = self.dw_conv(x_spatial)
        x_spatial = self.inner_proj(x_spatial)
        x = x_spatial.flatten(2).transpose(1, 2)
        
        return x + identity

    def forward(self, x, hw_shape):
        identity = x
        x = self.down_proj(x)
        x = self.inner_conv_block(x, hw_shape)
        x = self.up_proj(x)
        x = self.act(x)
        x = self.dropout(x)
        return x + identity
    


# ==========================================
# 2. The 3-Hook Tripwire Matrix (Residual Scaling)
# ==========================================
# Hook 1: Captures initial block input (X) before anything happens
def block_pre_hook(module, args):
    module._phase_1_identity = args[0]
    module._input_dimensions = args[1]

# Hook 2: Intercepts the tensor entering the MLP (Z_raw = X + Attn_Output)
def intermediate_pre_hook(module, args):
    parent = module._parent_block
    z_raw = args[0]
    parent._z_raw = z_raw
    m1 = parent.tuna_1(parent._phase_1_identity, parent._input_dimensions)
    z_new = (parent.tuna_x_scale_1 * z_raw) + (parent.tuna_scale_1 * m1)
    parent._phase_2_identity = z_new
    return (z_new,)

# Hook 3: Intercepts the final output of the block
def block_post_hook(module, args, output):
    out_raw = output[0]
    z_raw = module._z_raw
    z_new = module._phase_2_identity
    w_true = out_raw - z_raw + z_new
    m2 = module.tuna_2(z_new, module._input_dimensions)
    w_new = (module.tuna_x_scale_2 * w_true) + (module.tuna_scale_2 * m2)
    return (w_new,) + output[1:]



# ==========================================
# 3. UperNet Decoding Components
# ==========================================
class ConvLayer(nn.Module):
    def __init__(self, inputfeatures, outputinter, kernel_size=7, stride=1, padding=3, dilation=1, output=64, layertype=1, droupout=False):
        super(ConvLayer, self).__init__()
        if not droupout:
            self.layer1 = nn.Sequential(
                nn.Conv2d(inputfeatures, outputinter, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation),
                nn.BatchNorm2d(outputinter),
                nn.PReLU(num_parameters=1, init=0.25))
            self.layer2 = nn.Sequential(
                nn.Conv2d(outputinter, outputinter, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation),
                nn.BatchNorm2d(outputinter),
                nn.PReLU(num_parameters=1, init=0.25))
            self.layer3 = nn.Sequential(
                nn.Conv2d(outputinter, output, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation),
                nn.BatchNorm2d(output),
                nn.PReLU(num_parameters=1, init=0.25))
        else:
            self.layer1 = nn.Sequential(
                nn.Conv2d(inputfeatures, outputinter, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation),
                nn.BatchNorm2d(outputinter),
                nn.Dropout(p=0.30),
                nn.PReLU(num_parameters=1, init=0.25))
            self.layer2 = nn.Sequential(
                nn.Conv2d(outputinter, outputinter, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation),
                nn.BatchNorm2d(outputinter),
                nn.Dropout(p=0.30),
                nn.PReLU(num_parameters=1, init=0.25))
            self.layer3 = nn.Sequential(
                nn.Conv2d(outputinter, output, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation),
                nn.BatchNorm2d(output),
                nn.Dropout(p=0.30),
                nn.PReLU(num_parameters=1, init=0.25))

        self.layer4 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.layer5 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=False)
        self.layertype = layertype

    def forward(self, x):
        out1 = self.layer1(x)
        if self.layertype == 1:
            out1 = self.layer3(out1)
            out1, inds = self.layer4(out1)
            return out1, inds
        elif self.layertype == 2:
            out1 = self.layer2(out1)
            out1 = self.layer3(out1)
            out1, inds = self.layer4(out1)
            return out1, inds
        elif self.layertype == 3:
            out1 = self.layer3(out1)
            return out1
        elif self.layertype == 4:
            out1 = self.layer3(out1)
            out1 = self.layer5(out1)
            return out1

class ClassifyBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ClassifyBlock, self).__init__()
        self.layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        return self.layer(x)

class PSPhead(nn.Module):
    def __init__(self, input_dim=1536, output_dims=384, final_output_dims=1536, pool_scales=[1,2,3,6]):
        super(PSPhead, self).__init__()
        self.ppm_modules = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(pool),
                nn.Conv2d(input_dim, output_dims, kernel_size=1),
                nn.BatchNorm2d(output_dims),
                nn.PReLU(num_parameters=1, init=0.25)
            )
            for pool in pool_scales
        ])

        self.bottleneck = nn.Sequential(
            nn.Conv2d(input_dim + output_dims*len(pool_scales), final_output_dims, kernel_size=3, padding=1),
            nn.BatchNorm2d(final_output_dims),
            nn.PReLU(num_parameters=1, init=0.25)
        )

    def forward(self, x):
        ppm_outs = [x]
        for ppm in self.ppm_modules:
            ppm_out = F.interpolate(ppm(x), size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)
            ppm_outs.append(ppm_out)
        
        ppm_outs = torch.cat(ppm_outs, dim=1)
        x = self.bottleneck(ppm_outs)
        return x

class FPN_fuse(nn.Module):
    def __init__(self, feature_channels=[192, 384, 768, 1536], fpn_out=512):
        super(FPN_fuse, self).__init__()
        
        # Lateral convolutions applied to ALL stages to ensure feature adaptation
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, fpn_out, kernel_size=1)
            for in_ch in feature_channels
        ])

        # Smoothing convolutions for aliasing reduction
        self.smooth_convs = nn.ModuleList([
            nn.Conv2d(fpn_out, fpn_out, kernel_size=3, padding=1)
            for _ in range(len(feature_channels))
        ])

        # Final Fusion bottlenecks concatenated FPN maps back to standard depth
        self.conv_fusion = nn.Sequential(
            nn.Conv2d(len(feature_channels) * fpn_out, fpn_out, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(fpn_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, features):
        # 1. Uniform lateral projection
        lats = [lateral(f) for lateral, f in zip(self.lateral_convs, features)]

        # 2. Strict Top-Down Summation
        for i in range(len(lats) - 1, 0, -1):
            up = F.interpolate(lats[i], size=lats[i-1].shape[2:], mode='bilinear', align_corners=True)
            lats[i-1] = lats[i-1] + up

        # 3. Anti-aliasing smoothing
        ps = [smooth(l) for smooth, l in zip(self.smooth_convs, lats)]

        # 4. Multi-Scale Aggregation (Targeting Stage 1 resolution)
        target_h, target_w = ps[0].shape[2:]
        fused_ps = [ps[0]]
        for i in range(1, len(ps)):
            fused_ps.append(F.interpolate(ps[i], size=(target_h, target_w), mode='bilinear', align_corners=True))

        x = torch.cat(fused_ps, dim=1)
        return self.conv_fusion(x)

# ==========================================
# 4. Integrated SwinUperNet Engine
# ==========================================
class SwinUperNet(nn.Module):
    def __init__(self, num_classes=104):
        super(SwinUperNet, self).__init__()

        model_name = "microsoft/swinv2-large-patch4-window12to24-192to384-22kto1k-ft"
        self.backbone = Swinv2Backbone.from_pretrained(
            model_name, 
            ignore_mismatched_sizes=True, 
            out_features=["stage1", "stage2", "stage3", "stage4"]
        )

        # Freeze Backbone completely
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Inject TUNA adapters perfectly aligned with Post-Norm architecture
        self._inject_tuna_adapters()

        self.feature_channels = [192, 384, 768, 1536]
        self.PPMhead = PSPhead(input_dim=1536, output_dims=384, final_output_dims=1536)
        self.FPN = FPN_fuse(self.feature_channels, fpn_out=512)
        
        self.head = ConvLayer(512, 128, kernel_size=3, stride=1, padding=1, output=64, layertype=3, droupout=True)
        self.ClassifyBlock = ClassifyBlock(64, num_classes)

        self.PPMhead.apply(weights_init)
        self.FPN.apply(weights_init)
        self.head.apply(weights_init)
        self.ClassifyBlock.apply(weights_init)

    def _inject_tuna_adapters(self):
        """Builds the 3-Hook Tripwire Matrix to inject and scale TUNA natively."""
        conv_sizes = [7, 5, 5, 3]
        hidden_dims = [64, 64, 96, 192]
        
        for stage_idx, stage in enumerate(self.backbone.encoder.layers):
            conv_size = conv_sizes[stage_idx]
            hidden_dim = hidden_dims[stage_idx]
            
            for block in stage.blocks:
                dim = block.layernorm_before.weight.shape[0]
                
                # Initialize TUNA Modules
                block.tuna_1 = CleanTunaAdapter(dim, hidden_dim, conv_size)
                block.tuna_2 = CleanTunaAdapter(dim, hidden_dim, conv_size)
                
                # Initialize Scaling Vectors (X_scale starts at 1.0, TUNA starts at 1e-6)
                block.tuna_scale_1 = nn.Parameter(torch.ones(dim) * 1e-6)
                block.tuna_scale_2 = nn.Parameter(torch.ones(dim) * 1e-6)
                block.tuna_x_scale_1 = nn.Parameter(torch.ones(dim))
                block.tuna_x_scale_2 = nn.Parameter(torch.ones(dim))
                
                # Link the intermediate module back to the block to access TUNA parameters
                block.intermediate.__dict__['_parent_block'] = block
                
                # The 3-Hook Interception Wiring
                block.register_forward_pre_hook(block_pre_hook)                    # Hook 1: Pre-Block
                block.intermediate.register_forward_pre_hook(intermediate_pre_hook) # Hook 2: Pre-MLP
                block.register_forward_hook(block_post_hook)                       # Hook 3: Post-Block
                
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Injection Complete. Total Trainable Parameters (TUNA + Head): {trainable:,}")

    def forward(self, x):
        input_size = (x.size()[2], x.size()[3])

        # Feature extraction via Hugging Face wrapper
        backbone_output = self.backbone(pixel_values=x)
        features = list(backbone_output.feature_maps)

        if self.training:
            def run_ppm(z):
                return self.PPMhead(z)
            
            def run_fpn(f1, f2, f3, f4):
                return self.FPN([f1, f2, f3, f4])
            
            def run_head(z):
                return self.head(z)
            
            features[-1] = checkpoint(run_ppm, features[-1], use_reentrant=False)
            x = checkpoint(run_fpn, features[0], features[1], features[2], features[3], use_reentrant=False)
            x = checkpoint(run_head, x, use_reentrant=False)

        else:
            features[-1] = self.PPMhead(features[-1])
            x = self.FPN(features)
            x = self.head(x)
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=False)
        x = self.ClassifyBlock(x)

        return x