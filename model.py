from torchvision import models
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import Swinv2Backbone
from transformers import logging
logging.set_verbosity_error()

# ==========================================
# Weight Initialization
# ==========================================
def weights_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight, a=0.25, mode='fan_in', nonlinearity='leaky_relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        torch.nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.GroupNorm) or isinstance(m, nn.BatchNorm2d):
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
def block_pre_hook(module, args):
    module._phase_1_identity = args[0]
    module._input_dimensions = args[1]

def intermediate_pre_hook(module, args):
    parent = module._parent_block
    z_raw = args[0]
    parent._z_raw = z_raw
    m1 = parent.tuna_1(parent._phase_1_identity, parent._input_dimensions)
    z_new = (parent.tuna_x_scale_1 * z_raw) + (parent.tuna_scale_1 * m1)
    parent._phase_2_identity = z_new
    return (z_new,)

def block_post_hook(module, args, output):
    out_raw = output[0]
    z_raw = module._z_raw
    z_new = module._phase_2_identity
    w_true = out_raw - z_raw + z_new
    m2 = module.tuna_2(z_new, module._input_dimensions)
    w_new = (module.tuna_x_scale_2 * w_true) + (module.tuna_scale_2 * m2)
    return (w_new,) + output[1:]

# ==========================================
# 3. MMSegmentation-Equivalent FPN Head
# ==========================================
class ConvNormRelu(nn.Module):
    """Mimics MMSegmentation's ConvModule to wrap Conv2d + Norm + ReLU"""
    def __init__(self, in_c, out_c, k, p=0):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=k, padding=p, bias=False)
        self.norm = nn.GroupNorm(32, out_c) # Change to nn.BatchNorm2d(out_c) at your own risk
        self.act = nn.ReLU(inplace=True)
        
    def forward(self, x):
        return self.act(self.norm(self.conv(x)))

class FPNHead(nn.Module):
    """Exact replica of the FPNHead used in the Swin-TUNA codebase."""
    def __init__(self, in_channels=[192, 384, 768, 1536], channels=512, num_classes=104, dropout_ratio=0.1):
        super().__init__()
        
        # 1x1 Convs to unify channel dimensions
        self.lateral_convs = nn.ModuleList([
            ConvNormRelu(in_c, channels, k=1, p=0) for in_c in in_channels
        ])
        
        # 3x3 Convs to smooth features after top-down addition
        self.fpn_convs = nn.ModuleList([
            ConvNormRelu(channels, channels, k=3, p=1) for _ in in_channels
        ])
        
        # Final Bottleneck mapping concatenated features to head channels
        self.fpn_bottleneck = ConvNormRelu(len(in_channels) * channels, channels, k=3, p=1)
        
        self.dropout = nn.Dropout2d(p=dropout_ratio)
        self.conv_seg = nn.Conv2d(channels, num_classes, kernel_size=1)

    def forward(self, features):
        # 1. Lateral Projections
        laterals = [lateral_conv(features[i]) for i, lateral_conv in enumerate(self.lateral_convs)]
        
        # 2. Top-Down Fusion
        for i in range(len(laterals) - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=prev_shape, mode='bilinear', align_corners=False
            )
            
        # 3. Smooth Features
        fpn_outs = [self.fpn_convs[i](laterals[i]) for i in range(len(laterals))]
        
        # 4. Multi-Scale Aggregation (Upsample to highest resolution feature)
        fused_shape = fpn_outs[0].shape[2:]
        fused_outs = [fpn_outs[0]]
        for i in range(1, len(fpn_outs)):
            fused_outs.append(F.interpolate(fpn_outs[i], size=fused_shape, mode='bilinear', align_corners=False))
            
        # 5. Concatenate, Bottleneck, and Classify
        x = torch.cat(fused_outs, dim=1)
        x = self.fpn_bottleneck(x)
        x = self.dropout(x)
        x = self.conv_seg(x)
        
        return x

# ==========================================
# 4. Integrated Swin-TUNA FPN Engine
# ==========================================
class SwinTunaFPN(nn.Module):
    def __init__(self, num_classes=104):
        super().__init__()

        model_name = "microsoft/swinv2-large-patch4-window12to24-192to384-22kto1k-ft"
        self.backbone = Swinv2Backbone.from_pretrained(
            model_name, 
            ignore_mismatched_sizes=True, 
            out_features=["stage1", "stage2", "stage3", "stage4"]
        )

        for param in self.backbone.parameters():
            param.requires_grad = False

        self._inject_tuna_adapters()

        # The Exact TUNA FPN Configuration
        self.decode_head = FPNHead(
            in_channels=[192, 384, 768, 1536], 
            channels=512, 
            num_classes=num_classes, 
            dropout_ratio=0.1
        )

        self.decode_head.apply(weights_init)

    def _inject_tuna_adapters(self):
        conv_sizes = [7, 5, 5, 3]
        hidden_dims = [64, 64, 96, 192]
        
        for stage_idx, stage in enumerate(self.backbone.encoder.layers):
            conv_size = conv_sizes[stage_idx]
            hidden_dim = hidden_dims[stage_idx]
            
            for block in stage.blocks:
                dim = block.layernorm_before.weight.shape[0]
                
                block.tuna_1 = CleanTunaAdapter(dim, hidden_dim, conv_size)
                block.tuna_2 = CleanTunaAdapter(dim, hidden_dim, conv_size)
                
                block.tuna_scale_1 = nn.Parameter(torch.ones(dim) * 1e-6)
                block.tuna_scale_2 = nn.Parameter(torch.ones(dim) * 1e-6)
                block.tuna_x_scale_1 = nn.Parameter(torch.ones(dim))
                block.tuna_x_scale_2 = nn.Parameter(torch.ones(dim))
                
                block.intermediate.__dict__['_parent_block'] = block
                
                block.register_forward_pre_hook(block_pre_hook)
                block.intermediate.register_forward_pre_hook(intermediate_pre_hook)
                block.register_forward_hook(block_post_hook)
                
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Injection Complete. Total Trainable Parameters (TUNA + Head): {trainable:,}")

    def forward(self, x):
        input_size = (x.size()[2], x.size()[3])

        backbone_output = self.backbone(pixel_values=x)
        features = list(backbone_output.feature_maps)

        logits = self.decode_head(features)
        
        # Final upsample to match input image resolution (640x640)
        output = F.interpolate(logits, size=input_size, mode='bilinear', align_corners=False)

        return output