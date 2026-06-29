import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from src.models.mil_template import MIL
from src.models.layers import GlobalAttention, GlobalGatedAttention
from transformers import PretrainedConfig, PreTrainedModel, AutoConfig, AutoModel

MODEL_TYPE = 'spatial_abmil'


#pe only on visual features

# --------------------------------------------------------
# 1. Modular Positional Encoding Generator (PEG/PPEG/EPEG)
# --------------------------------------------------------
class PositionalEncoder(nn.Module):
    def __init__(self, dim, method='epeg', kernel_size=3):
        super().__init__()
        self.method = method.lower()

        # if self.method == 'ablate':
        #     pass
        
        # Standard PEG: Single Depth-wise Conv
        if self.method == 'peg':
            self.proj = nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, padding=kernel_size//2, groups=dim)
            
        # PPEG: Parallel branches (Multi-scale: 3x3, 5x5, 7x7)
        elif self.method == 'ppeg':
            self.branch1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
            self.branch2 = nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim)
            self.branch3 = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)

        # EPEG: Enhanced PEG (Cascaded / Deeper Context)
        elif self.method == 'epeg':
            self.proj = nn.Sequential(
                nn.Conv2d(dim, dim, 3, padding=1, groups=dim),
                nn.GELU(),
                nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
            )

        elif self.method == 'ablate':
            self.proj = nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, padding=kernel_size//2, groups=dim)

        else:
            raise ValueError(f"Unknown PEG method: {method}")

    def forward_conv(self, x_img):
        # Dispatch based on method type
        if self.method == 'ppeg':
            # Sum of multi-scale features
            return self.branch1(x_img) + self.branch2(x_img) + self.branch3(x_img)
        
        # Both PEG and EPEG use the 'self.proj' attribute
        return self.proj(x_img)

    def forward(self, x, coords=None, grid_strategy='sequence'):
        """
        Args:
            x: [B, N, C] Input features
            coords: [B, N, 2] (x,y) coordinates. Required if grid_strategy='coordinate'
            grid_strategy: 'sequence' (reshape sqrt(N)) or 'coordinate' (scatter to grid)
        """
        B, N, C = x.shape
        
        # --- Strategy 1: Sequence (Standard PEG assumption) ---
        if grid_strategy == 'sequence':
            # 1. Compute square side
            H = int(math.ceil(math.sqrt(N)))
            W = H
            num_pad = H * W - N
            
            # 2. Pad sequence to fit square
            if num_pad > 0:
                pad = torch.zeros(B, num_pad, C, device=x.device, dtype=x.dtype)
                x_reshaped = torch.cat([x, pad], dim=1)
            else:
                x_reshaped = x
                
            # 3. View as Image [B, C, H, W]
            x_img = x_reshaped.transpose(1, 2).view(B, C, H, W)
            
            # 4. Apply Convolution (PEG/PPEG/EPEG)
            feat = self.forward_conv(x_img)
            
            # 5. Flatten back to sequence
            feat = feat.flatten(2).transpose(1, 2)
            if num_pad > 0:
                feat = feat[:, :N, :]
                
            if self.method=='ablate':
                return x

            return x + feat # Residual connection

        # --- Strategy 2: Coordinate (True Spatial Layout) ---
        elif grid_strategy == 'coordinate':
            if coords is None:
                raise ValueError("grid_strategy='coordinate' requires 'coords' input.")
            
            # Note: This assumes coords are integers or can be mapped to a grid.
            grid_size = int(math.ceil(math.sqrt(N))) 
            
            # 1. Create Canvas
            canvas = torch.zeros(B, C, grid_size, grid_size, device=x.device, dtype=x.dtype)
            
            # 2. Map coordinates to grid indices
            # If coords are float [0, 1], scale them. If ints, use as is.
            if coords.max() <= 1.0:
                grid_coords = (coords * (grid_size - 1)).long()
            else:
                grid_coords = coords.long()
                
            # Safety clamp to grid bounds
            grid_coords[..., 0] = grid_coords[..., 0].clamp(0, grid_size-1)
            grid_coords[..., 1] = grid_coords[..., 1].clamp(0, grid_size-1)

            # 3. Scatter features to canvas (Batch loop for safety)
            for b in range(B):
                cx, cy = grid_coords[b, :, 0], grid_coords[b, :, 1]
                canvas[b, :, cy, cx] = x[b].t()

            # 4. Apply Convolution
            feat_map = self.forward_conv(canvas) 

            # 5. Gather features back from canvas
            feat_out = torch.zeros_like(x)
            for b in range(B):
                cx, cy = grid_coords[b, :, 0], grid_coords[b, :, 1]
                feat_out[b] = feat_map[b, :, cy, cx].t()

            return x + feat_out
       
# --------------------------------------------------------
# 2. Core Model: Spatial ABMIL
# --------------------------------------------------------

class Spatial_ABMIL(MIL):
    def __init__(
            self,
            in_dim: int = 1024,
            embed_dim: int = 512,
            spatial_input_dim: int = 70, 
            num_fc_layers: int = 2,
            dropout: float = 0.25,
            attn_dim: int = 384,
            gate: int = True,
            num_classes: int = 2,
            normalization_factor: float = 1000.0, 
            visual_dropout_rate: float = 0.0,
            
            # Positional Encoding Arguments
            peg_method: str = 'peg',        # Options: 'peg', 'ppeg', 'epeg'
            grid_strategy: str = 'sequence', # Options: 'sequence', 'coordinate'
            **kwargs
    ):
        super().__init__(in_dim=in_dim, embed_dim=embed_dim, num_classes=num_classes)
        
        self.normalization_factor = normalization_factor
        self.visual_dropout_rate = visual_dropout_rate
        self.grid_strategy = grid_strategy
        self.spatial_input_dim = spatial_input_dim

        # 1. Positional Encoding Generator (PEG/PPEG/EPEG)
        # Applied to Visual Features (in_dim=1024) BEFORE concatenation
        self.pos_encoder = PositionalEncoder(
            dim=in_dim, 
            method=peg_method,
            kernel_size=3
        )

        # 2. Spatial Projection Layer
        # Projects Spatial (70) -> Visual Size (1024)
        self.spatial_projector = nn.Sequential(
            nn.Linear(spatial_input_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(128, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(512, in_dim),
            nn.LayerNorm(in_dim),
            nn.ReLU(),
            nn.Dropout(dropout),   
        )
        
        # 3. Joint Feature Extractor
        # [Visual+Pos (1024)] + [Spatial (1024)] = 2048 -> embed_dim (512)
        self.joint_in_dim = in_dim + in_dim
        
        layers = []
        current_dim = self.joint_in_dim
        for i in range(num_fc_layers):
            layers.append(nn.Linear(current_dim, embed_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            current_dim = embed_dim
            
        self.feature_extractor = nn.Sequential(*layers)

        # 4. Global Attention
        attn_func = GlobalGatedAttention if gate else GlobalAttention
        self.global_attn = attn_func(
            L=embed_dim,
            D=attn_dim,
            dropout=dropout,
            num_classes=1
        )

        # 5. Classifier
        if num_classes > 0:
            self.classifier = nn.Linear(embed_dim, num_classes)

        self.initialize_weights()

    def forward_attention(self, h: torch.Tensor, precomputed_distances: torch.Tensor, coords=None, attn_mask=None, attn_only=True):
        """
        Args:
            h: [B, M, 1024] Visual features
            precomputed_distances: [B, M, 70] Spatial features.
            coords: [B, M, 2] Optional explicit coordinates (x, y). 
        """
        # Normalize Spatial Inputs
        spatial_input = precomputed_distances / self.normalization_factor
        
        # Coordinate Check
        if self.grid_strategy == 'coordinate':
            if coords is None:
                raise ValueError("When grid_strategy='coordinate', the 'coords' argument must be explicitly provided.")

        # Optional: Visual Dropout
        if self.training and self.visual_dropout_rate > 0:
            mask = (torch.rand(h.shape[0], h.shape[1], 1, device=h.device) > self.visual_dropout_rate).float()
            h = h * mask

        # --- STEP 1: Apply Position Encoding to Visual Features ---
        # [B, M, 1024] -> [B, M, 1024]
        # Uses self.pos_encoder which respects the 'peg_method' (peg/ppeg/epeg)
        h_pos = self.pos_encoder(h, coords=coords, grid_strategy=self.grid_strategy)

        # --- STEP 2: Project Spatial Features ---
        # [B, M, 70] -> [B, M, 1024]
        spatial_feat = self.spatial_projector(spatial_input)

        # --- STEP 3: Concatenation ---
        # [B, M, 1024 (Visual+Pos)] + [B, M, 1024 (Spatial)] -> [B, M, 2048]
        combined_input = torch.cat([h_pos, spatial_feat], dim=-1)

        # --- STEP 4: Project to Embedding Space ---
        # [B, M, 2048] -> [B, M, 512]
        h_joint = self.feature_extractor(combined_input)

        # --- STEP 5: Compute Attention Scores ---
        A_logits = self.global_attn(h_joint)
        A_logits = torch.transpose(A_logits, -2, -1) # [B, 1, M]
        
        if attn_mask is not None:
            A_logits = A_logits + (1 - attn_mask).unsqueeze(dim=1) * torch.finfo(A_logits.dtype).min

        if attn_only:
            return A_logits
            
        return h_joint, A_logits

    def forward_features(self, h: torch.Tensor, precomputed_distances: torch.Tensor, coords=None, attn_mask=None, return_attention: bool = True):
        h_joint, A_logits = self.forward_attention(
            h, 
            precomputed_distances=precomputed_distances,
            coords=coords,
            attn_mask=attn_mask, 
            attn_only=False
        )
        
        A = F.softmax(A_logits, dim=-1) 
        wsi_feat = torch.bmm(A, h_joint).squeeze(dim=1)  # [B, 512]
        
        log_dict = {'attention': A_logits if return_attention else None}
        return wsi_feat, log_dict

    def forward_head(self, h: torch.Tensor) -> torch.Tensor:
        return self.classifier(h)

    def forward(self, h: torch.Tensor,
                precomputed_distances: torch.Tensor, 
                coords: torch.Tensor = None, 
                loss_fn: nn.Module = None,
                label: torch.LongTensor = None,
                attn_mask=None,
                return_attention: bool = False,
                return_slide_feats: bool = False,
                **kwargs): 
        
        wsi_feats, log_dict = self.forward_features(
            h, 
            precomputed_distances=precomputed_distances,
            coords=coords,
            attn_mask=attn_mask, 
            return_attention=return_attention
        )
        
        logits = self.forward_head(wsi_feats)
        cls_loss = MIL.compute_loss(loss_fn, logits, label)
        
        results_dict = {'logits': logits, 'loss': cls_loss}
        log_dict['loss'] = cls_loss.item() if cls_loss is not None else -1
        
        if return_slide_feats:
            log_dict['slide_feats'] = wsi_feats
            
        return results_dict, log_dict

# --------------------------------------------------------
# 3. Configuration & AutoModel Setup
# --------------------------------------------------------

class SpatialABMILConfig(PretrainedConfig):
    model_type = MODEL_TYPE

    def __init__(self,
                 normalization_factor: float = 1000.0,
                 visual_dropout_rate: float = 0.0,
                 gate: bool = True,
                 embed_dim: int = 512,
                 attn_dim: int = 384,
                 num_fc_layers: int = 2,
                 dropout: float = 0.25,
                 in_dim: int = 1024,
                 num_classes: int = 2,
                 spatial_input_dim: int = 70,
                 # Arguments for Spatial Logic
                 peg_method: str = 'peg',        # Change this to 'peg' or 'ppeg' as needed
                 grid_strategy: str = 'sequence', # Change this to 'coordinate' as needed
                 **kwargs):
        super().__init__(**kwargs)
        self.normalization_factor = normalization_factor
        self.visual_dropout_rate = visual_dropout_rate
        self.gate = gate
        self.embed_dim = embed_dim
        self.attn_dim = attn_dim
        self.num_fc_layers = num_fc_layers
        self.dropout = dropout
        self.in_dim = in_dim
        self.num_classes = num_classes
        self.spatial_input_dim = spatial_input_dim
        self.peg_method = peg_method
        self.grid_strategy = grid_strategy
        self.auto_map = {
            "AutoConfig": "modeling_spatial_abmil.SpatialABMILConfig",
            "AutoModel": "modeling_spatial_abmil.SpatialABMILModel",
        }

class SpatialABMILModel(PreTrainedModel):
    config_class = SpatialABMILConfig

    def __init__(self, config: SpatialABMILConfig, **kwargs):
        super().__init__(config)
        self.model = Spatial_ABMIL(
            in_dim=config.in_dim,
            embed_dim=config.embed_dim,
            num_fc_layers=config.num_fc_layers,
            dropout=config.dropout,
            attn_dim=config.attn_dim,
            gate=config.gate,
            num_classes=config.num_classes,
            normalization_factor=config.normalization_factor,
            visual_dropout_rate=config.visual_dropout_rate,
            spatial_input_dim=config.spatial_input_dim,
            # Pass config values to model
            peg_method=config.peg_method,
            grid_strategy=config.grid_strategy
        )

    def forward(self, *args, **kwargs):
        return self.model.forward(*args, **kwargs)

    def forward_features(self, *args, **kwargs):
        return self.model.forward_features(*args, **kwargs)

    def forward_attention(self, *args, **kwargs):
        return self.model.forward_attention(*args, **kwargs)

    def forward_head(self, *args, **kwargs):
        return self.model.forward_head(*args, **kwargs)

# Register for AutoClasses
AutoConfig.register(SpatialABMILConfig.model_type, SpatialABMILConfig)
AutoModel.register(SpatialABMILConfig, SpatialABMILModel)