import torch
import torch.nn.functional as F

from torch import nn
from torch import Tensor
from PIL import Image
# from torchsummary import summary
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce

image = torch.rand(3, 224, 224)
image = image.unsqueeze(dim = 0)

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

# print(image.shape)

# patch_size = 16
# embedding_size = (image.shape[-1] // patch_size ) * (image.shape[-2] // patch_size)

# patch_embed = rearrange(image, 'b c (h s1) (w s2) -> b (h w) (s1 s2 c)', s1 = patch_size, s2 = patch_size)

# print(patch_embed.shape)

# PATCH EMBEDDING
class PatchEmbedding(nn.Module):
    
    def __init__(self, in_channels: int = 3, emb_size: int = 768, patch_size: int = 16, img_size: int = 224):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels = in_channels, out_channels = emb_size, kernel_size = patch_size, 
                      stride = patch_size, padding = 0),     # here outputshape for input shape [1, 3, 224, 224] -> [1, 768, 14, 14]
                                                             # image height / patch size -> 224/ 16 = 14  
            Rearrange('b e (h) (w) -> b (h w) e')            # here we get [1, 768, 14, 14] -> [1, 196, 768] --> 14*14 = 196 
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.positions = nn.Parameter(torch.randn((img_size // patch_size) ** 2 + 1, emb_size))

    def forward(self, x: Tensor) -> Tensor:
        x = x.to(device)
        
        b, _, _, _ = x.shape # get batch size
        
        x = self.projection(x)

        cls_tokens = repeat(self.cls_token, '() n e -> b n e',  b=b)

        x = torch.cat([cls_tokens, x], dim = 1)

        x += self.positions

        return x
    
# print(PatchEmbedding()(image).shape)


# ENCODER

## ATENTION
class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size: int = 768, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        # self.keys = nn.Linear(self.emb_size, self.emb_size)
        # self.queries = nn.Linear(self.emb_size, self.emb_size)
        # self.values = nn.Linear(self.emb_size, self.emb_size)

        # key, query and value also write in matrix form

        self.qkv = nn.Linear(emb_size, emb_size * 3)

        self.att_dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(self.emb_size, self.emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        # queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h = self.num_heads)
        # keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h = self.num_heads)
        # values = rearrange(self.values(x), "b n (h d) -> b h n d", h = self.num_heads)
        # here b -> batch size, h -> number of heads, n -> embeding, d -> emb / 8

        # above code snippet also write as below
        qkv = rearrange(self.qkv(x), "b n (h d qkv) -> (qkv) b h n d", h = self.num_heads, qkv = 3)

        queries, keys, values = qkv[0], qkv[1], qkv[2]
        # print(keys.shape) 

        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)

        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1/2)
        att = F.softmax(energy, dim = -1) / scaling
        att = self.att_dropout(att)

        out = torch.einsum('bhal, bhlv -> bhav', att, values)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.projection(out)
        return out
    

# patches_embedded = PatchEmbedding()(image)
# print(MultiHeadAttention()(patches_embedded).shape)

## RESIDUAL
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x
    


## FEEDFORWARD
class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size: int, expansion: int = 4, drop_p: float = 0.0):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )



## MAKE ENCODER BLOCK
class TransformerEncoderBlock(nn.Sequential):
    def __init__(self, emb_size: int = 768, drop_out: float = 0.1, 
                 forward_expansion: int = 4, forward_drop_p: float = 0.0, **kwargs):
        super().__init__(
            Residual(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, **kwargs),
                nn.Dropout(drop_out)
            )),
            Residual(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion = forward_expansion, drop_p = forward_drop_p
                ),
                nn.Dropout(drop_out)
            ))
            # nn.Sequential(
            #     nn.LayerNorm(emb_size),
            #     MultiHeadAttention(emb_size, **kwargs),
            #     nn.Dropout(drop_out)
            # ),
            # nn.Sequential(
            #     nn.LayerNorm(emb_size),
            #     FeedForwardBlock(
            #         emb_size, expansion = forward_expansion, drop_p = forward_drop_p
            #     ),
            #     nn.Dropout(drop_out)
            # )
        )



# print(TransformerEncoderBlock()(PatchEmbedding()(image)).shape)  


## TRANSFORMERENCODERBLOCK
class TransformerEncoder(nn.Sequential):
    def __init__( self, depth: int = 12, **kwargs):
        super().__init__(*[TransformerEncoderBlock(**kwargs) for _ in range(depth)])
        # super().__init__(*[Residual(TransformerEncoderBlock(**kwargs)) for _ in range(depth)]) ## wrong single residual


#CLASSIFICATION HEAD
class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size: int = 768, n_classes: int = 1000):
        super().__init__(
            Reduce('b n e -> b e', reduction = 'mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, n_classes)
        )


#VIT Model

class ViT(nn.Sequential):
    def __init__(self, in_channels: int = 3, patch_size: int = 16, embedding_size: int = 768,
                 img_size: int = 224, depth: int = 12, n_classes: int = 1000, **kwargs):
        super().__init__(
            PatchEmbedding(in_channels, embedding_size, patch_size, img_size),
            TransformerEncoder(depth, emb_size = embedding_size, *kwargs),
            ClassificationHead(embedding_size, n_classes)
        )

# summary(ViT(), (3, 224, 224), device = 'cpu')