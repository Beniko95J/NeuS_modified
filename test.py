import models.ref_utils as ref_utils
import torch


dir_enc_fn = ref_utils.generate_ide_fn(4)
refdirs = torch.randn(10000, 3)
roughness = 0
dir_enc = dir_enc_fn(refdirs, roughness)
print(dir_enc.shape)
