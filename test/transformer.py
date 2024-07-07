import sys
sys.path.append('../../')

from dlx.modules.transformer import (
    TransformerDecoderUnit,
    TransformerEncoderUnit,
    Transformer
)
import torch

model = Transformer()
input = torch.randn((2, 10, 512))
output = model(input)
pass