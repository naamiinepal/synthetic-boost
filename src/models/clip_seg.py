from torch import nn
from transformers import CLIPSegForImageSegmentation


class CLIPSeg(nn.Module):
    def __init__(
        self,
        clip_seg_hf_api: str,
        freeze_clip: bool = True,
        freeze_decoder: bool = True,
    ):
        super().__init__()

        self.clip_seg = CLIPSegForImageSegmentation.from_pretrained(clip_seg_hf_api)

        self.clip_seg.clip.requires_grad_(not freeze_clip)
        self.clip_seg.decoder.requires_grad_(not freeze_decoder)

    def forward(self, **kwargs):
        B, C, H, W = kwargs["pixel_values"].shape

        outputs = self.clip_seg(**kwargs)

        return outputs.logits.view(B, 1, H, W)
