import torch
import torch.nn as nn

from transformers import Blip2QFormerModel, Blip2QFormerConfig


class PerceiverResampler(nn.Module):
    def __init__(
        self,
        num_queries=32,
        hidden_size=768,
        qk_normalization=False,
        gradient_checkpointing=True,
        **kwargs
    ) -> None:
        super().__init__()

        config = Blip2QFormerConfig(hidden_size=hidden_size, **kwargs)
        config.qk_normalization = qk_normalization
        self.blip2qformer = Blip2QFormerModel(config)

        self.queries = nn.Parameter(torch.zeros(1, num_queries, hidden_size))
        self.queries.data.normal_(0, config.initializer_range)
        if gradient_checkpointing:
            self.blip2qformer.gradient_checkpointing_enable()

    def forward(self, **kwargs):
        query_embeds = kwargs.pop("query_embeds", self.queries)

        return self.blip2qformer(query_embeds=query_embeds, **kwargs)
