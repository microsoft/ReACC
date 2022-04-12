# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaModel
import logging

logger = logging.getLogger(__name__)

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

class RobertaLMHead(nn.Module):
    """Roberta Head for masked language modeling."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = F.gelu(x)
        x = self.layer_norm(x)

        # project back to size of vocabulary with bias
        x = self.decoder(x)

        return x

class SimpleModel(nn.Module):
    def __init__(self, config, args):
        super().__init__()
        if args.from_pretrained:
            self.encoder = RobertaModel.from_pretrained(args.pretrained_dir, add_pooling_layer=False)
            logger.warning(f"Loading encoder from {args.pretrained_dir}")
        else:
            self.encoder = RobertaModel(config, add_pooling_layer=False)
        self.encoder.resize_token_embeddings(args.vocab_size)

        self.config = config
        self.args = args
        self.lm_head = RobertaLMHead(config)
        self.n_vec = max(0, self.args.num_vec)
        self.tie_weights()

    def _tie_or_clone_weights(self, first_module, second_module):
        if self.config.torchscript:
            first_module.weight = nn.Parameter(second_module.weight.clone())
        else:
            first_module.weight = second_module.weight

    def tie_weights(self):
        self._tie_or_clone_weights(self.lm_head.decoder,
                                   self.encoder.embeddings.word_embeddings)  

    def forward(self, inputs_m, inputs1=None, inputs2=None, attn_mask=None, attn_mask1=None, attn_mask2=None, mlm_labels=None):
        outputs = self.encoder(inputs_m, attention_mask=attn_mask)[0]

        if inputs1 is None and inputs2 is None:
            # infer
            if self.n_vec > 0:
                outputs = nn.functional.normalize(outputs[:, :self.n_vec, :], dim=2)
            else:
                outputs = nn.functional.normalize(outputs[:, 0, :], dim=1)
            return outputs
        
        # training
        outputs1 = self.encoder(inputs1, attention_mask=attn_mask1)[0][:, 0, :]
        outputs2 = self.encoder(inputs2, attention_mask=attn_mask2)[0]

        lm_logits = self.lm_head(outputs)

        if mlm_labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            mlm_loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), mlm_labels.view(-1))
        else:
            mlm_loss = None

        if self.n_vec > 0:
            o1 = nn.functional.normalize(outputs1, dim=1)
            o2 = nn.functional.normalize(outputs2[:, :self.n_vec, :], dim=2)
            logits, _ = torch.max(torch.einsum('nc,mvc->nmv', [o1, o2]), -1)
        else:
            o1 = nn.functional.normalize(outputs1, dim=1)
            o2 = nn.functional.normalize(outputs2[:, 0, :], dim=1)
            logits = torch.einsum('nc,mc->nm', [o1, o2])
        logits /= self.args.moco_T
        labels = torch.arange(end=logits.shape[0], dtype=torch.long).cuda()

        nce_loss = F.cross_entropy(logits, labels)

        return lm_logits, mlm_loss, nce_loss