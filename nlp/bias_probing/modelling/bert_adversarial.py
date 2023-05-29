import os
import re
from os.path import abspath
from typing import Iterable, Union
import pandas as pd
import numpy as np

import torch
from torch import nn
from transformers import AutoModelForSequenceClassification, WEIGHTS_NAME, is_wandb_available
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.bert import BertConfig, BertForSequenceClassification, BertPreTrainedModel, BertModel
from transformers.utils.logging import get_logger
from torch.autograd  import  Function


if is_wandb_available():
    import wandb

logger = get_logger()

class GradientReversalFunction(Function):
    """
    Gradient Reversal Layer from:
    Unsupervised Domain Adaptation by Backpropagation (Ganin & Lempitsky, 2015)
    Forward pass is the identity function. In the backward pass,
    the upstream gradients are multiplied by -lambda (i.e. gradient is reversed)
    """

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grads):
        lambda_ = ctx.lambda_
        lambda_ = grads.new_tensor(lambda_)
        dx = -lambda_ * grads
        return dx, None


class GradientReversal(torch.nn.Module):
    def __init__(self, lambda_=1):
        super(GradientReversal, self).__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)


class BertAdversarialConfig(BertConfig):

    def __init__(self,
                 num_z_labels=3,
                 lambda_reversal=1.0,
                 *args, **kwargs
                 ):

        super().__init__(*args, **kwargs)
        self.num_z_labels = num_z_labels
        self.lambda_reversal = lambda_reversal
        
        
class BertAdversarial(BertPreTrainedModel):
    config_class = BertAdversarialConfig

    def __init__(self, config: BertAdversarialConfig):
        super().__init__(config)
        # self.weak_learner = weak_learner
        # self.bert = bert
        self.config = config
        self.num_labels = config.num_labels
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.bert = BertModel(config)
        self.loss_fn = nn.CrossEntropyLoss()
        self.lambda_reversal = self.config.lambda_reversal
        
        self.z_loss = nn.CrossEntropyLoss()
        self.z_model = nn.Sequential(GradientReversal(self.lambda_reversal),
                                     nn.Linear(config.hidden_size, config.num_z_labels))
        

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None, z_labels=None, id=None, **kwargs):
        inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids
        }
        _, pooled_output = self.bert(**inputs, return_dict=False)
        main_logits = self.classifier(self.dropout(pooled_output))
        main_loss = self.loss_fn(main_logits, labels)
        
        if not self.training:
            return main_loss, main_logits

        z_logits = self.z_model(pooled_output)
        z_loss = self.z_loss(z_logits, z_labels)
        
        logits = main_logits
        loss = main_loss + z_loss
        
        with open("/scratch/nhj4247/robustness/bias-probing/scripts/temp.txt", "a") as f:
            f.write("z_loss: " + str(z_loss.data) + "\n")
        
        return loss, logits

    