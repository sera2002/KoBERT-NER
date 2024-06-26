from __future__ import absolute_import, division, print_function, unicode_literals
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import BertModel, BertConfig
from torchcrf import CRF
from kobert.pytorch_kobert import get_pytorch_kobert_model, bert_config

class KobertCRF(nn.Module):
    """ KoBERT with CRF """
    def __init__(self, config, num_classes, vocab=None) -> None:
        super(KobertCRF, self).__init__()

        if vocab is None:
            self.bert, self.vocab = get_pytorch_kobert_model()
        else:
            self.bert = BertModel.from_pretrained("monologg/kobert", config=BertConfig.from_dict(bert_config))
            self.vocab = vocab

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.position_wise_ff = nn.Linear(config.hidden_size, num_classes)
        self.crf = CRF(num_tags=num_classes, batch_first=True)

    def forward(self, input_ids, token_type_ids=None, tags=None, mask=None):
        #print("forward start")
        if mask is None:
            attention_mask = input_ids.ne(self.vocab.token_to_idx[self.vocab.padding_token]).float()
        else:
            attention_mask = mask
        #print("attention_mask: ", attention_mask)

        # outputs: (last_encoder_layer, pooled_output, attention_weight)
        outputs = self.bert(input_ids=input_ids,
                            token_type_ids=token_type_ids,
                            attention_mask=attention_mask)
        last_encoder_layer = outputs[0]
        last_encoder_layer = self.dropout(last_encoder_layer)
        emissions = self.position_wise_ff(last_encoder_layer)
        
        #print("output: ", outputs)
        if tags is not None:
            #print("here is started")
            # print("input_ids: ", input_ids)
            # print("vocab의 padding_token: ", self.vocab.padding_token)
            # print("padding_token의 index: ", self.vocab.token_to_idx[self.vocab.padding_token])
            # print("attention_mask: ", attention_mask)
            # print("tags: ", tags)
            
            # change pad token label ids to -1
            tags[tags == -100] = 0

            # make mask for tags
            log_likelihood = -self.crf(emissions, tags, mask=attention_mask.byte(), reduction="mean")
            sequence_of_tags = self.crf.decode(emissions, mask=attention_mask.byte())
            return log_likelihood, sequence_of_tags
        else:
            sequence_of_tags = self.crf.decode(emissions)
            return sequence_of_tags
