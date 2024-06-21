from operator import itemgetter
import os
import logging
import argparse
from tqdm import tqdm, trange

from kobert_tokenizer import KoBERTTokenizer

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from transformers import AutoModelForTokenClassification

from utils import init_logger, get_labels

logger = logging.getLogger(__name__)


def get_device(no_cuda):
    return "cuda" if torch.cuda.is_available() and not no_cuda else "cpu"


def get_args(model_dir, epoch):
    return torch.load(os.path.join(model_dir, f'training_args_epoch_{epoch}.bin'))


def load_model(model_dir, device, args):
    
    # Check whether model exists
    if not os.path.exists(model_dir):
        raise Exception("Model doesn't exists! Train first!")

    try:
        model = AutoModelForTokenClassification.from_pretrained(model_dir)  # Config will be automatically loaded from model_dir
        model.to(device)
        model.eval()
        logger.info("***** Model Loaded *****")
    except:
        raise Exception("Some model files might be missing...")

    return model


def convert_input_file_to_tensor_dataset(lines,
                                         args,
                                         tokenizer,
                                         pad_token_label_id,
                                         cls_token_segment_id=0,
                                         pad_token_segment_id=0,
                                         sequence_a_segment_id=0,
                                         mask_padding_with_zero=True):
    # Setting based on the current model type
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    unk_token = tokenizer.unk_token
    pad_token_id = tokenizer.pad_token_id

    all_input_ids = []
    all_attention_mask = []
    all_token_type_ids = []
    all_slot_label_mask = []

    for words in lines:
        tokens = []
        slot_label_mask = []
        for word in words:
            word_tokens = tokenizer.tokenize(word)
            if not word_tokens:
                word_tokens = [unk_token]  # For handling the bad-encoded word
            tokens.extend(word_tokens)
            # Use the real label id for the first token of the word, and padding ids for the remaining tokens
            slot_label_mask.extend([0] + [pad_token_label_id] * (len(word_tokens) - 1))

        # Account for [CLS] and [SEP]
        special_tokens_count = 2
        if len(tokens) > args.max_seq_len - special_tokens_count:
            tokens = tokens[: (args.max_seq_len - special_tokens_count)]
            slot_label_mask = slot_label_mask[:(args.max_seq_len - special_tokens_count)]

        # Add [SEP] token
        tokens += [sep_token]
        token_type_ids = [sequence_a_segment_id] * len(tokens)
        slot_label_mask += [pad_token_label_id]

        # Add [CLS] token
        tokens = [cls_token] + tokens
        token_type_ids = [cls_token_segment_id] + token_type_ids
        slot_label_mask = [pad_token_label_id] + slot_label_mask

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = args.max_seq_len - len(input_ids)
        input_ids = input_ids + ([pad_token_id] * padding_length)
        attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)
        slot_label_mask = slot_label_mask + ([pad_token_label_id] * padding_length)

        all_input_ids.append(input_ids)
        all_attention_mask.append(attention_mask)
        all_token_type_ids.append(token_type_ids)
        all_slot_label_mask.append(slot_label_mask)

    # Change to Tensor
    all_input_ids = torch.tensor(all_input_ids, dtype=torch.long)
    all_attention_mask = torch.tensor(all_attention_mask, dtype=torch.long)
    all_token_type_ids = torch.tensor(all_token_type_ids, dtype=torch.long)
    all_slot_label_mask = torch.tensor(all_slot_label_mask, dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_slot_label_mask)

    return dataset


def predict(input_list, output_file, model_dir, epoch, batch_size, no_cuda):
    # load model and args
    args = get_args(model_dir, epoch)
    device = get_device(no_cuda)
    model = load_model(model_dir, device, args)
    label_lst = get_labels(args)
    logger.info(args)

    # Convert input file to TensorDataset
    pad_token_label_id = torch.nn.CrossEntropyLoss().ignore_index
    tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
    if len(input_list) == 0:
        return [], []
    lines = input_list  #read_input_file(pred_config)
    dataset = convert_input_file_to_tensor_dataset(lines, args, tokenizer, pad_token_label_id)

    # Predict
    sampler = SequentialSampler(dataset)
    data_loader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)

    all_slot_label_mask = None
    preds = None

    for batch in data_loader:
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            inputs = {"input_ids": batch[0],
                      "attention_mask": batch[1],
                      "labels": None}
            if args.model_type != "distilkobert":
                inputs["token_type_ids"] = batch[2]
            outputs = model(**inputs)
    
            logits = outputs[0]

            if preds is None:
                preds = logits.detach().cpu().numpy()
                all_slot_label_mask = batch[3].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                all_slot_label_mask = np.append(all_slot_label_mask, batch[3].detach().cpu().numpy(), axis=0)

    preds = np.argmax(preds, axis=2)
    slot_label_map = {i: label for i, label in enumerate(label_lst)}
    preds_list = [[] for _ in range(preds.shape[0])]

    for i in range(preds.shape[0]):
        for j in range(preds.shape[1]):
            if all_slot_label_mask[i, j] != pad_token_label_id:
                preds_list[i].append(slot_label_map[preds[i][j]])

    # Write to output file
    OG_list = []
    PS_list = []
    pair_list = []
    with open(output_file, "w", encoding="utf-8") as f:
        for words, preds in zip(lines, preds_list):
            line = ""
            OG_temp = ''
            PS_temp = ''
            tag_list = []
            hold_OG = []
            hold_PS = []
            new_OG = []
            new_PS = []
            open_idx_list = []

            for word, pred in zip(words, preds):
                if pred == 'O':
                    line = line + word + " "
                    if word == '(':
                        open_idx_list.append(len(tag_list) + len(hold_OG) + len(hold_PS))
                    tag_list.append(['O', word])
                else:
                    if pred[:2] != 'OG' and pred[:2] != 'PS':
                        tag_list.append(['O', word])

                    # B 태그인지 I 태그인지 확인 필요
                    if pred[:2] == 'OG' and pred[3] == 'B':
                        if len(OG_temp) == 0:
                            hold_OG.append(len(tag_list) + len(hold_OG) + len(hold_PS))
                            OG_temp = word
                        else:
                            OG_list.append(OG_temp)
                            new_OG.append(OG_temp)
                            hold_OG.append(len(tag_list) + len(hold_OG) + len(hold_PS))
                            OG_temp = word
                    elif pred[:2] == 'OG' and pred[3] == 'I':
                        if len(OG_temp) == 0:
                            print(pred)
                            print("wrong tag: ", word)
                            # 이 경우 어떻게 처리할 지 생각
                        else:
                            OG_temp = OG_temp + ' ' + word
                    if pred[:2] == 'PS' and pred[3] == 'B':
                        if len(PS_temp) == 0:
                            PS_temp = word
                            hold_PS.append(len(tag_list) + len(hold_PS) + len(hold_OG))
                        else:
                            PS_list.append(PS_temp)
                            new_PS.append(PS_temp)
                            hold_PS.append(len(tag_list) + len(hold_PS) + len(hold_OG))
                            PS_temp = word
                    elif pred[:2] == 'PS' and pred[3] == 'I':
                        if len(PS_temp) == 0:
                            print(pred)
                            print("wrong tag: ", word)
                            # 이 경우 어떻게 처리할 지 생각
                        else:
                            PS_temp = PS_temp + ' ' + word

                    line = line + "[{}:{}] ".format(word, pred)
            
            if len(hold_OG) > len(new_OG):
                new_OG.append(OG_temp)
                OG_list.append(OG_temp)
            if len(hold_PS) > len(new_PS):
                new_PS.append(PS_temp)
                PS_list.append(PS_temp)

            hold_list = []
            if len(hold_OG) > 0:
                hold_list = [{'index': idx, 'data': ['OG', word_OG]} for idx, word_OG in zip(hold_OG, new_OG)]
            if len(hold_PS) > 0:
                hold_list += [{'index': idx, 'data': ['PS', word_PS]} for idx, word_PS in zip(hold_PS, new_PS)]
            
            hold_list = sorted(hold_list, key=itemgetter('index'))
            for hold in hold_list:
                tag_list.insert(hold['index'], hold['data'])

            # pair_list에 추가
            for open_idx in open_idx_list:
                if open_idx == 0 or open_idx == len(tag_list) - 1:
                    # 여는 괄호가 맨 앞에 있거나 맨 뒤에 있는 경우
                    continue
                left_item = tag_list[open_idx - 1]
                right_item = tag_list[open_idx + 1]

                if left_item[0] == 'OG' and right_item[0] == 'OG':
                    pair_list.append((left_item[1], right_item[1]))
                if left_item[0] == 'PS' and right_item[0] == 'PS':
                    pair_list.append((left_item[1], right_item[1]))

            f.write("{}\n".format(line.strip()))
    
    logger.info("Prediction Done!")
    return OG_list, PS_list, pair_list

if __name__ == "__main__":
    init_logger()
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_file", default="sample_pred_in.txt", type=str, help="Input file for prediction")
    parser.add_argument("--output_file", default="sample_pred_out.txt", type=str, help="Output file for prediction")
    parser.add_argument("--model_dir", default="./model", type=str, help="Path to save, load model")

    parser.add_argument("--batch_size", default=32, type=int, help="Batch size for prediction")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")

    pred_config = parser.parse_args()
    #predict(pred_config)
