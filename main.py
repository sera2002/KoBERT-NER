import argparse

from kobert_tokenizer import KoBERTTokenizer
import gluonnlp as nlp

from trainer import Trainer
from trainer_crf import TrainerCRF
from utils import init_logger, load_tokenizer, set_seed, MODEL_CLASSES, MODEL_PATH_MAP
from data_loader import load_and_cache_examples


def main(args):
    init_logger()
    set_seed(args)
    
    tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
    vocab = nlp.vocab.BERTVocab.from_sentencepiece(tokenizer.vocab_file, padding_token='[PAD]')
    #load_tokenizer(args)

    train_dataset = None
    dev_dataset = None
    test_dataset = None

    if args.do_eval:
        test_dataset = load_and_cache_examples(args, tokenizer, mode="test")
    if args.do_train:
        train_dataset = load_and_cache_examples(args, tokenizer, mode="train")
        dev_dataset = load_and_cache_examples(args, tokenizer, mode="dev")
        print("validation dataset 크기 : ", len(dev_dataset))

    if args.crf:
        trainer = TrainerCRF(args, train_dataset, dev_dataset, test_dataset, vocab)
    else:
        trainer = Trainer(args, train_dataset, dev_dataset, test_dataset)

    if args.do_train:
        trainer.train()

    if args.do_eval:
        trainer.load_model()
        trainer.evaluate("test", "eval")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--crf", default=False, type=bool, help="CRF Layer")
    parser.add_argument("--task", default="ner", type=str, help="The name of the task to train")
    parser.add_argument("--model_dir", default="./model", type=str, help="Path to save, load model")
    parser.add_argument("--data_dir", default="./data", type=str, help="The input data dir")
    parser.add_argument("--pred_dir", default="./preds", type=str, help="The prediction file dir")

    parser.add_argument("--train_file", default="data_NXNE/train.tsv", type=str, help="Train file")
    parser.add_argument("--test_file", default="data_NXNE/test.tsv", type=str, help="Test file")
    parser.add_argument("--dev_file", default="data_NXNE/validation.tsv", type=str, help="Validation file")
    parser.add_argument("--label_file", default="label.txt", type=str, help="Slot Label file")
    parser.add_argument("--write_pred", action="store_true", help="Write prediction during evaluation")

    parser.add_argument("--model_type", default="kobert", type=str, help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))

    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")
    parser.add_argument("--train_batch_size", default=32, type=int, help="Batch size for training.")
    parser.add_argument("--eval_batch_size", default=64, type=int, help="Batch size for evaluation.")
    parser.add_argument("--max_seq_len", default=50, type=int, help="The maximum total input sequence length after tokenization.")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=20.0, type=float, help="Total number of training epochs to perform.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--max_steps", default=-1, type=int, help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")

    parser.add_argument('--logging_steps', type=int, default=1000, help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=1000, help="Save checkpoint every X updates steps.")

    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the test set.")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")

    args = parser.parse_args()

    args.model_name_or_path = MODEL_PATH_MAP[args.model_type]
    main(args)
