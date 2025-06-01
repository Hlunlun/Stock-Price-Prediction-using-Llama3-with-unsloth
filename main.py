import os
import sys
import unsloth
import argparse
from utils import *
from stock_api import StockAPI
from stock_predictor import StockPredictor
# from ui import MainWindow
# from PyQt5.QtWidgets import QApplication

def argparse_args():
    parser = argparse.ArgumentParser(description='Stock Prediction')

    # Mode
    parser.add_argument('--train', type=int, default=1)

    # Data
    parser.add_argument('--data-dir', type=str, default='data')
    parser.add_argument('--company-list-file', type=str, default='tech_cop_news.json')
    parser.add_argument('--year', type=int, default=2024)
    parser.add_argument('--month', type=int, default=9)
    parser.add_argument('--choose-stock', type=int, default=1)
    # parser.add_argument('--fields', type=list, default=['date', 'capacity', 'turnover', 'open', 'high', 'low', 'close', 'change', 'transaction'])
    # parser.add_argument('--fields', type=list, default=['date', 'turnover', 'open', 'high', 'low', 'close', 'transaction'])
    # parser.add_argument('--fields', type=list, default=['date', 'open', 'high', 'low', 'close', 'transaction'])
    parser.add_argument('--fields', type=list, default=['date', 'open', 'high', 'low', 'close', 'change', 'transaction'])

    # Load Pretrained Model
    parser.add_argument('--model-name', type=str, default = "unsloth/Llama-3.2-3B-Instruct", 
                        choices=["unsloth/Llama-3.2-3B-Instruct", "unsloth/Meta-Llama-3.1-8B-bnb-4bit",])
    parser.add_argument('--max-seq-length', type=int, default = 2048)
    parser.add_argument('--load_in_4bit', type=bool, default= True)
    parser.add_argument('--load_in_8bit', type=bool, default= False)
    parser.add_argument('--model-dir', type=str, default='model')

    # LoRA
    parser.add_argument('--rank', type=int, default = 16)
    parser.add_argument('--lora-alpha', type=int, default = 8)
    parser.add_argument('--lora-dropout', type=float, default = 0)
    parser.add_argument('--bias', type=str, default = "none")
    parser.add_argument('--random-state', type=int, default = 3407)

    # Trainer
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--gradient-accumulation-steps', type=int, default=8)
    parser.add_argument('--warmup-steps', type=int, default=5)
    parser.add_argument('--num-epochs', type=int, default=1)
    parser.add_argument('--max-steps', type=int, default=100)
    parser.add_argument('--learning-rate', type=float, default=2e-4)
    parser.add_argument('--logging-steps', type=int, default=1)
    parser.add_argument('--optim', type=str, default="adamw_8bit")
    parser.add_argument('--weight-decay', type=float, default=0.05)
    parser.add_argument('--lr-scheduler-type', type=str, default="cosine")
    parser.add_argument('--output-dir', type=str, default="outputs")
    parser.add_argument('--report-to', type=str, default="tensorboard")

    args = parser.parse_args()
    return args

def system_init(args):    
    predictor = StockPredictor(args)
    if args.train == 1:    
        predictor.train();
    else:
        # predictor.inference()
        predictor.plot_compare_graph()
        # app = QApplication(sys.argv)
        # window = MainWindow()
        # window.show()


def main():
    args = argparse_args()
    system_init(args)


if __name__ == '__main__':
    main()