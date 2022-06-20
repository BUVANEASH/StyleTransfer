import json
import argparse

from trainer import Trainer

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset','-d', type=str, default = '../data/raw',
                        help='Path to dataset')
    parser.add_argument('--config','-c', type=str, default = 'config/config.json',
                        help='Path to model config file')
    parser.add_argument('--logdir','-l', type=str, default = '../results/models',
                        help='Log directory path')
    
    args = parser.parse_args()

    config = json.load(open(args.config, 'r'))
    config.update({"raw_data" : args.dataset,
                   "logdir" : args.logdir})

    trainer = Trainer(**config)

    _log_str = "\n"
    params = "{0:25} | {1:25}"
    for k,v in trainer.__dict__.items():
        _log_str += params.format(k,str(v)) + "\n"
    print(_log_str)

    trainer.build()

    trainer.train()

if __name__ == '__main__':
    main()