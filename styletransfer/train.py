from trainer import Trainer
from hyperparams import Hyperparams as hp

def main():

    modeltrainer = Trainer()

    modeltrainer.update(hp.__dict__)

    _log_str = "\n"
    params = "{0:25} | {1:25}"
    for k,v in modeltrainer.__dict__.items():
        _log_str += params.format(k,str(v)) + "\n"
    print(_log_str)

    modeltrainer.build()

    modeltrainer.train()

if __name__ == '__main__':
    main()