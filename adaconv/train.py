from trainer import Trainer
from hyperparams import Hyperparams as hp

def main():

    modeltrainer = Trainer()

    modeltrainer.update(hp.__dict__)

    params = "{0:25} | {1:25}"
    for k,v in modeltrainer.__dict__.items():
        print(params.format(k,str(v)))

    modeltrainer.build()

    modeltrainer.train()

if __name__ == '__main__':
    main()