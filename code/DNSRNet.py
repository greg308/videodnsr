import sys
import os
from NEU import NEUUpsample
from NRMA import NRMANet
from Trainer import Trainer
from Tester import Tester
from LogValueObserver import LogValueObserver
import Vimeo90K
from Settings import SETTINGS_DEFAULT

class DNSR:
    def __init__(self):
        self.settings = SETTINGS_DEFAULT

    def run(self):
        if len(sys.argv) > 1:
            if sys.argv[1] == "train":
                if len(sys.argv) == 4 and os.path.isfile(sys.argv[3]):
                    self.train(sys.argv[2], sys.argv[3])
                    return
            elif sys.argv[1] == "eval":
                if len(sys.argv) == 4 and os.path.isfile(sys.argv[3]):
                    self.evaluate(sys.argv[2], sys.argv[3])
                    return

        print("Please provide correct command line arguments")
        sys.exit(0)
        
    def prepare_data(self):
        Vimeo90K.prepare_data(self.settings["scale"])

    def train(self, model_name, training_path):
        model = NRMANet().to('cuda')
        model.load_model(model_name)

        trainer = Trainer(model, model_name, self.settings, training_path)

        trainer.load_training_state()
        trainer.train()
        trainer.save_training_state()
        
    def evaluate(self, model_name, evaluation_path):
        model = NRMANet().to('cuda')
        model.load_model(model_name)

        evaluator = Tester(model, model_name, self.settings, evaluation_path, False)
        evaluator.eval()

    def display(self, model_name, evaluation_path):
        model = NRMANet().to('cuda')
        model.load_model(model_name)

        evaluator = Tester(model, model_name, self.settings, evaluation_path, True)
        evaluator.eval()

    def observe_log(self, tag, file_path):
        value_observer = LogValueObserver()
        value_observer.observe(tag, file_path)
        value_observer.show_observations()

if __name__ == "__main__":
    dnsr = DNSR()
    dnsr.run()
