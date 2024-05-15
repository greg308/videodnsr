import os
import random
import numpy as np
import matplotlib.pyplot as plt
import signal
import threading
import torch
from Logger import Logger
from Loss import CharbonnierLoss
from LRScheduler import CosineAnnealingLR_Restart
import Vimeo90K

class Trainer:
    def __init__(self, model, model_name, settings, train_list_path):
        self.model = model
        self.model_name = model_name
        self.settings = settings
        self.train_list_path = train_list_path
        self.logger = Logger(self.model_name)

        self.stop_training = False
        self.epoch = 0
        self.iter = 0

        self.train_set = Vimeo90K.create_dataset(self.settings, Vimeo90K.read_image_paths_from_file(self.train_list_path))
        self.train_loader = Vimeo90K.create_dataloader(self.train_set, self.settings["batch_size"])

        self.loss_function = CharbonnierLoss().to('cuda')

        # optimiser
        optim_params = []
        for _, v in self.model.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
        self.optimiser = torch.optim.AdamW(optim_params, lr=self.settings["training"]["learning_rate"],
                                            weight_decay=self.settings["training"]["weight_decay"],
                                            betas=(self.settings["training"]["beta1"], self.settings["training"]["beta2"]))
        # schedulers
        self.scheduler = CosineAnnealingLR_Restart(
                self.optimiser, self.settings["training"]["T_period"], 
                eta_min=self.settings["training"]["eta_min"],
                restarts=self.settings["training"]["restarts"], 
                weights=self.settings["training"]["restart_weights"]
        )

    def train(self):
        self.model.train()

        def signal_handler():
            print("Stopping training. Please wait...")
            self.stop_training = True
            plt.close()

        # run the signal handler on a new thread so the print statements dont conflict with ones which are already running while interrupted
        signal.signal(signal.SIGINT, lambda _, __: threading.Timer(0.01, signal_handler).start())

        while not self.stop_training and self.epoch < self.settings['num_epochs']:
            self.logger.log("Epoch", f"Epoch {self.epoch}/{self.settings['num_epochs']}")

            for _, train_data in enumerate(self.train_loader):
                if self.iter > self.settings["max_iters_per_epoch"]:
                    break

                loss = self.train_one_sequence(train_data)

                current_iter = self.iter
                self.iter += 1

                # update learning rate
                self.update_learning_rate(current_iter, warmup_iter=self.settings["training"]["warmup_iter"])

                self.logger.log("Iter", f"Epoch: {self.epoch} | Iter: {current_iter} | Loss: {loss}")
                self.logger.silent_log("Loss", loss)

                if current_iter != 0 and current_iter % self.settings["training_save_interval"] == 0:
                    self.model.save_model(self.model_name)
                    self.save_training_state()

            self.epoch += 1
            self.iter = 0

            self.model.save_model(self.model_name)
            self.save_training_state()

        self.model.save_model(self.model_name)
        self.save_training_state()

    def train_one_sequence(self, train_data):
        inputs = train_data["LRs"].to('cuda')   #[5, 4, 3, 64, 96]
        targets = train_data['HRs'].to('cuda')  #[5, 7, 3, 256, 384]

        # pick a random timestamp for the 2 input frames
        _, num_input_frames, _, _, _ = inputs.size()
        i = random.randint(0, num_input_frames-2)
        j=i+1
        batch_size = len(inputs)

        input_frames = inputs[:, i:j+1, :, :, :]
        target_frames = targets[:, 2*i:2*j+1, :, :, :]
        self.model.calc_encoder(inputs)
        output_frames = self.model(input_frames, [(i, j)] * batch_size)

        # display the first output of the batch
        # self.observe_sequence(input_frames[1], output_frames[1], target_frames[1])

        #compute loss
        loss = self.loss_function(output_frames, target_frames)
        
        self.optimiser.zero_grad() #zero the gradients
        loss.backward() #backward pass
        self.optimiser.step() #update weights

        # return avg loss
        return loss.item()

    def observe_sequence(self, input, output, target):
        num_outputs = output.size()[0]
        fig, axs = plt.subplots(3, num_outputs, figsize=(12,8), sharex=True, sharey=True)
        fig.subplots_adjust(wspace=0, hspace=0)
        for i in range(num_outputs):
            
            current_output = output[i].permute(1, 2, 0).cpu().detach().clone().numpy()
            current_target = target[i].permute(1, 2, 0).cpu().clone().numpy()

            if i % 2 == 0:
                current_input = input[i//2].permute(1, 2, 0).cpu().numpy()
                current_input = np.repeat(np.repeat(current_input, self.settings["scale"], axis=0), self.settings["scale"], axis=1)
                axs[0][i].imshow(current_input)
                
            axs[0][i].axis('off')
            axs[1][i].imshow(current_output)
            axs[1][i].axis('off')
            axs[2][i].imshow(current_target)
            axs[2][i].axis('off')

        plt.show()

    def _set_lr(self, lr_groups):
        ''' set learning rate for warmup,
        lr_groups_l: list for lr_groups. each for a optimizer'''
        for param_group, lr in zip(self.optimizer.param_groups, lr_groups):
            param_group['lr'] = lr

    def _get_init_lr(self):
        # get the initial lr, which is set by the scheduler
        init_lr_groups = [v['initial_lr'] for v in self.optimizer.param_groups]
        return init_lr_groups

    def update_learning_rate(self, cur_iter, warmup_iter=-1):
        self.scheduler.step()
        # set up warm up learning rate
        if cur_iter < warmup_iter:
            # get initial lr for each group
            init_lr_groups = self._get_init_lr()
            # modify warming-up learning rates
            warmup_lr = [v / warmup_iter * cur_iter for v in init_lr_groups]
            # set learning rate
            self._set_lr(warmup_lr)

    def load_training_state(self):
        training_state_path = f"training_states/{self.model_name}_state.pth"

        if os.path.exists(training_state_path):
            training_state = torch.load(training_state_path)
            self.epoch = training_state["epoch"]
            self.iter = training_state["iter"]
            self.optimiser.load_state_dict(training_state["optimiser"])
            self.scheduler.load_state_dict(training_state["scheduler"])
            print(f"Loaded training state: {self.epoch}-{self.iter}")
        else:
            print(f"{training_state_path} doesn't exist.")

    def save_training_state(self):
        training_state_path = f"training_states/{self.model_name}_state.pth"
        
        training_state = {
            "epoch": self.epoch,
            "iter": self.iter,
            "optimiser": self.optimiser.state_dict(),
            "scheduler": self.scheduler.state_dict()
        }

        if not os.path.exists("training_states"):
            os.makedirs("training_states")

        torch.save(training_state, training_state_path)
        self.logger.log("Save", "Saved training state")


        
