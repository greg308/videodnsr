import numpy as np
import matplotlib.pyplot as plt
import torch
import Vimeo90K

from skimage.metrics import peak_signal_noise_ratio

class Tester:
    def __init__(self, model, model_name, settings, vimeo_path, image_path, num_input_images):
        self.model = model
        self.model_name = model_name
        self.settings = settings
        self.vimeo_path = vimeo_path
        self.image_path = image_path
        self.num_input_images = num_input_images

        self.test_set = Vimeo90K.create_dataset(self.settings, [image_path])
        self.test_loader = Vimeo90K.create_dataloader(self.test_set, 1)


    def eval(self):
        PSNRS = []
        for _, test_data in enumerate(self.test_loader):
            PSNR = self.eval_one_sequence(test_data)
            PSNRS.append(PSNR)
        
        total_avg_PSNR = sum(PSNRS)/len(PSNRS)
        print(f"Total Average PSNR: {total_avg_PSNR}")


    def eval_one_sequence(self, test_data):
        inputs = test_data["LRs"].to('cuda')   #[1, 4, 3, 64, 96]
        targets = test_data['HRs'].to('cuda')  #[1, 7, 3, 256, 384]

        batch_size = len(inputs)
        input_sequence = inputs[0]
        output_sequence = torch.tensor([]).to('cuda')
        target_sequence = targets[0]

        self.model.calc_encoder(inputs)
        for i in range(len(input_sequence)-1):
            j = i+1

            input_frames = inputs[:, i:j+1, :, :, :]
            output_frames = self.model(input_frames, [(i, j)] * batch_size)

            #take the first 2 out of 3 outputs for all but the last frame pair. for the last, take all 3 outputs
            if j == len(input_sequence)-1:
                output_sequence = torch.cat((output_sequence, output_frames[0]), dim=0)
            else:
                output_sequence = torch.cat((output_sequence, output_frames[0, :2]), dim=0)

        # calc psnrs
        PSNRs = []
        for (output_im, target_im) in zip(output_sequence, target_sequence):
            output_im = output_im.permute(1, 2, 0).cpu().detach().numpy()
            target_im = target_im.permute(1, 2, 0).cpu().detach().numpy()
            PSNRs.append(peak_signal_noise_ratio(target_im, output_im))
        avg_PSNR = sum(PSNRs)/len(PSNRs)

        # restrict to the requested input size
        input_sequence = input_sequence[:self.num_input_images, :, :, :]
        output_sequence = output_sequence[:(2*self.num_input_images-1), :, :, :]
        target_sequence = target_sequence[:(2*self.num_input_images-1), :, :, :]

        # display the first sequence of the batch
        self.observe_sequence(input_sequence, output_sequence, target_sequence)

        return avg_PSNR


    def observe_sequence(self, input, output, target):
        num_outputs = output.size()[0]
        fig, axs = plt.subplots(3, num_outputs, sharex=True, sharey=True)
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
