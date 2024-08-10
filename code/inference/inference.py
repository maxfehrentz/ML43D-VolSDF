import sys
from datetime import datetime
sys.path.append('../code')
import argparse
import GPUtil
import os
from PIL import Image
from pyhocon import ConfigFactory
import torch
from tqdm import tqdm
import numpy as np
import pandas as pd

import utils.general as utils
import utils.plots as plt
from utils import rend_util

class VolSDFInferenceRunner:
    def __init__(self, **kwargs):
        torch.set_default_dtype(torch.float32)
        torch.set_num_threads(1)

        self.conf = ConfigFactory.parse_file(kwargs['conf'])
        self.batch_size = kwargs['batch_size']
        self.nepochs = kwargs['nepochs']
        self.exps_folder_name = kwargs['exps_folder_name']
        self.GPU_INDEX = kwargs['gpu_index']

        self.expname = self.conf.get_string('train.expname')
        self.checkpoint = kwargs['checkpoint']

        self.modeldir = self.conf.get_string('model.modeldir')

        utils.mkdir_ifnotexists(os.path.join('../', self.exps_folder_name, self.expname))
        self.expdir = os.path.join('../', self.exps_folder_name, self.expname)
        self.timestamp = '{:%Y_%m_%d_%H_%M_%S}'.format(datetime.now())
        utils.mkdir_ifnotexists(os.path.join(self.expdir, self.timestamp))

        self.plots_dir = os.path.join(self.expdir, self.timestamp, 'inference_plots')
        utils.mkdir_ifnotexists(self.plots_dir)

        print(os.path.join('../', self.exps_folder_name, self.expname))
        
        # Set up GPU
        if not self.GPU_INDEX == 'ignore':
            os.environ["CUDA_VISIBLE_DEVICES"] = '{0}'.format(self.GPU_INDEX)

        print('shell command : {0}'.format(' '.join(sys.argv)))

        print('Loading data ...')

        # Laod train dataset
        train_dataset_conf = self.conf.get_config('dataset')
        self.scan_ids = train_dataset_conf.get_list('scan_ids')

        self.train_dataset = utils.get_class(self.conf.get_string('train.dataset_class'))(**train_dataset_conf)

        if len(self.scan_ids) == 0:
            self.scan_ids = self.train_dataset.scan_ids

        self.ds_len = len(self.train_dataset)
        print('Finish loading train data. Data-set size: {0}'.format(self.ds_len))

        self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset,
                                                        #    batch_size=self.conf.get_int('plot.plot_nimgs'),
                                                           batch_size=self.batch_size,
                                                           shuffle=True,
                                                           collate_fn=self.train_dataset.collate_fn)
        
        self.eval_train_dataloader = torch.utils.data.DataLoader(self.train_dataset,
                                                    batch_size=1,
                                                    shuffle=False,
                                                    collate_fn=self.train_dataset.collate_fn)

        # Load validation dataset
        val_dataset_conf = self.conf.get_config('val_dataset')
        self.val_dataset = utils.get_class(self.conf.get_string('val.dataset_class'))(**val_dataset_conf)

        self.val_dataloader = torch.utils.data.DataLoader(self.val_dataset,
                                                           batch_size=self.conf.get_int('plot.plot_nimgs'),
                                                           shuffle=True,
                                                           collate_fn=self.val_dataset.collate_fn)

        self.eval_val_dataloader = torch.utils.data.DataLoader(self.val_dataset,
                                                    batch_size=1,
                                                    shuffle=False,
                                                    collate_fn=self.val_dataset.collate_fn)

        # Load model
        conf_model = self.conf.get_config('model')
        conf_model['num_scenes'] = len(self.scan_ids)
        self.model = utils.get_class(self.conf.get_string('train.model_class'))(conf=conf_model)
        if torch.cuda.is_available():
            self.model.cuda()

        self.loss = utils.get_class(self.conf.get_string('train.loss_class'))(**self.conf.get_config('loss'))

        self.lr = self.conf.get_float('train.learning_rate')
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.sched_decay_rate = self.conf.get_float('train.sched_decay_rate', default=0.1)
        self.sched_decay_steps = self.nepochs * len(self.train_dataset)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, self.sched_decay_rate ** (1./self.sched_decay_steps))

        # Inference stuff
        self.lr_inference = self.conf.get_float('val.learning_rate') # used for latent optimization during inference
        self.optim_steps_inference = self.conf.get_int('val.optim_steps_inference') # number of optimization steps for latent optimization during inference
        # Create optimizer for inference optimization of the latent
        self.optimizer_latent = torch.optim.Adam(self.model.parameters(), lr=self.lr_inference)
        # TODO: might add scheduler similar to training here later for the inference optimization of the latent

        self.load_checkpoint()

        self.num_pixels = self.conf.get_int('train.num_pixels')
        self.total_pixels = self.train_dataset.total_pixels
        self.img_res = self.train_dataset.img_res
        self.n_batches = len(self.train_dataloader)
        self.plot_freq = self.conf.get_int('train.plot_freq')
        self.checkpoint_freq = self.conf.get_int('train.checkpoint_freq', default=100)
        self.split_n_pixels = self.conf.get_int('train.split_n_pixels', default=10000)
        self.plot_conf = self.conf.get_config('plot')

        # Inference stuff
        self.num_pixels_inference = self.conf.get_int('val.num_pixels')

    def load_checkpoint(self):
        checkpoint_path = os.path.join(self.modeldir, 'checkpoints', 'ModelParameters', f"{self.checkpoint}.pth")
        print(f"Loading checkpoint from {checkpoint_path}")
        
        saved_model_state = torch.load(checkpoint_path)
        self.model.load_state_dict(saved_model_state["model_state_dict"])
        self.start_epoch = saved_model_state['epoch']
        print(f"Loaded checkpoint from epoch {self.start_epoch}")


    def get_latent_codes(self):
        """Load and return the latent codes which were optimized during training."""
        latent_path = os.path.join(self.modeldir, 'checkpoints', 'LatentCodes', f"{self.checkpoint}.pth")
        print(f"Loading latent codes from {latent_path}")

        latent_codes = torch.load(latent_path)["latent_codes"]
        self.latent_codes = latent_codes


    def inference_from_latent(self, model_input, ground_truth, indices):
        self.model.eval()

        latent_idx = model_input['scene_idx']
        z = self.latent_codes[latent_idx]
        self.model.implicit_network.z_inference.data = z

        model_outputs = self.model(model_input)
        psnr = rend_util.get_psnr(model_outputs['rgb_values'], ground_truth['rgb'].cuda().reshape(-1,3))

        print(f"PSNR: {psnr.item():.2f}")

        self.visualize(model_input, ground_truth, -1, indices)


    def interpolate(self, model_input, ground_truth, indices):
        self.model.eval()
        num_interpolation_steps = 4

        latent_idx_1 = model_input['scene_idx']
        latent_codes = [self.latent_codes[latent_idx_1], self.latent_codes[latent_idx_1 + 1]]

        for i in range(0, num_interpolation_steps + 1):
          interpolated_code = latent_codes[0] * (1 - i / num_interpolation_steps) + latent_codes[1] * (i / num_interpolation_steps)
          self.model.implicit_network.z_inference.data = interpolated_code

          self.visualize(model_input, ground_truth, i, indices)

    
    def inference_and_visualization(self, model_input, ground_truth, indices):
        self.model.eval()
        # Will be used for latent optimization, only a subset of the data due to memory constraints
        # Draw a subset of the pixels for latent optimization
        random_indices = torch.randperm(self.total_pixels)[:self.num_pixels_inference]

        model_input_subset = {
            "intrinsics": model_input["intrinsics"],
            "uv": model_input["uv"][:, random_indices, :],
            "pose": model_input['pose'],
            "scene_idx": model_input['scene_idx']
        }
        ground_truth_subset = {
            "rgb": ground_truth["rgb"][:, random_indices, :]
        }

        # Initialize the latent
        self.model.implicit_network.init_latent()

        # Freeze everything but latent
        self.model.requires_grad = False
        self.model.implicit_network.z_inference.requires_grad = True


        for i in range(self.optim_steps_inference):
            # Only taking subset of model_input for latent optimization due to memory constraints
            model_outputs = self.model(model_input_subset)

            loss_output = self.loss(model_outputs, ground_truth_subset)
            loss = loss_output['loss']

            self.optimizer_latent.zero_grad()
            loss.backward()

            # print(self.model.implicit_network.z_inference.data)
            # print(f"grads: {self.model.implicit_network.z_inference.grad}")

            self.optimizer_latent.step()

            print(f"latent optimization step {i}:"
                f"loss = {loss.item()}, "
                f"rgb_loss = {loss_output['rgb_loss'].item()}, "
                f"eikonal_loss = {loss_output['eikonal_loss'].item()}")

            if i == 0 or i == self.optim_steps_inference / 2 or i == self.optim_steps_inference - 1:
                # TODO: make sure this does not mess up with the compute graph or gradients, cannot call with
                #  torch.no_grad() as visualizing actually requires the gradients
                self.visualize(model_input, ground_truth, i, indices)


    def visualize(self, model_input, ground_truth, stage, indices):
        split = utils.split_input(model_input, self.total_pixels, n_pixels=self.split_n_pixels)
        res = []
        for s in tqdm(split):
            out = self.model(s)
            d = {'rgb_values': out['rgb_values'].detach(),
                 'normal_map': out['normal_map'].detach()}
            res.append(d)

        batch_size = ground_truth['rgb'].shape[0]
        model_outputs = utils.merge_output(res, self.total_pixels, batch_size)
        plot_data = self.get_plot_data(model_outputs, model_input['pose'], ground_truth['rgb'])

        plt.plot(self.model.implicit_network,
                 indices,
                 model_input['scene_idx'],
                 plot_data,
                 self.plots_dir,
                 f"inference_{stage}",
                 self.img_res,
                 **self.plot_conf)


    def get_plot_data(self, model_outputs, pose, rgb_gt):
        batch_size, num_samples, _ = rgb_gt.shape

        rgb_eval = model_outputs['rgb_values'].reshape(batch_size, num_samples, 3)
        normal_map = model_outputs['normal_map'].reshape(batch_size, num_samples, 3)
        normal_map = (normal_map + 1.) / 2.

        plot_data = {
            'rgb_gt': rgb_gt,
            'pose': pose,
            'rgb_eval': rgb_eval,
            'normal_map': normal_map,
        }

        return plot_data


    def unsquzee_and_load_for_inference(self, indices, model_input, ground_truth):
        if not isinstance(indices, torch.Tensor):
            indices = torch.tensor(indices).unsqueeze(0)

        if ground_truth["rgb"].shape[0] != 1:
            ground_truth["rgb"] = ground_truth["rgb"].unsqueeze(0)

        for key, value in model_input.items():
            if torch.is_tensor(value):
                if value.dim() == 0 or value.shape[0] != 1:
                    model_input[key] = model_input[key].unsqueeze(0)

        model_input["intrinsics"] = model_input["intrinsics"].cuda()
        model_input["uv"] = model_input["uv"].cuda()
        model_input['pose'] = model_input['pose'].cuda()
        model_input['scene_idx'] = model_input['scene_idx'].cuda()

        return indices, model_input, ground_truth


    def eval_generalization(self):
        self.model.eval()
        file_path = 'eval.txt'

        self.evaldir = os.path.join('../', 'evaluate_generalization')
        utils.mkdir_ifnotexists(self.evaldir)

        self.eval_expdir = os.path.join(self.evaldir, self.expname)
        utils.mkdir_ifnotexists(self.eval_expdir)
        
        self.eval_timestamp = '{:%Y_%m_%d_%H_%M_%S}'.format(datetime.now())
        utils.mkdir_ifnotexists(os.path.join(self.eval_expdir, self.eval_timestamp))

        self.eval_images_dir = os.path.join(self.eval_expdir, self.eval_timestamp, 'eval_images')
        utils.mkdir_ifnotexists(self.eval_images_dir)

        psnrs = []
        for data_index, (indices, model_input, ground_truth) in enumerate(self.eval_val_dataloader):
                print(f"Processing data index {data_index}")
                model_input["intrinsics"] = model_input["intrinsics"].cuda()
                model_input["uv"] = model_input["uv"].cuda()
                model_input['pose'] = model_input['pose'].cuda()
                model_input['scene_idx'] = model_input['scene_idx'].cuda()

                # Initialize the latent
                self.model.implicit_network.init_latent()

                # Freeze everything but latent
                self.model.requires_grad = False
                self.model.implicit_network.z_inference.requires_grad = True


                for i in range(self.optim_steps_inference):
                    model_outputs = self.model(model_input)

                    loss_output = self.loss(model_outputs, ground_truth)
                    loss = loss_output['loss']

                    self.optimizer_latent.zero_grad()
                    loss.backward()

                    self.optimizer_latent.step()

                    print(f"latent optimization step {i}:"
                        f"loss = {loss.item()}, "
                        f"rgb_loss = {loss_output['rgb_loss'].item()}, "
                        f"eikonal_loss = {loss_output['eikonal_loss'].item()}")

                    if i == 0 or i == self.optim_steps_inference //3 or i == 2 * (self.optim_steps_inference // 3) or i == self.optim_steps_inference - 1:
                        # TODO: make sure this does not mess up with the compute graph or gradients, cannot call with
                        #  torch.no_grad() as visualizing actually requires the gradients
                        print("visualizing")
                        self.visualize(model_input, ground_truth, i, indices)

                batch_size = ground_truth['rgb'].shape[0]
                rgb_eval = model_outputs['rgb_values']
                rgb_eval = rgb_eval.reshape(batch_size, self.total_pixels, 3)

                rgb_eval = plt.lin2img(rgb_eval, self.img_res).detach().cpu().numpy()[0]
                rgb_eval = rgb_eval.transpose(1, 2, 0)

                img = Image.fromarray((rgb_eval * 255).astype(np.uint8))
                img.save('{0}/eval_{1}.png'.format(self.eval_images_dir,'%03d' % indices[0]))

                psnr = rend_util.get_psnr(model_outputs['rgb_values'], ground_truth['rgb'].cuda().reshape(-1, 3)).item()
                print(f"PSNR: {psnr:.2f}")
                psnrs.append(psnr)
                with open(file_path, 'a') as file:
                    formatted_string = f"PSNR for {data_index}: {psnr:.2f}"
                    file.write(formatted_string + '\n')

                
                print("visualizing")
                self.visualize(model_input, ground_truth, data_index, indices)

        psnrs = np.array(psnrs).astype(np.float64)
        print("RENDERING EVALUATION: psnr mean = {0} ; psnr std = {1}".format("%.2f" % psnrs.mean(), "%.2f" % psnrs.std()))
        file_path = 'eval.txt'
        with open(file_path, 'a') as file:
            formatted_string = "RENDERING EVALUATION val 10: psnr mean = {0} ; psnr std = {1}".format("%.2f" % psnrs.mean(), "%.2f" % psnrs.std())
            file.write(formatted_string + '\n')
        psnrs = np.concatenate([psnrs, psnrs.mean()[None], psnrs.std()[None]])
        pd.DataFrame(psnrs).to_csv('{0}/psnr.csv'.format(self.eval_expdir))


            


    def eval_new_model_3_scene(self):
        self.model.eval()

        self.evaldir = os.path.join('../', 'evaluation')
        utils.mkdir_ifnotexists(self.evaldir)

        self.eval_expdir = os.path.join(self.evaldir, self.expname)
        utils.mkdir_ifnotexists(self.eval_expdir)
        
        self.eval_timestamp = '{:%Y_%m_%d_%H_%M_%S}'.format(datetime.now())
        utils.mkdir_ifnotexists(os.path.join(self.eval_expdir, self.eval_timestamp))

        self.eval_images_dir = os.path.join(self.eval_expdir, self.eval_timestamp, 'eval_images')
        utils.mkdir_ifnotexists(self.eval_images_dir)

        # the order of latent codes seems to be defined by the order in which secenes we loaded during training. 
        # need to fix that to allow some mapping to true scene ids. the easiest way is probably to store the list of scan_ids generated during training

        latent2scene_map = {0: 0, 1: 1, 2: 2}

        psnrs = []
        for data_index, (indices, model_input, ground_truth) in enumerate(self.eval_train_dataloader):

            model_input["intrinsics"] = model_input["intrinsics"].cuda()
            model_input["uv"] = model_input["uv"].cuda()
            model_input['pose'] = model_input['pose'].cuda()
            model_input['scene_idx'] = model_input['scene_idx'].cuda()

            latent_idx = latent2scene_map.get(int(model_input['scene_idx']))
            z = self.latent_codes[(latent_idx)]
            self.model.implicit_network.z_inference.data = z

            split = utils.split_input(model_input, self.total_pixels, n_pixels=self.split_n_pixels)
            res = []
            for s in tqdm(split):
                torch.cuda.empty_cache()
                out = self.model(s)
                res.append({
                    'rgb_values': out['rgb_values'].detach(),
                })

            batch_size = ground_truth['rgb'].shape[0]
            model_outputs = utils.merge_output(res, self.total_pixels, batch_size)
            rgb_eval = model_outputs['rgb_values']
            rgb_eval = rgb_eval.reshape(batch_size, self.total_pixels, 3)

            rgb_eval = plt.lin2img(rgb_eval, self.img_res).detach().cpu().numpy()[0]
            rgb_eval = rgb_eval.transpose(1, 2, 0)

            img = Image.fromarray((rgb_eval * 255).astype(np.uint8))
            img.save('{0}/eval_{1}.png'.format(self.eval_images_dir,'%03d' % indices[0]))

            psnr = rend_util.get_psnr(model_outputs['rgb_values'], ground_truth['rgb'].cuda().reshape(-1, 3)).item()
            print(f"PSNR: {psnr:.2f}")
            psnrs.append(psnr)
            
            if data_index % 24 == 0:
                print("visualizing")
                self.visualize(model_input, ground_truth, data_index, indices)

        psnrs = np.array(psnrs).astype(np.float64)
        print("RENDERING EVALUATION: psnr mean = {0} ; psnr std = {1}".format("%.2f" % psnrs.mean(), "%.2f" % psnrs.std()))
        file_path = 'eval.txt'
        with open(file_path, 'a') as file:
            formatted_string = "RENDERING EVALUATION: psnr mean = {0} ; psnr std = {1}".format("%.2f" % psnrs.mean(), "%.2f" % psnrs.std())
            file.write(formatted_string + '\n')
        psnrs = np.concatenate([psnrs, psnrs.mean()[None], psnrs.std()[None]])
        pd.DataFrame(psnrs).to_csv('{0}/psnr.csv'.format(self.eval_expdir))




        
    def run_inference(self):
        self.get_latent_codes()

        # ask whether to run iference or evaluation
        to_run = input("Do you want to run inference or evaluation? (i/e): ")
        if to_run == 'e':
            eval_infer_or_generalization = input("Do you want to run evaluation or generalization? (e/g): ")
            if eval_infer_or_generalization == 'e':
                self.eval_new_model_3_scene()
            elif eval_infer_or_generalization == 'g':
                self.eval_generalization()
            return
        elif to_run == 'i':

            print("Running inference...")

            self.train_dataset.change_sampling_idx(-1)

            # it is a gloabal index meaning that with a scene having 24 samples, the frist 24 indices are for the first scene from different views and so on
            # index_to_infer = input("Enter the index of the sample to infer: ")
            # indices, model_input, ground_truth = self.train_dataset[int(index_to_infer)]
            indices, model_input, ground_truth = next(iter(self.train_dataloader))
            

            indices, model_input, ground_truth = self.unsquzee_and_load_for_inference(indices, model_input, ground_truth)
            self.get_latent_codes()

            # ask if to run inference on a trained latent
            run_inference_on_latent = input("Do you want to run inference on the trained latent? (y/n): ")
            if run_inference_on_latent == 'y':
                #run inference on a trained latent
                self.inference_from_latent(model_input, ground_truth, indices)
            
            # ask if to run interpolation between two trained latents
            run_interpolation = input("Do you want to run interpolation between two trained latents? (y/n): ")
            if run_interpolation == 'y':
                # run interpolation between two trained latents
                self.interpolate(model_input, ground_truth, indices)
            
            # ask if to run gereration from random latent and optimization to unseen sample
            run_inference_and_visualization = input("Do you want to run gereration from random latent and optimization to unseen sample? (y/n): ")
            if run_inference_and_visualization == 'y':

                self.val_dataset.change_sampling_idx(-1)

                indices, model_input, ground_truth = next(iter(self.val_dataloader))
                indices, model_input, ground_truth = self.unsquzee_and_load_for_inference(indices, model_input, ground_truth)
                
                # run gereration from random latent and optimization to unseen sample
                self.inference_and_visualization(model_input, ground_truth, indices)


            print("Inference completed.")

if __name__ == '__main__':

    parser = argparse.ArgumentParser() # TODO: refactor, delete configs not used in inference
    parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
    parser.add_argument('--nepoch', type=int, default=2000, help='number of epochs to train for')
    parser.add_argument('--conf', type=str, default='./confs/dtu.conf')
    parser.add_argument('--expname', type=str, default='')
    parser.add_argument("--exps_folder", type=str, default="exps")
    parser.add_argument('--gpu', type=str, default='auto', help='GPU to use [default: GPU auto]')
    parser.add_argument('--is_continue', default=False, action="store_true",
                        help='If set, indicates continuing from a previous run.')
    parser.add_argument('--timestamp', default='latest', type=str,
                        help='The timestamp of the run to be used in case of continuing from a previous run.')
    parser.add_argument('--checkpoint', default='latest', type=str,
                        help='Checkpoint to use for inference')
    parser.add_argument('--scan_id', type=int, default=-1, help='If set, taken to be the scan id.')
    parser.add_argument('--cancel_vis', default=False, action="store_true",
                        help='If set, cancel visualization in intermediate epochs.')

    opt = parser.parse_args()

    if opt.gpu == "auto":
        deviceIDs = GPUtil.getAvailable(order='memory', limit=1, maxLoad=0.5, maxMemory=0.5, includeNan=False,
                                        excludeID=[], excludeUUID=[])
        gpu = deviceIDs[0]
    else:
        gpu = opt.gpu



    inference_runner = VolSDFInferenceRunner(conf=opt.conf,
                                    batch_size=opt.batch_size,
                                    nepochs=opt.nepoch,
                                    expname=opt.expname,
                                    gpu_index=gpu,
                                    exps_folder_name=opt.exps_folder,
                                    is_continue=opt.is_continue,
                                    timestamp=opt.timestamp,
                                    checkpoint=opt.checkpoint,
                                    scan_id=opt.scan_id,
                                    )

    inference_runner.run_inference()