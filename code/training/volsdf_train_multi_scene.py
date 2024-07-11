import os
from datetime import datetime
from pyhocon import ConfigFactory
import sys
import torch
from tqdm import tqdm

import utils.general as utils
import utils.plots as plt
from utils import rend_util

from torch.utils.tensorboard import SummaryWriter

class VolSDFTrainRunner():
    def __init__(self,**kwargs):
        torch.set_default_dtype(torch.float32)
        torch.set_num_threads(1)

        self.conf = ConfigFactory.parse_file(kwargs['conf'])
        self.batch_size = kwargs['batch_size']
        self.nepochs = kwargs['nepochs']
        self.exps_folder_name = kwargs['exps_folder_name']
        self.GPU_INDEX = kwargs['gpu_index']

        self.expname = self.conf.get_string('train.expname') + kwargs['expname']

        if kwargs['is_continue'] and kwargs['timestamp'] == 'latest':
            if os.path.exists(os.path.join('../',kwargs['exps_folder_name'],self.expname)):
                timestamps = os.listdir(os.path.join('../',kwargs['exps_folder_name'],self.expname))
                if (len(timestamps)) == 0:
                    is_continue = False
                    timestamp = None
                else:
                    timestamp = sorted(timestamps)[-1]
                    is_continue = True
            else:
                is_continue = False
                timestamp = None
        else:
            timestamp = kwargs['timestamp']
            is_continue = kwargs['is_continue']

        utils.mkdir_ifnotexists(os.path.join('../',self.exps_folder_name))
        self.expdir = os.path.join('../', self.exps_folder_name, self.expname)
        utils.mkdir_ifnotexists(self.expdir)
        self.timestamp = '{:%Y_%m_%d_%H_%M_%S}'.format(datetime.now())
        utils.mkdir_ifnotexists(os.path.join(self.expdir, self.timestamp))

        self.plots_dir = os.path.join(self.expdir, self.timestamp, 'plots')
        utils.mkdir_ifnotexists(self.plots_dir)

        # create checkpoints dirs
        self.checkpoints_path = os.path.join(self.expdir, self.timestamp, 'checkpoints')
        utils.mkdir_ifnotexists(self.checkpoints_path)
        self.model_params_subdir = "ModelParameters"
        self.optimizer_params_subdir = "OptimizerParameters"
        self.scheduler_params_subdir = "SchedulerParameters"
        self.latent_codes_subdir = "LatentCodes"

        utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.model_params_subdir))
        utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.optimizer_params_subdir))
        utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.scheduler_params_subdir))
        utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.latent_codes_subdir))

        os.system("""cp -r {0} "{1}" """.format(kwargs['conf'], os.path.join(self.expdir, self.timestamp, 'runconf.conf')))

        if (not self.GPU_INDEX == 'ignore'):
            os.environ["CUDA_VISIBLE_DEVICES"] = '{0}'.format(self.GPU_INDEX)

        print('shell command : {0}'.format(' '.join(sys.argv)))

        print('Loading data ...')

        dataset_conf = self.conf.get_config('dataset')
        self.scan_ids = dataset_conf.get_list('scan_ids')
        
        dataset_class = utils.get_class(self.conf.get_string('train.dataset_class'))
        self.train_dataset = dataset_class(**dataset_conf)

        self.ds_len = len(self.train_dataset)
        print('Finish loading data. Data-set size: {0}'.format(self.ds_len))

        # if len(self.scan_ids) == 3 and self.scan_ids[0] < 24 and self.scan_ids[-1] > 0:  # TODO: check if necessary
        #     self.nepochs = int(200000 / self.ds_len)
        #     print('RUNNING FOR {0} epochs'.format(self.nepochs))

        self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset,
                                                            batch_size=self.batch_size,
                                                            shuffle=True,
                                                            collate_fn=self.train_dataset.collate_fn
                                                            )
        self.plot_dataloader = torch.utils.data.DataLoader(self.train_dataset,
                                                           batch_size=self.conf.get_int('plot.plot_nimgs'),
                                                           shuffle=True,
                                                           collate_fn=self.train_dataset.collate_fn
                                                           )

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

        self.do_vis = kwargs['do_vis']

        self.start_epoch = 0
        if is_continue: # check if latent vectors are loaded as part of pararmeters when continued from checkpoint
            old_checkpnts_dir = os.path.join(self.expdir, timestamp, 'checkpoints')

            saved_model_state = torch.load(
                os.path.join(old_checkpnts_dir, 'ModelParameters', str(kwargs['checkpoint']) + ".pth"))
            self.model.load_state_dict(saved_model_state["model_state_dict"])
            self.start_epoch = saved_model_state['epoch']

            data = torch.load(
                os.path.join(old_checkpnts_dir, 'OptimizerParameters', str(kwargs['checkpoint']) + ".pth"))
            self.optimizer.load_state_dict(data["optimizer_state_dict"])

            data = torch.load(
                os.path.join(old_checkpnts_dir, self.scheduler_params_subdir, str(kwargs['checkpoint']) + ".pth"))
            self.scheduler.load_state_dict(data["scheduler_state_dict"])

        self.num_pixels = self.conf.get_int('train.num_pixels')
        self.total_pixels = self.train_dataset.total_pixels
        self.img_res = self.train_dataset.img_res
        self.n_batches = len(self.train_dataloader)
        self.plot_freq = self.conf.get_int('train.plot_freq')
        self.checkpoint_freq = self.conf.get_int('train.checkpoint_freq', default=100)
        self.split_n_pixels = self.conf.get_int('train.split_n_pixels', default=10000)
        self.plot_conf = self.conf.get_config('plot')

        self.writer = SummaryWriter(os.path.join(self.expdir, self.timestamp, 'logs'))

    def save_checkpoints(self, epoch):
        torch.save(
            {"epoch": epoch, "model_state_dict": self.model.state_dict()},
            os.path.join(self.checkpoints_path, self.model_params_subdir, str(epoch) + ".pth"))
        torch.save(
            {"epoch": epoch, "model_state_dict": self.model.state_dict()},
            os.path.join(self.checkpoints_path, self.model_params_subdir, "latest.pth"))

        torch.save(
            {"epoch": epoch, "optimizer_state_dict": self.optimizer.state_dict()},
            os.path.join(self.checkpoints_path, self.optimizer_params_subdir, str(epoch) + ".pth"))
        torch.save(
            {"epoch": epoch, "optimizer_state_dict": self.optimizer.state_dict()},
            os.path.join(self.checkpoints_path, self.optimizer_params_subdir, "latest.pth"))

        torch.save(
            {"epoch": epoch, "scheduler_state_dict": self.scheduler.state_dict()},
            os.path.join(self.checkpoints_path, self.scheduler_params_subdir, str(epoch) + ".pth"))
        torch.save(
            {"epoch": epoch, "scheduler_state_dict": self.scheduler.state_dict()},
            os.path.join(self.checkpoints_path, self.scheduler_params_subdir, "latest.pth"))

        torch.save(
            {"epoch": epoch, "latent_codes": self.model.implicit_network.z},
            os.path.join(self.checkpoints_path, self.latent_codes_subdir, str(epoch) + ".pth"))
        torch.save(
            {"epoch": epoch, "latent_codes": self.model.implicit_network.z},
            os.path.join(self.checkpoints_path, self.latent_codes_subdir, "latest.pth"))
        
    def log_gradients(self, epoch, data_index):
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                self.writer.add_histogram(f'gradients/{name}', param.grad, epoch * self.n_batches + data_index)

    def run(self):
        print("training...")

        for epoch in range(self.start_epoch, self.nepochs + 1):

            if epoch % self.checkpoint_freq == 0:
                self.save_checkpoints(epoch)

            if self.do_vis and epoch % self.plot_freq == 0 and epoch > 0:
                self.model.eval()

                self.train_dataset.change_sampling_idx(-1)
                indices, model_input, ground_truth = next(iter(self.plot_dataloader))

                model_input["intrinsics"] = model_input["intrinsics"].cuda()
                model_input["uv"] = model_input["uv"].cuda()
                model_input['pose'] = model_input['pose'].cuda()
                model_input['scene_idx'] = model_input['scene_idx'].cuda() # TODO: check if idx or true id is required. See multi_scene_dataset.py for details

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
                         plot_data,
                         self.plots_dir,
                         epoch,
                         self.img_res,
                         **self.plot_conf
                         )

                self.model.train()

            self.train_dataset.change_sampling_idx(self.num_pixels)

            running_loss = 0
            for data_index, (indices, model_input, ground_truth) in enumerate(self.train_dataloader):
                model_input["intrinsics"] = model_input["intrinsics"].cuda()
                model_input["uv"] = model_input["uv"].cuda()
                model_input['pose'] = model_input['pose'].cuda()
                model_input['scene_idx'] = model_input['scene_idx'].cuda() # TODO: check if idx or true id is required. See multi_scene_dataset.py for details

                model_outputs = self.model(model_input)

                loss_output = self.loss(model_outputs, ground_truth)

                loss = loss_output['loss']
                
                running_loss += loss.item()

                self.optimizer.zero_grad()
                loss.backward()

                self.log_gradients(epoch, data_index)

                self.optimizer.step()

                psnr = rend_util.get_psnr(model_outputs['rgb_values'],
                                          ground_truth['rgb'].cuda().reshape(-1,3))

                if data_index == 0:
                    print(
                        '{0}_{1} [{2}] ({3}/{4}): loss = {5}, rgb_loss = {6}, eikonal_loss = {7}, psnr = {8}'
                            .format(self.expname, self.timestamp, epoch, data_index, self.n_batches, loss.item(),
                                    loss_output['rgb_loss'].item(),
                                    loss_output['eikonal_loss'].item(),
                                    psnr.item()))

                self.train_dataset.change_sampling_idx(self.num_pixels)
                self.scheduler.step()

            self.writer.add_scalar('train_loss', running_loss / len(self.train_dataloader), global_step=epoch)
    
        self.writer.close()
        self.save_checkpoints(epoch)

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