import torch, os, sys, time, skimage, lpips, cv2, shutil, importlib
import numpy as np
import matplotlib.cm as cm
from collections import defaultdict
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

sys.path.append("/code/EGVD/data")
from dataset_ep2 import *
from dataset_e_test import *

### DDP
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

## 杂货铺
if True:

    class Tee(object):
        def __init__(self, filename):
            self.file = open(filename, 'a', buffering=1)
            self.terminal = sys.stdout

        def __del__(self):
            self.file.close()

        def write(self, data):
            self.file.write(data)
            self.terminal.write(data)

        def flush(self):
            self.file.flush()
    
    class AccumDict:
        def __init__(self, num_f=3):
            self.d = defaultdict(list)
            self.num_f = num_f
            
        def add(self, k, v):
            self.d[k] += [v]
            
        def __dict__(self):
            return self.d

        def __getitem__(self, key):
            return self.d[key]
        
        def __str__(self):
            s = ''
            for k in self.d:
                if not self.d[k]:
                    continue
                cur = self.d[k][-1]
                avg = np.mean(self.d[k])
                format_str = '{:.%df}' % self.num_f
                cur_str = format_str.format(cur)
                avg_str = format_str.format(avg)
                s += f'{k} {cur_str} ({avg_str})\t\t'
            return s
        
        def __repr__(self):
            return self.__str__()
    
    class TicToc:
        def __init__(self):
            self.tic_toc_tic = None

        def tic(self):
            self.tic_toc_tic = time.time()

        def toc(self):
            assert self.tic_toc_tic, 'You forgot to call tic()'
            return (time.time() - self.tic_toc_tic) * 1000

        def tocp(self, str):
            print(f"{str} took {self.toc():.4f}ms")

        @staticmethod
        def print_timing(timing, name=''):
            print(f'\n=== {name} Timimg ===')
            for fn, times in timing.items():
                min, max, mean, p95 = np.min(times), np.max(times), np.mean(times), np.percentile(times, 95)
                print(f'{fn}:\tmin: {min:.4f}\tmax: {max:.4f}\tmean: {mean:.4f}ms\tp95: {p95:.4f}ms')
    
    def num_param(model):
        return sum([p.numel() for p in model.parameters()])

    def makedir(directory, mode="train"):

        if mode=="train":
            if not os.path.exists(directory):
                os.makedirs(directory)
            else:
                shutil.rmtree(directory)
                os.makedirs(directory)
        else:
            if not os.path.exists(directory):
                os.makedirs(directory)
    
    def setup_logging(save_dir, mode="train"):

        if mode == "train":
            tee = Tee(f"{save_dir}/log.txt")
            sys.stdout, sys.stderr = tee, tee
        else:
            if os.path.exists(f"{save_dir}/test_log.txt"):
                os.remove(f"{save_dir}/test_log.txt")

            tee = Tee(f"{save_dir}/test_log.txt")
            sys.stdout, sys.stderr = tee, tee
    
    class HiddenPrints:
        def __enter__(self):
            # 保存原始的标准输出和标准错误流
            self._original_stdout = sys.stdout
            self._original_stderr = sys.stderr

            # 将标准输出和标准错误重定向到空设备（/dev/null）
            sys.stdout = open('/dev/null', 'w')
            sys.stderr = open('/dev/null', 'w')

        def __exit__(self, exc_type, exc_val, exc_tb):
            # 恢复原始的标准输出和标准错误流
            sys.stdout.close()
            sys.stdout = self._original_stdout
            sys.stderr.close()
            sys.stderr = self._original_stderr


    def get_metrics(pred, gt, use_lpips=False, loss_fn_vgg=None, device=None):

        '''
        Compute image metrics 

        Args:
            pred (np array): (H, W, 3)
            gt (np array): (H, W, 3)
        '''
        pred = pred.astype(np.float32) / 255.0
        gt = gt.astype(np.float32) / 255.0
        
        ssim = skimage.metrics.structural_similarity(pred, gt, channel_axis = -1, data_range=1)
        psnr = skimage.metrics.peak_signal_noise_ratio(pred, gt, data_range=1)

        if use_lpips:
            if loss_fn_vgg is None:
                loss_fn_vgg = lpips.LPIPS(net='vgg')
            gt_ = torch.from_numpy(gt).permute(2,0,1).unsqueeze(0)*2 - 1.0
            pred_ = torch.from_numpy(pred).permute(2,0,1).unsqueeze(0)*2 - 1.0
            if device is not None:
                loss_fn_vgg.to(device)
                lp = loss_fn_vgg(gt_.to(device), pred_.to(device)).detach().cpu().numpy().item()
            else:
                lp = loss_fn_vgg(gt_, pred_).detach().numpy().item()
            return ssim, psnr, lp
        return ssim, psnr


class trainer:

    def __init__(self, args):

        self.args = args

    def train(self, local_rank, nprocs):
        sys.path.append("/code/EGVD/network")
        baseline = importlib.import_module(self.args.baseline_type)
        EAVD = getattr(baseline, "EAVD")
        
        print("*"*80)
        print("Begin training")
        print("*"*80)

        lowest_loss = 1e+10
        lowest_train_loss = 1e+10

        self.local_rank = local_rank
        self.nprocs = nprocs

        torch.cuda.set_device(self.local_rank)
        dist.init_process_group(backend="nccl")

        if dist.get_rank() == 0:

            save_dir = self.args.save_dir
            save_args = ["train", "eval", "checkpoint"]
            makedir(save_dir)

            for save_item in save_args:
                makedir(f"{save_dir}/{save_item}")
        
            writer = SummaryWriter(logdir=self.args.log_dir, flush_secs=10)
            setup_logging(save_dir)
            print(f"Now saving data in {save_dir}")
        
        self.model = EAVD(self.args, self.local_rank)

        if dist.get_rank() == 0:
            self.count_network_parameters()

        train_log_iter = 0
        train_step_iter = 0

        val_log_iter = 0
        val_step_iter = 0

        for epoch in range(self.args.epochs):

            self.model.train()
            self.current_epoch = epoch

            ### TRAIN
            if True:

                ad = AccumDict()
                tt = TicToc()

                for scene_idx in range(self.args.train_scene_num):
                    
                    train_dataset = eval(self.args.train_dataset)(self.args.root_dir, self.args.data_type, "train", scene_idx, self.args.sequence_length, \
                                                    num_bins = self.args.num_bins, dataset_type=self.args.train_dataset_type, opts = self.args)
                    train_sampler = DistributedSampler(train_dataset)
                    train_dataloader = DataLoader(dataset = train_dataset, batch_size = self.args.batch_size, shuffle = (train_sampler is None), sampler = train_sampler, num_workers = 4, pin_memory = True)

                    max_it = np.sum([len(train_dataset)]) // self.args.batch_size // self.nprocs         

                    train_dataloader.sampler.set_epoch(self.current_epoch)

                    for iteration, item in enumerate(train_dataloader):

                        train_step_iter += 1

                        tt.tic()

                        torch.cuda.empty_cache()

                        self.model.set_input(item)
                        self.model.forward()
                        loss, metrics = self.model.get_losses()

                        if dist.get_rank() == 0:

                            ad.add("loss_sum", loss["sum"])
                            ad.add("psnr", metrics["psnr"])
                            ad.add("ssim", metrics["ssim"])

                            ad.add("batch_time", tt.toc)

                            writer.add_scalar(f"TRAIN/loss_sum", ad["loss_sum"][-1], train_step_iter)
                            writer.add_scalar(f"TRAIN/psnr", ad["psnr"][-1], train_step_iter)
                            writer.add_scalar(f"TRAIN/ssim", ad["ssim"][-1], train_step_iter)
                            writer.add_scalar(f"lr", self.model.optimizer.param_groups[0]["lr"], train_step_iter)

                            if (train_step_iter + 1) % (self.args.log_freq) == 0:

                                train_log_iter += 1

                                # to do
                                s = f"TRAIN: [{epoch}][{scene_idx}][{train_step_iter}/{max_it-1}]\t"
                                a,b,c = np.mean(ad["loss_sum"]), np.mean(ad["psnr"]), np.mean(ad["ssim"])
                                str_A = f"Loss:{a:03f}  PSNR:{b:03f}  SSIM:{c:03f}"
                                s += str_A
                                print(s)
                                # s += f"Loss:{np.mean(ad["loss_sum"])}/PSNR:{np.mean(ad["psnr"])}/SSIM:{np.mean(ad["ssim"])}"
                                # s += str(ad)

                                tmp = np.hstack((
                                    cv2.cvtColor(self.model.vis_input, cv2.COLOR_BGR2RGB),
                                    cv2.cvtColor(self.model.vis_out, cv2.COLOR_BGR2RGB),
                                    cv2.cvtColor(self.model.vis_gt, cv2.COLOR_BGR2RGB),
                                    cv2.cvtColor(self.model.vis_event_left, cv2.COLOR_BGR2RGB),
                                    cv2.cvtColor(self.model.vis_event_right, cv2.COLOR_BGR2RGB),
                                ))

                                cv2.imwrite(f"{save_dir}/train/{(train_log_iter%20):05d}.png", tmp)
                
                ### adjuct learning rate, todo此时是在一个epoch之后
                self.model.update_learning_rate()

                if dist.get_rank() == 0:

                    train_avg_loss = np.mean(ad["loss_sum"])
                    train_avg_psnr = np.mean(ad["psnr"])
                    train_avg_ssim = np.mean(ad["ssim"])

                    print("TRAIN LOSS", train_avg_loss)
                    print("TRAIN PSNR", train_avg_psnr, "TRAIN SSIM", train_avg_ssim)
                    print("net_lr", self.model.optimizer.param_groups[0]["lr"])

                    if train_avg_loss < lowest_train_loss:

                        lowest_train_loss = train_avg_loss

                        checkpint_name = f"{save_dir}/checkpoint/train_best.pth.tar"
                        self.save_checkpoint(checkpint_name)

            ### VAL
            if True:

                if (self.current_epoch+1) >= self.args.test_threshold \
                    and (self.current_epoch+1) % self.args.save_freq == 0:

                    if dist.get_rank() == 0:

                        print('Begin Validating ...')

                        self.model.eval()

                        Avg_ad = AccumDict()

                        with torch.no_grad():

                            scene_types = self.args.scene_types

                            for current_type in scene_types:

                                ad = AccumDict()
                                tt = TicToc()

                                val_dataset = eval(self.args.test_dataset)(self.args.test_h5_file, self.args.test_sequence_length, current_type, data_type = self.args.test_data_type)
                                val_data_loader = DataLoader(dataset=val_dataset, batch_size=self.args.batch_size_val, shuffle=False, num_workers = 4, pin_memory = True, drop_last=True)
                                
                                max_it = np.sum([len(val_dataset)]) // self.args.batch_size_val
                                lstm_state = None
                                for iteration, item in enumerate(val_data_loader):

                                    tt.tic()

                                    val_step_iter += 1

                                    self.model.set_input_test(item)
                                    lstm_state = self.model.forward_test(lstm_state)
                                    loss, metrics = self.model.get_losses()
                                    
                                    ad.add("loss_sum", loss["sum"])
                                    ad.add("psnr", metrics["psnr"])
                                    ad.add("ssim", metrics["ssim"])

                                    tmp = np.hstack((
                                    cv2.cvtColor(self.model.vis_input, cv2.COLOR_BGR2RGB),
                                    cv2.cvtColor(self.model.vis_out, cv2.COLOR_BGR2RGB),
                                    cv2.cvtColor(self.model.vis_gt, cv2.COLOR_BGR2RGB),
                                    cv2.cvtColor(self.model.vis_event_left, cv2.COLOR_BGR2RGB),
                                    cv2.cvtColor(self.model.vis_event_right, cv2.COLOR_BGR2RGB),
                                        ))

                                    cv2.imwrite(f"{save_dir}/eval/{val_step_iter%20:05d}.png", tmp)

                                    if (val_step_iter + 1) % (self.args.val_log_freq) == 0:

                                        s = f"VAL: [{epoch}][{val_step_iter}/{max_it-1}]\t"
                                        a,b,c = np.mean(ad["loss_sum"]), np.mean(ad["psnr"]), np.mean(ad["ssim"])
                                        str_A = f"Loss:{a:03f}  PSNR:{b:03f}  SSIM:{c:03f}"
                                        s += str_A
                                        print(s)

                                avg_loss = np.mean(ad["loss_sum"])
                                avg_psnr = np.mean(ad["psnr"])
                                avg_ssim = np.mean(ad["ssim"])

                                print(f"{current_type} Loss:{avg_loss}, PSNR:{avg_psnr}, SSIM:{avg_ssim}")

                                Avg_ad.add("loss_sum", avg_loss)
                                Avg_ad.add("psnr", avg_psnr)
                                Avg_ad.add("ssim", avg_ssim)

                            avg_loss = np.mean(Avg_ad["loss_sum"])
                            avg_psnr = np.mean(Avg_ad["psnr"])
                            avg_ssim = np.mean(Avg_ad["ssim"])
                            writer.add_scalar(f"VAL/loss", avg_loss, epoch)
                            writer.add_scalar(f"VAL/psnr", avg_psnr, epoch)
                            writer.add_scalar(f"VAL/ssim", avg_ssim, epoch)
                            print("VAL LOSS", avg_loss)
                            print("VAL PSNR", avg_psnr)
                            print("VAL SSIM", avg_ssim)

                        if (self.current_epoch+1) % 100 == 0:

                            checkpint_name = f"{save_dir}/checkpoint/model_{(self.current_epoch+1)}.pth.tar"
                            self.save_checkpoint(checkpint_name)

                        if avg_loss < lowest_loss:

                            lowest_loss = avg_loss
                            checkpint_name = f"{save_dir}/checkpoint/best.pth.tar"
                            self.save_checkpoint(checkpint_name)
    
    def test(self, local_rank, nprocs):
        sys.path.append("/code/EGVD/network")
        baseline = importlib.import_module(self.args.baseline_type)
        EAVD = getattr(baseline, "EAVD")
        
        print("*"*80)
        print("Begin testing")
        print("*"*80)

        self.local_rank = local_rank
        self.nprocs = nprocs

        torch.cuda.set_device(self.local_rank)
        dist.init_process_group(backend="nccl")

        save_dir = self.args.save_dir
        save_args = ["test"]
        makedir(save_dir, mode=self.args.main_mode)

        for save_item in save_args:
            makedir(f"{save_dir}/{save_item}", mode=self.args.main_mode)

        setup_logging(save_dir, mode=self.args.main_mode)
        print(f"Now saving data in {save_dir}")
        
        self.model = EAVD(self.args, self.local_rank)
        checkpoint_path = f"{save_dir}/checkpoint/best.pth.tar"

        for model_name in self.model.train_model_names:

            if isinstance(model_name, str):
                
                net = getattr(self.model, model_name)
                net.load_state_dict(torch.load(checkpoint_path)[model_name])

        self.count_network_parameters()

        with torch.no_grad():
            
            average_ad = AccumDict()
            scene_types = self.args.scene_types

            for current_type in scene_types:

                os.makedirs(f"{self.args.save_dir}/test/{current_type}/output", exist_ok=True)
                os.makedirs(f"{self.args.save_dir}/test/{current_type}/gt", exist_ok=True)

                ad = AccumDict()

                val_dataset = eval(self.args.test_dataset)(self.args.test_h5_file, self.args.test_sequence_length, current_type, data_type = self.args.test_data_type, transform_flag=False)
                val_data_loader = DataLoader(dataset=val_dataset, batch_size=self.args.batch_size_test, shuffle=False, num_workers = 4, pin_memory = True, drop_last=False)
                
                for idx in range(self.args.test_sequence_length//2):
                    str_frame = f"{current_type}/{idx:08d}, PSNR:{0:.2f}  SSIM:{0:.4f}  LPIPS:{0:.4f}"
                    print(str_frame)

                lstm_state = None
                for iteration, item in enumerate(val_data_loader):

                    self.model.set_input_test(item)
                    if self.args.baseline_type == "base_m1q4":
                        self.model.forward_test()
                    else:
                        lstm_state = self.model.forward_test(lstm_state)
                    out = cv2.cvtColor(self.model.vis_out, cv2.COLOR_RGB2BGR)
                    gt = cv2.cvtColor(self.model.vis_gt, cv2.COLOR_RGB2BGR)

                    # cv2.imwrite(f"{self.args.save_dir}/test/{current_type}/output/{(iteration+(self.args.test_sequence_length//2)):08d}.png", \
                    #                 out)
                    
                    # cv2.imwrite(f"{self.args.save_dir}/test/{current_type}/gt/{(iteration+(self.args.test_sequence_length//2)):08d}.png", \
                    #                 gt)
                    with HiddenPrints():
                        ssim, psnr, lp = get_metrics(out, gt, use_lpips=True)

                    # loss, metrics = self.model.get_losses()
                    # psnr = metrics["psnr"]
                    # ssim = metrics["ssim"]

                    str_frame = f"{current_type}/{(iteration+(self.args.test_sequence_length//2)):08d}  PSNR:{psnr:.2f}  SSIM:{ssim:.4f}  LPIPS:{lp:.4f}"
                    print(str_frame)

                    ad.add("psnr", psnr)
                    ad.add("ssim", ssim)
                    ad.add("lpips", lp)

                for idx in range(self.args.test_sequence_length//2):
                    str_frame = f"{current_type}/{(iteration + idx):08d}, PSNR:{0:.2f}  SSIM:{0:.4f}  LPIPS:{0:.4f}"
                    print(str_frame)

                a,b,c = np.mean(ad["psnr"]), np.mean(ad["ssim"]), np.mean(ad["lpips"])
                str_A = f"{current_type} PSNR:{a:.2f}  SSIM:{b:.4f}  LPIPS:{c:.4f}"
                print(str_A)

                average_ad.add("psnr", a)
                average_ad.add("ssim", b)
                average_ad.add("lpips", c)
                a,b,c = np.mean(average_ad["psnr"]), np.mean(average_ad["ssim"]), np.mean(average_ad["lpips"])
                str_A = f"Best_PSNR:{a:.2f}  Best_SSIM:{b:.4f}  Best_LPIPS:{c:.4f}"
                print(str_A)

    ### 杂货铺
    if True:

        def count_network_parameters(self):

            all_num_params = 0

            print("/n=======================================================================")

            for name in self.model.eval_model_names:

                if isinstance(name, str):

                    net = getattr(self.model, name)

                num_params = num_param(net)

                all_num_params += num_params

                print(f"===> Model {name} has {num_params} parameters")
            
            print(f"The whole Model has {all_num_params} parameters")

            print("/n=======================================================================")

        def save_checkpoint(self, checkpint_name):

            save_dict = {"epoch":self.current_epoch + 1, 
                        "optimizer": self.model.optimizer.state_dict()}
            
            for model_name in self.model.train_model_names:

                if isinstance(model_name, str):
                    
                    net = getattr(self.model, model_name)

                    save_dict[model_name] = net.state_dict()

            torch.save(save_dict, checkpint_name)

        def vis_depth(self, t, i=0):

            out = t[:]

            if len(out.shape) == 4:

                out = out [i][i]

            out = out.detach().cpu().numpy()
            cmap = cm.viridis
            depth_colored = (cmap((out/out.max()).clip(0.0,1.0))*255).astype(np.uint8)

            return depth_colored