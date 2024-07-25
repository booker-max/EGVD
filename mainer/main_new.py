"""
每一次切换任务需要改
1. from trainer_raw import trainer 改成需要导入trainer的路径
2. 需要更改 base_options_path
3. 需要更改 parser.add_argument("--save_dir", type=str, default = )保存模型的路径
"""

import argparse, yaml, os, random, sys
import numpy as np
import torch.backends.cudnn as cudnn
import torch

import warnings
warnings.filterwarnings("ignore", message="The default behavior for interpolate/upsample")
warnings.filterwarnings("ignore", message="nn.functional.sigmoid is deprecated.")
warnings.filterwarnings("ignore", message="nn.functional.tanh is deprecated.")

def get_args():

    save_dir = "/data/booker/LN_base/TMM/exp"
    checkpoint_dir = "RDDNet_NN"
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, default=f"{save_dir}/EventRepresentation/event_imgae")
    parser.add_argument("--checkpoint_dir", type=str, default=f"{save_dir}/EventRepresentation/event_imgae")
    parser.add_argument("--data_type", type=str, default=f"event_imgae")
    parser.add_argument("--test_data_type", type=str, default=f"event_imgae")
    parser.add_argument("--train_scene_num", type=int, default=8)
    parser.add_argument("--train_dataset_type", type=str, default=f"event_imgae")
    parser.add_argument("--train_dataset", type=str, default=f"event_imgae")
    parser.add_argument("--test_dataset", type=str, default=f"event_imgae")
    parser.add_argument("--scene_types", nargs="+", default=f"event_imgae")
    parser.add_argument("--trainer_type", type=str, default=f"trainer_eventrepresentation")
    parser.add_argument("--baseline_type", type=str, default=f"base_restormer")
    parser.add_argument("--ablation_type", type=str, default=f"MSEG_e")
    parser.add_argument("--main_mode", type=str, default=f"train")
    parser.add_argument("--enable_vis", action="store_true")
    parser.add_argument("--transform_flag", action="store_true")
    parser.add_argument("--enable_pretrain", action="store_true")
    parser.add_argument("--pretrain_old", action="store_true")
    parser.add_argument("--EAVD_flag", action="store_true")
    parser.add_argument("--scdformer_resolution",  nargs="+", default=None)
    parser.add_argument("--base_path", type=str, default=f"/code/EGVD/options/base.yaml")
    parser.add_argument("--sequence_length", type=int, default=7) #这个只能为1或者7
    parser.add_argument("--test_sequence_length", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--batch_size_val", type=int, default=4)
    parser.add_argument("--batch_size_test", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--checkpoint", type=str, default=f"{save_dir}/{checkpoint_dir}/checkpoint/best.pth.tar")
    parser.add_argument('--local_rank', default = 0, type = int, help = "node rank for distributed training")

    args, _ = parser.parse_known_args()

    with open(args.base_path) as f:

        config = yaml.load(f, Loader=yaml.FullLoader)

    parser.set_defaults(**config)

    return parser.parse_args()

def init_seeds(seed=0, cuda_deterministic=True):

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if cuda_deterministic:
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:
        cudnn.deterministic = False
        cudnn.benchmark = True

def main():

    rank = int(os.environ["RANK"])
    args = get_args()
    args.nprocs = torch.cuda.device_count()
    args.rank = rank

    random_seed = 1234
    ## to do, 这里是rank, 还是local_rank
    init_seeds(random_seed + args.local_rank)

    print(f"Now rank is {args.rank}, process is {args.local_rank}")

    import sys,importlib
    sys.path.append("/code/EGVD/trainer")
    trainer_v1 = importlib.import_module(args.trainer_type).trainer(args)
    
    if args.main_mode == "test":
        trainer_v1.test(args.local_rank, args.nprocs)
    elif args.main_mode == "test_real" or args.main_mode == "test_realv2":
        trainer_v1.test_real(args.local_rank, args.nprocs)
    else:
        trainer_v1.train(args.local_rank, args.nprocs)

if __name__ == "__main__":

    main()