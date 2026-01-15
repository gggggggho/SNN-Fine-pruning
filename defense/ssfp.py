'''
Fine-Pruning: Defending Against Backdooring Attacks on Deep Neural Networks

@inproceedings{liu2018fine,
        title={Fine-pruning: Defending against backdooring attacks on deep neural networks},
        author={Liu, Kang and Dolan-Gavitt, Brendan and Garg, Siddharth},
        booktitle={International symposium on research in attacks, intrusions, and defenses},
        pages={273--294},
        year={2018},
        organization={Springer}
        }

basic structure:
1. config args, save_path, fix random seed
2. load the backdoor attack data and backdoor test data
3. load the backdoor attack model
4. fp defense:
    a. hook the activation layer representation of each data
    b. rank the mean of activation for each neural
    c. according to the sorting results, prune and test the accuracy
    d. save the model with the greatest difference between ACC and ASR
5. test the result and get ASR, ACC, RC

'''
from spikingjelly.datasets.n_mnist import NMNIST
from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
from spikingjelly.activation_based import functional, learning, layer, neuron
from torch.utils.data import DataLoader
import argparse
import os, sys
import numpy as np
import torch
import torch.nn as nn
import math
import shutil
import gc
import pandas as pd

np.int = int

sys.path.append('../')
sys.path.append(os.getcwd())

from pprint import pformat
import yaml
import logging
import time
from copy import deepcopy
import torch.nn.utils.prune as prune

from models import get_model
from poisoned_dataset import create_backdoor_data_loader_adaptive
from defense.base import defense
from utils.trainer_cls import Metric_Aggregator, \
    general_plot_for_epoch
from utils.aggregate_block.fix_random import fix_random
from utils.aggregate_block.dataset_and_transform_generate import get_input_shape, get_num_classes
from utils.save_load_attack import save_defense_result
import torch.multiprocessing as mp
import utils.trainer_cls as _tc

mp.set_sharing_strategy('file_system')

# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ----- allocator & backend hygiene (add this block) -----
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True,max_split_size_mb:128,garbage_collection_threshold:0.8"
)
try:
    torch.backends.cudnn.benchmark = False
except Exception:
    pass
# --------------------------------------------------------

_orig_all_acc = _tc.all_acc


def _patched_all_acc(preds, labels):
    if preds.dim() > 1:
        preds = preds.argmax(dim=-1)
    if labels.dim() > 1:
        labels = labels.argmax(dim=-1)
    preds = preds.view(-1)
    labels = labels.view(-1)
    assert preds.numel() == labels.numel(), f"len mismatch: {preds.numel()} vs {labels.numel()}"
    return (preds == labels).float().mean().item()


_tc.all_acc = _patched_all_acc

def eval_loader(model, loader, device, eval_microbatch: int = 8, amp: bool = False, non_blocking: bool = True):
    dev = torch.device(device)
    model.eval().to(device)
    correct = 0
    total = 0
    with torch.no_grad():
        for frames, labels in loader:
            if isinstance(frames, np.ndarray):
                frames = torch.from_numpy(frames)
            if isinstance(labels, np.ndarray):
                labels = torch.from_numpy(labels)

            if frames.dim() == 5:  # [B,T,C,H,W]
                frames = frames.permute(1, 0, 2, 3, 4).contiguous()  # [T,B,C,H,W]
            else:
                pass

                # one-hot -> index
            if labels.dim() > 1:
                labels = labels.argmax(dim=1)

            B = labels.size(0)
            for i in range(0, B, eval_microbatch):
                f_mb = frames[:, i:i + eval_microbatch].to(dev, non_blocking=non_blocking)
                l_mb = labels[i:i + eval_microbatch].to(dev, non_blocking=non_blocking)

                functional.reset_net(model)
                with torch.amp.autocast('cuda', enabled=amp):
                    spike_seq = model(f_mb)  # [T, mb, C]
                    out = spike_seq.mean(dim=0)  # [mb, C]
                    preds = out.argmax(dim=1)

                correct += (preds == l_mb).sum().item()
                total += l_mb.size(0)

                del f_mb, l_mb, spike_seq, out, preds

            del frames, labels
        torch.cuda.empty_cache()

    return 100. * correct / max(1, total)


def eval_asr(model, dataloader, target_label, device, eval_microbatch: int = 8, amp: bool = False, non_blocking: bool = True):
    dev = torch.device(device)
    model.eval().to(device)

    total = 0
    success = 0
    tgt = torch.as_tensor(target_label, device=dev)

    with torch.no_grad():
        for frames, _ in dataloader:
            if isinstance(frames, np.ndarray):
                frames = torch.from_numpy(frames)

            # [B,T,C,H,W] -> [T,B,C,H,W] on CPU
            if frames.dim() == 5:
                frames = frames.permute(1, 0, 2, 3, 4).contiguous()

            B = frames.size(1) if frames.dim() == 5 else frames.size(0)
            for i in range(0, B, eval_microbatch):
                if frames.dim() == 5:
                    f_mb = frames[:, i:i+eval_microbatch].to(dev, non_blocking=non_blocking)
                else:
                    f_mb = frames[:, i:i+eval_microbatch].to(dev, non_blocking=non_blocking)

                functional.reset_net(model)
                with torch.amp.autocast('cuda', enabled=amp):
                    out = model(f_mb).mean(0)              # [mb, C]
                    pred = out.argmax(1)                   # [mb]
                success += (pred == tgt).sum().item()
                total += pred.numel()

                del f_mb, out, pred

            del frames
        torch.cuda.empty_cache()

    return 100. * success / max(1, total)


class CleanToMixLoader:

    def __init__(self, base_loader):
        self.base = base_loader
        self._seen = 0

    def __len__(self):
        return len(self.base)

    def __iter__(self):
        seen = 0
        for batch in self.base:
            if isinstance(batch, (tuple, list)) and len(batch) >= 2:
                x, y = batch[0], batch[1]
            else:
                raise ValueError("trainloader batch가 (x, y) 형태가 아닙니다.")

            b = y.size(0)
            idx = torch.arange(seen, seen + b, dtype=torch.long)
            seen += b
            poison_indicator = torch.zeros(b, dtype=torch.bool)
            original_targets = y.clone()
            yield x, y, idx, poison_indicator, original_targets


class BdToMixLoader:
    def __init__(self, base_loader):
        self.base = base_loader

    def __len__(self):
        return len(self.base)

    def __iter__(self):
        seen = 0
        for x, y in self.base:
            b = y.size(0)
            idx = torch.arange(seen, seen + b, dtype=torch.long)
            seen += b
            poison_indicator = torch.ones(b, dtype=torch.bool)
            original_targets = y.clone()
            yield x, y, idx, poison_indicator, original_targets


def _maybe_permute_frames(frames: torch.Tensor):
    # [B,T,C,H,W] -> [T,B,C,H,W]
    return frames.permute(1, 0, 2, 3, 4).contiguous() if frames.dim() == 5 else frames


def hybrid_stdp_gd_finetune(
    model: nn.Module,
    clean_train_loader,
    clean_test_loader,
    bd_test_loader,
    epochs: int,
    device: torch.device,
    tau_pre: float,
    tau_post: float,
    a_pre: float,
    a_post: float,
    stdp_lr: float,
    gd_lr: float,
    stdp_clamp: float,
    save_dir: str,
    trigger_label: int,
    amp: bool = False,
    non_blocking: bool = True
):
    dev = torch.device(device)
    model.to(dev).train()
    step_mode = 'm'  # 멀티스텝
    functional.set_step_mode(model, step_mode)

    # Conv들에 STDPLearner 연결
    stdp_learners = []
    conv_params = []

    synapse = model.conv_fc[0]
    post_synaptic_neuron = model.conv_fc[2]

    learner = learning.STDPLearner(
        step_mode=step_mode,
        synapse=synapse,
        sn=post_synaptic_neuron,
        tau_pre=tau_pre,
        tau_post=tau_post,
        # f_pre/f_post는 weight에 작용하는 스케일러 역할
        f_pre=lambda w: torch.clamp(w, -stdp_clamp, stdp_clamp) * a_pre,
        f_post=lambda w: torch.clamp(w, -stdp_clamp, stdp_clamp) * a_post
    )
    stdp_learners.append(learner)
    conv_params += list(p for p in synapse.parameters() if p.requires_grad)

    # GD 파라미터
    conv_param_ids = set(id(p) for p in conv_params)
    gd_params = [p for p in model.parameters() if p.requires_grad and id(p) not in conv_param_ids]

    # 옵티마이저
    optimizer_stdp = torch.optim.SGD(conv_params, lr=stdp_lr, momentum=0.0)
    optimizer_gd   = torch.optim.Adam(gd_params,  lr=gd_lr)

    os.makedirs(save_dir, exist_ok=True)

    for learner in stdp_learners:
        learner.disable()
    torch.cuda.empty_cache()

    # 초기 평가
    test_acc_list, test_asr_list = [], []
    acc0 = eval_loader(model, clean_test_loader, dev, amp=amp, non_blocking=non_blocking)
    asr0 = eval_asr(model, bd_test_loader, trigger_label, dev, amp=amp, non_blocking=non_blocking)
    logging.info(f"[Hybrid-FT] Before: ACC={acc0:.2f}%  ASR={asr0:.2f}%")
    test_acc_list.append(float(acc0/100))
    test_asr_list.append(float(asr0/100))

    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    for ep in range(epochs):
        model.train()
        for frames, labels in clean_train_loader:
            frames = frames.to(dev, non_blocking=non_blocking)
            labels = labels.to(dev, non_blocking=non_blocking)
            if frames.dim() == 5:
                frames = _maybe_permute_frames(frames)  # [T,B,C,H,W]

            optimizer_gd.zero_grad(set_to_none=True)
            optimizer_stdp.zero_grad(set_to_none=True)
            functional.reset_net(model)
            for L in stdp_learners:
                L.reset()

            for learner in stdp_learners:
                learner.reset()
                learner.enable()

            with torch.cuda.amp.autocast(enabled=amp):
                out_seq = model(frames)           # [T,B,C]
                y_pred  = out_seq.mean(0)         # [B,C]
                loss    = torch.nn.functional.cross_entropy(y_pred, labels)

            scaler.scale(loss).backward()

            for p in conv_params:
                if p.grad is not None:
                    p.grad.zero_()

            for L in stdp_learners:
                L.step(on_grad=True)

            scaler.step(optimizer_gd)
            scaler.update()
            optimizer_stdp.step()

            for learner in stdp_learners:
                learner.disable()

            functional.reset_net(model)
            for L in stdp_learners:
                L.reset()

        # Epoch 평가
        acc = eval_loader(model, clean_test_loader, dev, amp=amp, non_blocking=non_blocking)
        asr = eval_asr(model, bd_test_loader, trigger_label, dev, amp=amp, non_blocking=non_blocking)
        logging.info(f"[Hybrid-FT] Epoch {ep+1}/{epochs}: ACC={acc:.2f}%  ASR={asr:.2f}%")
        test_acc_list.append(float(acc/100))
        test_asr_list.append(float(asr/100))

        # 플롯/로그
        try:
            _tc.plot_acc_like_metric_pure(
                [], test_acc_list, test_asr_list,
                save_folder_path=save_dir,
                save_file_name="acc_like_metric_plots"
            )
        except Exception as e:
            logging.debug(f"[Hybrid-FT] plotting fail: {e}")

    excel_path = os.path.join(save_dir, "finetune_log.xlsx")

    with pd.ExcelWriter(excel_path) as writer:
        df_finetune = pd.DataFrame({
            'epoch': range(len(test_acc_list)),
            'test_acc': test_acc_list,
            'test_asr': test_asr_list
        })
        df_finetune.to_excel(writer, sheet_name='finetuning_log', index=False)

    # 종료
    for L in stdp_learners:
        L.disable()
    return model

class FinePrune(defense):
    r"""Fine-Pruning: Defending Against Backdooring Attacks on Deep Neural Networks

    basic structure:

    1. config args, save_path, fix random seed
    2. load the backdoor attack data and backdoor test data
    3. load the backdoor attack model
    4. fp defense:
        a. hook the activation layer representation of each data
        b. rank the mean of activation for each neural
        c. according to the sorting results, prune and test the accuracy
        d. save the model with the greatest difference between ACC and ASR
    5. test the result and get ASR, ACC, RC

    .. code-block:: python

        parser = argparse.ArgumentParser(description=sys.argv[0])
        FinePrune.add_arguments(parser)
        args = parser.parse_args()
        FinePrune_method = FinePrune(args)
        if "result_file" not in args.__dict__:
            args.result_file = 'one_epochs_debug_badnet_attack'
        elif args.result_file is None:
            args.result_file = 'one_epochs_debug_badnet_attack'
        result = FinePrune_method.defense(args.result_file)

    .. Note::
        @inproceedings{liu2018fine,
        title={Fine-pruning: Defending against backdooring attacks on deep neural networks},
        author={Liu, Kang and Dolan-Gavitt, Brendan and Garg, Siddharth},
        booktitle={International symposium on research in attacks, intrusions, and defenses},
        pages={273--294},
        year={2018},
        organization={Springer}
        }

    Args:
        baisc args: in the base class
        ratio (float): the ratio of clean data loader
        index (str): the index of clean data
        acc_ratio (float): the tolerance ration of the clean accuracy
        once_prune_ratio (float): how many percent once prune. in 0 to 1

    """

    def __init__(self):
        super(FinePrune).__init__()
        pass

    def set_args(self, parser):
        parser.add_argument('--epsilon', default=0.1, type=float, help='The percentage of poisoned data')
        parser.add_argument('--least', action='store_true', help='Use least active area for smart attack')
        parser.add_argument('--most_polarity', action='store_true',
                            help='Use most active polarity in the area for smart attack')
        parser.add_argument('--momentum', default=0.9, type=float, help='Momentum')
        parser.add_argument('--n_masks', default=2, type=int,
                            help='The number of masks. Only if the trigger type is smart')
        parser.add_argument('--polarity', default=0, type=int, help='The polarity of the trigger', choices=[0, 1, 2, 3])
        parser.add_argument('--trigger_size', default=0.1, type=float,
                            help='The size of the trigger as the percentage of the image size')
        parser.add_argument('--trigger_type', default='static', type=str, help='static | moving | smart')
        parser.add_argument('--pos', default='top-left', type=str, help='The position of the trigger',
                            choices=['top-left', 'top-right', 'bottom-left', 'bottom-right', 'middle', 'random'])
        parser.add_argument('--trigger_label', type=int, default=0, help='The index of the trigger label')
        parser.add_argument('--data_dir', type=str, default='data', help='Path to root folder of datasets')
        parser.add_argument('--T', type=int, required=True, help='Time steps')
        parser.add_argument('--adam_betas', nargs=2, type=float, default=(0.9, 0.999),
                            help='Beta coefficients for Adam optimizer')
        parser.add_argument('--adam_eps', type=float, default=1e-8, help='Epsilon for Adam optimizer')

        parser.add_argument('--model_path', type=str, required=True, help='Path to trained backdoor model (.pth)')
        parser.add_argument('--save_path', type=str, default=None, help='Custom directory to save defense results')
        parser.add_argument("-pm", "--pin_memory", type=lambda x: str(x) in ['True', 'true', '1'],
                            help="dataloader pin_memory")
        parser.add_argument('--sgd_momentum', type=float)
        parser.add_argument('--wd', type=float, default=0.0, help='weight decay of sgd')
        parser.add_argument('--client_optimizer', type=int)
        parser.add_argument('--amp', type=lambda x: str(x).lower() in ['true', '1'], default=False)
        parser.add_argument('--frequency_save', type=int,
                            help=' frequency_save, 0 is never')
        parser.add_argument('--device', type=str, help='cuda, cpu')
        parser.add_argument("-nb", "--non_blocking", type=lambda x: str(x) in ['True', 'true', '1'],
                            help=".to(), set the non_blocking = ?")

        parser.add_argument('--prune_by', type=str, default='spike',
                            choices=['spike', 'activation'], help='프루닝 기준')
        parser.add_argument('--spike_threshold', type=float, default=0.0,
                            help='스파이크 판정 임계값(>thr → spike=1)')
        parser.add_argument('--stdp_tau_pre', type=float, default=5.0, help='STDP tau_pre')
        parser.add_argument('--stdp_tau_post', type=float, default=20.0, help='STDP tau_post')
        parser.add_argument('--stdp_clamp', type=float, default=2.0, help='|w| clamp for STDP f_pre/f_post')
        parser.add_argument('--stdp_a_pre', type=float, default=0.003, help='STDP LTP scale')
        parser.add_argument('--stdp_a_post', type=float, default=-0.004, help='STDP LTD scale (negative)')

        parser.add_argument('--gd_lr', type=float, default=1e-3, help='GD(FC 등) learning rate (Adam)')
        parser.add_argument('--stdp_lr', type=float, default=0.1, help='STDP(Conv)용 옵티마 스텝 스케일러')

        parser.add_argument('--dataset', type=str, help='nmnist, gesture, cifar10, gtsrb, celeba, tiny')
        parser.add_argument("--num_classes", type=int)
        parser.add_argument("--input_height", type=int)
        parser.add_argument("--input_width", type=int)
        parser.add_argument("--input_channel", type=int)

        parser.add_argument('--epochs', type=int, default=10)
        parser.add_argument('--batch_size', type=int)
        parser.add_argument("--num_workers", type=float)
        parser.add_argument('--lr', default=0.001, type=float)
        parser.add_argument('--lr_scheduler', type=str, default='none', help='the scheduler of lr')

        parser.add_argument('--attack', type=str)
        parser.add_argument('--poison_rate', type=float)
        parser.add_argument('--target_type', type=str, help='all2one, all2all, cleanLabel')
        parser.add_argument('--target_label', type=int)
        parser.add_argument('--model', type=str, help='resnet18')
        parser.add_argument('--random_seed', type=int, help='random seed')
        parser.add_argument('--index', type=str, help='index of clean data')
        parser.add_argument('--result_file', type=str, help='the location of result')
        parser.add_argument('--yaml_path', type=str, default="./config/defense/fp/config.yaml", help='the path of yaml')

        # set the parameter for the fp defense
        parser.add_argument('--ratio', type=float, help='the ratio of clean data loader')
        parser.add_argument('--acc_ratio', type=float, help='the tolerance ration of the clean accuracy')
        parser.add_argument("--once_prune_ratio", type=float, help="how many percent once prune. in 0 to 1")
        return parser

    def add_yaml_to_args(self, args):
        with open(args.yaml_path, 'r') as f:
            defaults = yaml.safe_load(f)
        defaults.update({k: v for k, v in args.__dict__.items() if v is not None})
        args.__dict__ = defaults

    def process_args(self, args):
        args.terminal_info = sys.argv
        args.num_classes = get_num_classes(args.dataset)
        args.input_height, args.input_width, args.input_channel = get_input_shape(args.dataset)
        args.img_size = (args.input_height, args.input_width, args.input_channel)
        args.data_dir = args.data_dir
        args.type = args.trigger_type
        if args.save_path:
            args.defense_save_path = args.save_path
        else:
            cfg_name = f"hssfp_{args.dataset}_{args.type}_{args.epsilon}_{args.trigger_size}_{args.pos}_{args.once_prune_ratio}_{args.acc_ratio}_{args.epochs}"
            args.defense_save_path = os.path.join("experiments", cfg_name)

        os.makedirs(args.defense_save_path, exist_ok=True)
        return args

        defense_save_path = "record" + os.path.sep + args.result_file + os.path.sep + "defense" + os.path.sep + "fp"
        # if os.path.exists(defense_save_path):
        #     shutil.rmtree(defense_save_path)
        os.makedirs(defense_save_path, exist_ok=True)
        # save_path = '/record/' + args.result_file
        # if args.checkpoint_save is None:
        #     args.checkpoint_save = save_path + '/record/defence/fp/'
        #     if not (os.path.exists(os.getcwd() + args.checkpoint_save)):
        #         os.makedirs(os.getcwd() + args.checkpoint_save)
        # if args.log is None:
        #     args.log = save_path + '/saved/fp/'
        #     if not (os.path.exists(os.getcwd() + args.log)):
        #         os.makedirs(os.getcwd() + args.log)
        # args.save_path = save_path
        args.defense_save_path = defense_save_path
        return args

    def prepare(self, args):

        if args.dataset == 'nmnist':
            root = args.data_dir
            mn = os.path.join(root, 'mnist')
            nn = os.path.join(root, 'nmnist')
            if os.path.isdir(mn) and not os.path.exists(nn):
                try:
                    os.symlink(mn, nn)  # 윈도우에서 관리자 권한 필요할 수 있음
                except (AttributeError, OSError):
                    shutil.copytree(mn, nn)
        elif args.dataset == 'gesture':
            root = args.data_dir
            os.path.join(root, 'gesture')

        ### set the logger
        logFormatter = logging.Formatter(
            fmt='%(asctime)s [%(levelname)-8s] [%(filename)s:%(lineno)d] %(message)s',
            datefmt='%Y-%m-%d:%H:%M:%S',
        )
        logger = logging.getLogger()

        os.makedirs(args.defense_save_path, exist_ok=True)

        # file Handler
        fileHandler = logging.FileHandler(
            args.defense_save_path + '/' + time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()) + '.log')
        fileHandler.setFormatter(logFormatter)
        fileHandler.setLevel(logging.DEBUG)
        logger.addHandler(fileHandler)
        # consoleHandler
        consoleHandler = logging.StreamHandler()
        consoleHandler.setFormatter(logFormatter)
        consoleHandler.setLevel(logging.INFO)
        logger.addHandler(consoleHandler)
        # overall logger level should <= min(handler) otherwise no log will be recorded.
        logger.setLevel(0)
        # disable other debug, since too many debug
        logging.getLogger('PIL').setLevel(logging.WARNING)
        logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)

        logging.info(pformat(args.__dict__))

        logging.debug("Only INFO or above level log will show in cmd. DEBUG level log only will show in log file.")

        '''
                load_dict = {
                        'model_name': load_file['model_name'],
                        'model': load_file['model'],
                        'clean_train': clean_train_dataset_with_transform,
                        'clean_test' : clean_test_dataset_with_transform,
                        'bd_train': bd_train_dataset_with_transform,
                        'bd_test': bd_test_dataset_with_transform,
                    }
                '''

        fix_random(args.random_seed)
        self.args = args

        # 백도어 학습된 SNN 모델 불러오기
        ckpt = torch.load(
            args.model_path,
            map_location=args.device,
            weights_only=False
        )
        if isinstance(ckpt, dict):
            model = get_model(args.dataset, args.T)
            if 'model_state_dict' in ckpt:
                state = ckpt['model_state_dict']
            elif 'model' in ckpt:
                state = ckpt['model']
            elif 'state_dict' in ckpt:
                state = ckpt['state_dict']
            else:
                raise KeyError("Checkpoint dict has no 'model_state_dict' or 'model' key.")
            model.load_state_dict(state)
        else:
            model = ckpt

        model.to(args.device).eval().requires_grad_(False)
        self.netC = model

        # clean/backdoor train & test DataLoader 생성
        clean_train_loader, bd_train_loader, clean_test_loader, bd_test_loader = (
            create_backdoor_data_loader_adaptive(args))

        split_dir = f"frames_number_{args.T}_split_by_number"
        root = os.path.join(args.data_dir, args.dataset)

        if args.dataset == 'nmnist':
            orig_testset = NMNIST(
                root=root,
                train=False,
                data_type='frame',
                frames_number=args.T,
                split_by='number',
                custom_integrated_frames_dir_name=split_dir,
                transform=None
            )
            orig_test_loader = DataLoader(
                orig_testset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=0
            )

        elif args.dataset == 'gesture':
            orig_testset = DVS128Gesture(
                root=root,
                train=False,
                data_type='frame',
                frames_number=args.T,
                split_by='number',
                custom_integrated_frames_dir_name=split_dir,
                transform=None
            )
            orig_test_loader = DataLoader(
                orig_testset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=0
            )

        self.clean_train_loader = clean_train_loader
        self.clean_test_loader = clean_test_loader
        self.bd_test_loader = bd_test_loader
        self.bd_train_loader = bd_train_loader
        self.orig_test_loader = orig_test_loader

    def defense(self):

        netC = self.netC
        args = self.args

        trainloader = self.clean_train_loader
        clean_test_loader = self.clean_test_loader
        bd_test_loader = self.bd_test_loader

        # baseline 확인
        baseline_acc = eval_loader(self.netC, self.orig_test_loader, args.device, amp=args.amp, non_blocking=args.non_blocking)
        print("Baseline ACC:", baseline_acc)

        reset_hook = netC.register_forward_pre_hook(
            lambda module, inputs: functional.reset_net(netC)
        )

        initial_asr = eval_asr(self.netC, bd_test_loader, args.trigger_label, args.device, amp=args.amp, non_blocking=args.non_blocking)
        print(f"Before any pruning, backdoor ASR = {initial_asr:.2f}%")

        # 모델 구조에 따라  레이어 변경
        if args.dataset == 'nmnist':
            feat_layer = netC.conv_fc[11]
        elif args.dataset == 'gesture':
            feat_layer = netC.conv_fc[23]

        hook = feat_layer.register_forward_hook(lambda module, inp, out: setattr(self, 'result_mid', out.detach()))

        logging.info("Forwarding all the training dataset:")
        with torch.no_grad():
            spike_counts = None
            total = 0
            for inputs, _ in trainloader:
                frames = inputs.to(args.device).transpose(0, 1)
                functional.reset_net(netC)
                _ = netC(frames)
                x = self.result_mid
                x = x.reshape(-1, x.size(-1))

                if args.prune_by == 'spike':
                    if x.dtype == torch.bool:
                        x_spk = x.float()
                    else:
                        x_spk = (x > args.spike_threshold).float()
                    v = x_spk.sum(dim=0)
                else:
                    v = x.abs().sum(dim=0)
                if spike_counts is None:
                    spike_counts = torch.zeros_like(v)
                spike_counts += v
                total += x.size(0)
        hook.remove()
        reset_hook.remove()

        seq_sort = torch.argsort(spike_counts)
        logging.info(f"get seq_sort, (len={len(seq_sort)}), seq_sort:{seq_sort}")
        # del container

        # 모델 구조에 따라  레이어 변경
        if args.dataset == 'nmnist':
            out_layer = netC.conv_fc[13]
        elif args.dataset == 'gesture':
            out_layer = netC.conv_fc[25]
        prune_mask = torch.ones_like(out_layer.weight)
        prune_mask = torch.ones_like(out_layer.weight)

        prune_info_recorder = Metric_Aggregator()
        test_acc_list = []
        test_asr_list = []

        # start from 0, so unprune case will also be tested.
        # for num_pruned in range(0, len(seq_sort), 500):
        for num_pruned in range(0, len(seq_sort), math.ceil(len(seq_sort) * args.once_prune_ratio)):
            net_pruned = deepcopy(netC).to(args.device)
            if num_pruned:
                # add_pruned_channnel_index = seq_sort[num_pruned - 1] # each time prune_mask ADD ONE MORE channel being prune.
                pruned_hidden = seq_sort[:num_pruned]  # everytime we prune all
                prune_mask[:, pruned_hidden] = 0
                prune.custom_from_mask(out_layer, name='weight', mask=prune_mask.to(args.device))

                # prune_ratio = 100. * float(torch.sum(first_linear_module_in_last_child.weight_mask == 0)) / float(first_linear_module_in_last_child.weight_mask.nelement())
                # logging.info(f"Pruned {num_pruned}/{len(seq_sort)}  ({float(prune_ratio):.2f}%) filters")

            # test
            test_acc = eval_loader(net_pruned, self.orig_test_loader, args.device, amp=args.amp, non_blocking=args.non_blocking)
            test_asr = eval_asr(net_pruned, bd_test_loader, args.trigger_label, args.device, amp=args.amp, non_blocking=args.non_blocking)

            prune_info_recorder({
                "num_pruned": num_pruned,
                "all_filter_num": len(seq_sort),
                "test_acc": test_acc,
                "test_asr": test_asr,
            })

            test_acc_list.append(float(test_acc))
            test_asr_list.append(float(test_asr))

            if num_pruned == 0:
                test_acc_cl_ori = test_acc
                last_net = deepcopy(net_pruned).cpu()
                last_index = 0
            if (test_acc / test_acc_cl_ori) >= args.acc_ratio:
                last_net = deepcopy(net_pruned).cpu()
                last_index = num_pruned
            else:
                last_net = deepcopy(net_pruned)
                break

        prune_info_recorder.to_dataframe().to_csv(os.path.join(self.args.defense_save_path, "prune_log.csv"))
        prune_info_recorder.summary().to_csv(os.path.join(self.args.defense_save_path, "prune_log_summary.csv"))
        general_plot_for_epoch(
            {
                "test_acc": test_acc_list,
                "test_asr": test_asr_list,
            },
            os.path.join(self.args.defense_save_path, "prune_log_plot.jpg"),
            ylabel='percentage',
            xlabel="num_pruned",
        )

        logging.info(
            f"End prune. Pruned {num_pruned}/{len(seq_sort)} test_acc:{test_acc:.2f}  test_asr:{test_asr:.2f}  ")

        # finetune
        last_net.train()
        last_net.requires_grad_(True)
        last_net = last_net.to(args.device)

        for _, p in last_net.named_parameters():
            p.requires_grad_(True)

        finetune_dir = os.path.join(self.args.defense_save_path, "finetune")
        os.makedirs(finetune_dir, exist_ok=True)

        del netC 
        gc.collect()
        torch.cuda.empty_cache()

        last_net = hybrid_stdp_gd_finetune(
            model=last_net,
            clean_train_loader=trainloader,
            clean_test_loader=self.clean_test_loader,
            bd_test_loader=bd_test_loader,
            epochs=args.epochs,
            device=torch.device(args.device),
            tau_pre=args.stdp_tau_pre,
            tau_post=args.stdp_tau_post,
            a_pre=args.stdp_a_pre,
            a_post=args.stdp_a_post,
            stdp_lr=args.stdp_lr,
            gd_lr=args.gd_lr,
            stdp_clamp=args.stdp_clamp,
            save_dir=finetune_dir,
            trigger_label=args.trigger_label,
            amp=args.amp,
            non_blocking=args.non_blocking
        )

        save_defense_result(
            model_name=args.model,
            num_classes=args.num_classes,
            model=last_net.cpu().state_dict(),
            save_path=self.args.defense_save_path,
        )

        # mask = deepcopy(first_linear_module_in_last_child.weight_mask)
        # prune.remove(first_linear_module_in_last_child, 'weight')
        #
        # torch.save(
        #     {
        #         'model_name': args.model,
        #         'model': last_net.cpu().state_dict(),
        #         'seq_sort': seq_sort,
        #         "num_pruned":num_pruned,
        #         "mask":mask,
        #         "last_child_name":last_child_name,
        #         "first_module_name":first_module_name,
        #     },
        #     self.args.defense_save_path+os.path.sep+"defense_result.pt"
        # )


if __name__ == '__main__':
    fp = FinePrune()
    parser = argparse.ArgumentParser(description=sys.argv[0])
    parser = fp.set_args(parser)
    args = parser.parse_args()
    # fp.add_yaml_to_args(args)
    args = fp.process_args(args)
    fp.prepare(args)
    fp.defense()
