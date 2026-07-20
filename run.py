import argparse
import random

import numpy as np
import torch

from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from utils.print_args import print_args


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def build_parser():
    parser = argparse.ArgumentParser(description="MoSTE")

    parser.add_argument("--task_name", type=str, default="long_term_forecast")
    parser.add_argument("--is_training", type=int, default=1)
    parser.add_argument("--model_id", type=str, default="MoSTE_BTH")
    parser.add_argument("--model", type=str, default="MoSTE")

    parser.add_argument("--data", type=str, default="only_do")
    parser.add_argument("--root_path", type=str, default="./data/data_series/")
    parser.add_argument("--data_path", type=str, default="BTH.csv")
    parser.add_argument(
        "--prior_graph_path",
        type=str,
        default="./data/data_adj/adj_BTH_new.pkl",
    )
    parser.add_argument("--features", type=str, default="M", choices=["M", "MS", "S"])
    parser.add_argument("--target", type=str, default="yuqiaoshuikuchukou")
    parser.add_argument("--freq", type=str, default="h", choices=["h"])
    parser.add_argument("--checkpoints", type=str, default="./checkpoints/")
    parser.add_argument("--results", type=str, default="./results/")
    parser.add_argument("--results_npy", type=str, default="./results_npy/")

    parser.add_argument("--seq_len", type=int, default=96)
    parser.add_argument("--label_len", type=int, default=48)
    parser.add_argument("--pred_len", type=int, default=96)
    parser.add_argument("--inverse", action="store_true", default=False)

    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--enc_in", type=int, default=24)
    parser.add_argument("--c_out", type=int, default=24)
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--e_layers", type=int, default=2)
    parser.add_argument("--d_ff", type=int, default=128)
    parser.add_argument("--moving_avg", type=int, default=25)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--embed", type=str, default="timeF", choices=["timeF"])
    parser.add_argument("--down_sampling_layers", type=int, default=3)
    parser.add_argument("--down_sampling_window", type=int, default=2)
    parser.add_argument(
        "--down_sampling_method",
        type=str,
        default="avg",
        choices=["avg", "max", "conv"],
    )
    parser.add_argument("--channel_independence", type=int, default=1)
    parser.add_argument(
        "--decomp_method",
        type=str,
        default="moving_avg",
        choices=["moving_avg", "dft_decomp"],
    )
    parser.add_argument("--use_norm", type=int, default=1)
    parser.add_argument("--steps_per_day", type=int, default=6)
    parser.add_argument("--structure_weight", type=float, default=0.1)

    parser.add_argument("--prior_layers", type=int, default=2)
    parser.add_argument(
        "--prior_module_type",
        type=str,
        default="sharing",
        choices=["sharing", "individual"],
    )
    parser.add_argument(
        "--prior_activation",
        type=str,
        default="GLU",
        choices=["GLU", "relu"],
    )
    parser.add_argument("--prior_use_mask", type=str2bool, default=True)
    parser.add_argument("--prior_temporal_embedding", type=str2bool, default=True)
    parser.add_argument("--prior_spatial_embedding", type=str2bool, default=True)
    parser.add_argument("--prior_projection_hidden", type=int, default=128)

    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--itr", type=int, default=1)
    parser.add_argument("--train_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--des", type=str, default="Exp")
    parser.add_argument("--print_every", type=int, default=100)

    parser.add_argument("--use_gpu", type=str2bool, default=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--use_multi_gpu", action="store_true", default=False)
    parser.add_argument("--devices", type=str, default="0")
    return parser


def build_setting(args, iteration):
    return (
        f"{args.task_name}_{args.model_id}_{args.model}_{args.data}_"
        f"ft{args.features}_sl{args.seq_len}_ll{args.label_len}_"
        f"pl{args.pred_len}_dm{args.d_model}_el{args.e_layers}_df{args.d_ff}_{args.des}_{iteration}"
    )


def main():
    args = build_parser().parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    args.use_gpu = bool(torch.cuda.is_available() and args.use_gpu)
    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(" ", "")
        args.device_ids = [int(device_id) for device_id in args.devices.split(",")]
        args.gpu = args.device_ids[0]

    print("Args in experiment:")
    print_args(args)

    if args.is_training:
        for iteration in range(args.itr):
            setting = build_setting(args, iteration)
            exp = Exp_Long_Term_Forecast(args)
            print(f">>>>>>>start training : {setting}>>>>>>>>>>>>>>>>>>>>>>>>>>")
            exp.train(setting)
            print(f">>>>>>>testing : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
            exp.test(setting)
            if args.use_gpu:
                torch.cuda.empty_cache()
    else:
        setting = build_setting(args, 0)
        exp = Exp_Long_Term_Forecast(args)
        print(f">>>>>>>testing : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        exp.test(setting, test=1)


if __name__ == "__main__":
    main()
