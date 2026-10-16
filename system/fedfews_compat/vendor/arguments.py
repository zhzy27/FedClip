import argparse

def str2bool(v):
    """
    Convert string to boolean for argparse.

    argparse's type=bool is broken: bool("false") == True!
    This function correctly handles various string representations.
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def source_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-go', "--goal", type=str, default="test",
                            help="The goal for this experiment")
    parser.add_argument('-dev', "--device", type=str, default="cuda",
                            choices=["cpu", "cuda"])
    parser.add_argument('-did', "--device_id", type=str, default="0")
    parser.add_argument('-data', "--dataset", type=str, default="MNIST")
    parser.add_argument('-ncl', "--num_classes", type=int, default=10)
    parser.add_argument('-m', "--model", type=str, default="CNN")
    parser.add_argument('-lbs', "--batch_size", type=int, default=10)
    parser.add_argument('-lr', "--local_learning_rate", type=float, default=0.005,
                            help="Local learning rate")
    parser.add_argument('-ld', "--learning_rate_decay", type=bool, default=False)
    parser.add_argument('-ldg', "--learning_rate_decay_gamma", type=float, default=0.99)
    parser.add_argument('-gr', "--global_rounds", type=int, default=2000)
    parser.add_argument('-tc', "--top_cnt", type=int, default=100,
                            help="For auto_break")
    parser.add_argument('-ls', "--local_epochs", type=int, default=1,
                            help="Multiple update steps in one local epoch.")
    parser.add_argument('-algo', "--algorithm", type=str, default="FedAvg")
    parser.add_argument('-jr', "--join_ratio", type=float, default=1.0,
                            help="Ratio of clients per round")
    parser.add_argument('-rjr', "--random_join_ratio", type=bool, default=False,
                            help="Random ratio of clients per round")
    parser.add_argument('-nc', "--num_clients", type=int, default=20,
                            help="Total number of clients")
    parser.add_argument('-pv', "--prev", type=int, default=0,
                            help="Previous Running times")
    parser.add_argument('-t', "--times", type=int, default=1,
                            help="Running times")
    parser.add_argument('-eg', "--eval_gap", type=int, default=1,
                            help="Rounds gap for evaluation")
    parser.add_argument('-nth', "--num_threads", type=int, default=1,
                            help="Number of threads for parallel client training (default: 1, sequential)")
    parser.add_argument('-sfn', "--save_folder_name", type=str, default='items',
                           help="Folder name for saving models (deprecated, use --model-dir)")
    parser.add_argument('-mdir', "--model_dir", type=str, default=None,
                           help="Directory for saving models (default: ./models/)")
    parser.add_argument('-rdir', "--result_dir", type=str, default=None,
                           help="Directory for saving results (default: ./results/)")
    parser.add_argument('-ab', "--auto_break", type=bool, default=False)
    parser.add_argument('-dlg', "--dlg_eval", type=bool, default=False)
    parser.add_argument('-dlgg', "--dlg_gap", type=int, default=100)
    parser.add_argument('-bnpc', "--batch_num_per_client", type=int, default=2)
    parser.add_argument('-nnc', "--num_new_clients", type=int, default=0)
    parser.add_argument('-ften', "--fine_tuning_epoch_new", type=int, default=0)
    parser.add_argument('-fd', "--feature_dim", type=int, default=512)
    parser.add_argument('-vs', "--vocab_size", type=int, default=80,
                            help="Set this for text tasks. 80 for Shakespeare. 32000 for AG_News and SogouNews.")
    parser.add_argument('-ml', "--max_len", type=int, default=200)
    parser.add_argument('-fs', "--few_shot", type=int, default=0)
    parser.add_argument('-cdr', "--client_drop_rate", type=float, default=0.0,
                            help="Rate for clients that train but drop out")
    parser.add_argument('-tsr', "--train_slow_rate", type=float, default=0.0,
                            help="The rate for slow clients when training locally")
    parser.add_argument('-ssr', "--send_slow_rate", type=float, default=0.0,
                            help="The rate for slow clients when sending global model")
    parser.add_argument('-ts', "--time_select", type=bool, default=False,
                            help="Whether to group and select clients at each round according to time cost")
    parser.add_argument('-tth', "--time_threthold", type=float, default=10000,
                            help="The threthold for droping slow clients")
    parser.add_argument('-bt', "--beta", type=float, default=0.0)
    parser.add_argument('-lam', "--lamda", type=float, default=1.0,
                            help="Regularization weight")
    parser.add_argument('-mu', "--mu", type=float, default=0.0)
    parser.add_argument('-K', "--K", type=int, default=5,
                            help="Number of personalized training steps for pFedMe")
    parser.add_argument('-lrp', "--p_learning_rate", type=float, default=0.01,
                            help="personalized learning rate to caculate theta aproximately using K steps")
    parser.add_argument('-M', "--M", type=int, default=5,
                            help="Server only sends M client models to one client at each round")
    parser.add_argument('-itk', "--itk", type=int, default=4000,
                            help="The iterations for solving quadratic subproblems")
    parser.add_argument('-alk', "--alphaK", type=float, default=1.0,
                            help="lambda/sqrt(GLOABL-ITRATION) according to the paper")
    parser.add_argument('-sg', "--sigma", type=float, default=1.0)
    parser.add_argument('-al', "--alpha", type=float, default=1.0)
    parser.add_argument('-pls', "--plocal_epochs", type=int, default=1)
    parser.add_argument('-tau', "--tau", type=float, default=1.0)
    parser.add_argument('-fte', "--fine_tuning_epochs", type=int, default=10)
    parser.add_argument('-dlr', "--dr_learning_rate", type=float, default=0.0)
    parser.add_argument('-L', "--L", type=float, default=1.0)
    parser.add_argument('-nd', "--noise_dim", type=int, default=512)
    parser.add_argument('-glr', "--generator_learning_rate", type=float, default=0.005)
    parser.add_argument('-hd', "--hidden_dim", type=int, default=512)
    parser.add_argument('-se', "--server_epochs", type=int, default=1000)
    parser.add_argument('-lf', "--localize_feature_extractor", type=bool, default=False)
    parser.add_argument('-slr', "--server_learning_rate", type=float, default=1.0)
    parser.add_argument('-et', "--eta", type=float, default=1.0)
    parser.add_argument('-s', "--rand_percent", type=int, default=80)
    parser.add_argument('-p', "--layer_idx", type=int, default=2,
                            help="More fine-graind than its original paper.")
    parser.add_argument('-mlr', "--mentee_learning_rate", type=float, default=0.005)
    parser.add_argument('-Ts', "--T_start", type=float, default=0.95)
    parser.add_argument('-Te', "--T_end", type=float, default=0.98)
    parser.add_argument('-mo', "--momentum", type=float, default=0.1)
    parser.add_argument('-klw', "--kl_weight", type=float, default=0.0)
    parser.add_argument('-fsb', "--first_stage_bound", type=int, default=0)
    parser.add_argument('-ca', "--fedcross_alpha", type=float, default=0.99)
    parser.add_argument('-cmss', "--collaberative_model_select_strategy", type=int, default=1)
    parser.add_argument('-nsm', "--num_server_models", type=int, default=5,
                            help='Number of server models K for FedFew')
    parser.add_argument('-smu', "--smooth_mu", type=float, default=0.01,
                            help='Smoothing parameter μ')
    parser.add_argument('-rep', "--use_rep_mode", type=str2bool, default=False,
                            help='Use representation learning mode (base-head split) for FedFewS')
    parser.add_argument('-uc', "--use_cache", type=str2bool, default=True,
                            help='Enable data caching to avoid repeated file I/O (default: True)')
    return parser
