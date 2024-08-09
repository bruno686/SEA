import wandb
import os,sys
os.chdir(sys.path[0])
import yaml
import argparse
from utils.quick_start import quick_start
os.environ['NUMEXPR_MAX_THREADS'] = '48'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='SEA', help='name of models')
    parser.add_argument('--dataset', '-d', type=str, default='baby', help='name of datasets')
    parser.add_argument('--mg', action="store_true", default=False, help='whether to use Mirror Gradient, default is False')
    parser.add_argument('--alpha_contrast', type=float, default=0.012)
    parser.add_argument('--temp', type=float, default=0.046)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--out_dim', type=int, default=32)
    parser.add_argument('--reg_weight', type=float, default=0)
    parser.add_argument('--ureg', type=float, default=0.006)
    parser.add_argument('--gpu', type=int, default=0)
    args, _ = parser.parse_known_args()

    config_dict = {
        'gpu_id': args.gpu,
    }
    wandb.login
    wandb.init(project="MM_OOD", config=args, notes='参数分析', mode='online')

    quick_start(model=args.model, dataset=args.dataset, config_dict=config_dict, args=args,  save_model=True, mg=args.mg)


