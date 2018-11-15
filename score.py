from utils import combine_cv
from os.path import join, exists, realpath, dirname
from os import system, makedirs
import argparse
import numpy as np

cwd = dirname(realpath(__file__))

def parse_args():
    parser = argparse.ArgumentParser(description="Launch a list of commands on EC2.")
    parser.add_argument("-o", "--outdir", required=True, help="")
    parser.add_argument("-c", "--mhc_class", required=True, help="")
    parser.add_argument("-g", "--gpu", default='0', help="")
    return parser.parse_args()

args = parse_args()

assert(args.mhc_class in ['1', '2'])

cwd = dirname(realpath(__file__))

template = 'CUDA_VISIBLE_DEVICES={} PYTHONPATH={} python {}/main.py -d {} -m {} -p {} -o {} -pw {} --pred_func {}'

pred_func = {'1':'embed', '2':'pred'}
models = {'1':'mhccat2pep_pepres_novar_v3_beta_rescale_eps1e-2', '2':'mhccat2pep_pepres_novar_v3_normal_noeps'}

for trial in range(10):
    for init in range(2):
        cmd = template.format(
                args.gpu,
                join('models', 'class'+args.mhc_class, models[args.mhc_class]),
                cwd,
                join('models', 'class'+args.mhc_class, 'trial{}'.format(trial+1)),
                models[args.mhc_class]+'_init{}'.format(init+1),
                join(args.outdir, 'data.h5.batch'),
                join(args.outdir, 'PUFFIN.trial{}.init{}'.format(trial+1, init+1)),
                'best',
                pred_func[args.mhc_class]
                )
        print(cmd)

combine_cv(
    args.outdir,
    range(10),
    range(2),
    args.mhc_class
)

