from utils import map_mhc, padseq
from os.path import join, exists, realpath, dirname
from os import system, makedirs
import argparse

cwd = dirname(realpath(__file__))

def parse_args():
    parser = argparse.ArgumentParser(description="Launch a list of commands on EC2.")
    parser.add_argument("-i", "--inputfile", required=True, help="")
    parser.add_argument("-o", "--outdir", required=True, help="")
    parser.add_argument("-c", "--mhc_class", required=True, help="")
    return parser.parse_args()

args = parse_args()

assert(args.mhc_class in ['1', '2'])

pseudo_seq_file = join(cwd, 'data', 'pseudosequence.2016.all.X.dat') if args.mhc_class=='2' else join(cwd, 'data', 'MHC_pseudo.dat')
pad2len = 40 if args.mhc_class=='2' else 30
expected_len = 15 if args.mhc_class=='2' else 9
mapperfile = join(cwd, 'data', 'onehot_first20BLOSUM50')

if not exists(args.outdir):
    makedirs(args.outdir)

pseudo_seq_dict = dict()
with open(pseudo_seq_file) as f:
    for x in f:
        line = x.split()
        pseudo_seq_dict[line[0]] = line[1]

map_mhc(
    args.inputfile,
    args.outdir,
    'data',
    pseudo_seq_dict,
)

padseq(
    join(args.outdir, 'data.pep'),
    pad2len=pad2len,
)

cmd = 'python {}/embed.py --mhcfile {} --pepfile {} --labelfile {}  --mapper {} --outfileprefix  {} ' + \
'--expected_pep_len {}'

cmd = cmd.format(
        cwd,
        join(args.outdir, 'data.mhc'),
        join(args.outdir, 'data.pep.padded'),
        join(args.outdir, 'data.label'),
        mapperfile,
        join(args.outdir, 'data.h5.batch'),
        expected_len,
        )

system(cmd)

