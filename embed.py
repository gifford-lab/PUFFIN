import h5py, argparse
from itertools import izip
from os.path import join
import numpy as np

def outputHDF5(mhc, pep, peplen, label, filename):
    mhc = np.asarray(mhc)
    pep = np.asarray(pep)
    print 'mhc size:', mhc.shape
    print 'pep size:', pep.shape
    assert(len(mhc) == len(pep))
    assert(len(mhc) == len(label))
    comp_kwargs = {'compression': 'gzip', 'compression_opts': 4}
    with h5py.File(filename, 'w') as f:
        f.create_dataset('mhc', data=mhc, **comp_kwargs)
        f.create_dataset('pep', data=pep, **comp_kwargs)
        f.create_dataset('label', data=label, **comp_kwargs)
        f.create_dataset('peplen', data=peplen, **comp_kwargs)

def embed(seq, mapper):
    return np.asarray([mapper[s] for s in seq]).transpose()

def lenpep_feature(pep):
    lenpep = len(pep) - pep.count('J')
    f1 = 1.0/(1.0 + np.exp((lenpep-args.expected_pep_len)/2.0))
    return f1, 1.0-f1

def embed_all(mhc_f, pep_f, label_f, mapper, outfile_prefix, bs=50000):
    mhc = []
    pep = []
    label = []
    peplen = []

    cnt = 0
    bs_cnt = 0
    for mhc_line, pep_line, label_line in izip(mhc_f, pep_f, label_f):
        mhc.append(embed(mhc_line.split()[1], mapper))
        pep.append(embed(pep_line.split()[1], mapper))
        peplen.append(lenpep_feature(pep_line.split()[1]))
        label.append(map(float, label_line.split()[1:]))
        if (cnt+1) % bs == 0:
            bs_cnt += 1
            outputHDF5(mhc, pep, peplen, label, outfile_prefix+str(bs_cnt))
            mhc = []
            pep = []
            label=  []
            peplen = []
        cnt += 1

    if len(mhc) > 0:
        bs_cnt += 1
        outputHDF5(mhc, pep, peplen, label, outfile_prefix+str(bs_cnt))

def parse_args():
    parser = argparse.ArgumentParser(description="Convert sequence and target for Caffe")

    # Positional (unnamed) arguments:
    parser.add_argument("--mhcfile",  type=str, help="Sequence in FASTA/TSV format (with .fa/.fasta or .tsv extension)")
    parser.add_argument("--pepfile",  type=str,help="Label of the sequence. One number per line")
    parser.add_argument("--labelfile",  type=str, help="Output file (example: $MODEL_TOPDIR$/data/train.h5). ")
    parser.add_argument("--outfileprefix",  type=str, help="Output file (example: $MODEL_TOPDIR$/data/train.h5). ")
    parser.add_argument("--mapper",  type=str, help="Output file (example: $MODEL_TOPDIR$/data/train.h5). ")
    parser.add_argument("--expected_pep_len",  type=int, help="Output file (example: $MODEL_TOPDIR$/data/train.h5). ")
    return parser.parse_args()

args = parse_args()

with open(args.mapper) as f:
    mapper = dict()
    for x in f:
        line = x.split()
        mapper[line[0]] = map(float, x.split()[1:])

with open(args.mhcfile) as f1, open(args.pepfile) as f2, open(args.labelfile) as f3:
    embed_all(f1, f2, f3, mapper, args.outfileprefix)
