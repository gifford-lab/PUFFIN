from os.path import join, exists
from os import makedirs
import numpy as np
import h5py
from scipy.stats import norm, beta

def map_mhc(rawfile, outdir, dt, pseudo_seq_dict):
    alleles_not_recognized = set()
    with open(rawfile) as f, \
            open(join(outdir, dt+'.mhc'), 'w') as f1, \
            open(join(outdir, dt+'.pep'), 'w') as f2, \
            open(join(outdir, dt+'.label'), 'w') as f3:
            for idx, x in enumerate(f):
                line = x.split()
                prefix = '>mhc_seq' + str(idx) + '\t'
                if line[2] not in pseudo_seq_dict:
                    alleles_not_recognized.add(line[2])
                    continue
                f1.write(prefix + pseudo_seq_dict[line[2]]+'\n')
                f2.write(prefix + line[0]+'\n')
                f3.write(prefix + line[1]+'\n')

    if len(alleles_not_recognized)>0:
        print 'The following alleles are not recognized:', alleles_not_recognized

def padseq(file2pad, pad2len, padding = 'J', padded_suffix='.padded'):
    with open(file2pad) as fin, open(file2pad+padded_suffix, 'w') as fout:
        for idx, x in enumerate(fin):
            line = x.split()
            seq = list(line[1])
	    assert(len(seq)<=pad2len)
            fout.write(line[0] + '\t' + ''.join(seq + [padding]*(pad2len - len(seq))) +'\n')

def cal_mean(params, mhc_class, eps=0.01):
    return params[:, :, 0] if mhc_class=='2' else (params[:, :, 0] / (params[:, :, 0] + params[:, :, 1]) - eps) / (1-2*eps)

def cal_var(params, mhc_class, eps=0.01):
    if mhc_class=='2':
        return params[:, :, 1]**2
    else:
        alpha = params[:, :, 0]
        beta = params[:, :, 1]
        return alpha * beta / (alpha+beta)**2 / (alpha+beta+1) /((1-2*eps)**2)

def binding_likelihood(params, mhc_class, thresh=1-np.log(500)/np.log(50000), eps=0.01):
    if mhc_class=='2':
    	return 1 - norm.cdf(thresh, loc=params[:, 0], scale=params[:, 1])
    else:
    	return 1 - beta.cdf(eps+(1-2*eps)*thresh, params[:, 0], params[:, 1])

def combine_cv(datadir, trials, inits, mhc_class):
    models2combine = [
        join(datadir, 'PUFFIN.trial{}.init{}'.format(trial+1, init+1)) \
        for init in inits for trial in trials]

    combine_pred = join(datadir, 'PUFFIN.combined')

    allpred = []
    for predir in models2combine:
        batch=1
        if not exists(join(predir, 'h5.batch'+str(batch))):
            print 'file doesn\'t exist! {}'.format(join(predir, 'h5.batch'+str(batch)))
            continue
            #assert(False)
        while exists(join(predir, 'h5.batch'+str(batch))):
            with h5py.File(join(predir, 'h5.batch'+str(batch)), 'r') as f:
                newdata = f['pred'][()]
                if np.sum(np.isnan(newdata)) > 0:
                    print batch, predir, np.sum(np.isnan(newdata))
                pred = np.vstack((
                    pred, newdata)) if batch > 1 else newdata
            batch += 1
        allpred.append(pred)

    allpred = np.asarray(allpred)

    allmean = cal_mean(allpred, mhc_class)
    allvar = cal_var(allpred, mhc_class)
    allstd = np.sqrt(allvar)

    mean_pred = np.mean(allmean, axis=0)
    epistemic = np.var(allmean, axis=0)
    aleatoric = np.mean(allstd, axis=0)**2
    bl = binding_likelihood(np.mean(allpred, axis=0), mhc_class)

    out = np.vstack((mean_pred, epistemic, aleatoric, bl)).transpose()

    with open(combine_pred, 'w') as f:
        f.write('{}\n'.format('\t'.join(['mean_pred', 'epistemic_var', 'aleatoric_var', 'binding_likelihood'])))
	for x in out:
            f.write('{}\n'.format('\t'.join(map(str, x))))
