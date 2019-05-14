from os.path import join, exists
from os import system, makedirs

topdirs = ['/cluster/zeng/code/research/MHCpeptide/data/classI/dataset/netMHCpan3.0_ge100_reg_onehotT20B50/CValldata',
        '/cluster/zeng/code/research/MHCpeptide/data/IEDB/dataset/NetMHCIIpan-3.2_ge100_reg_onehotT20B50/CValldata']

models = ['mhccat2pep_pepres_novar_v3_beta_rescale_eps1e-2', 'mhccat2pep_pepres_novar_v3_normal_noeps']
outdir = 'models'

for mhc_class in range(2):
    for trial in range(10):
        t_outdir = join(outdir, 'class{}'.format(mhc_class+1), 'trial{}'.format(trial+1))
        if exists(t_outdir):
            system('rm -r '+t_outdir)
        makedirs(t_outdir)

        for init in range(2):
            t_t_outdir = join(t_outdir, models[mhc_class]+'_init{}'.format(init+1))
            makedirs(t_t_outdir)
            cmd = 'cp -r {} {}'.format(
                    join(topdirs[mhc_class], 'trial{}'.format(trial+1), models[mhc_class]+'_init{}'.format(init+1), 'best*'),
                    t_t_outdir
                    )
            system(cmd)

    system(' '.join([
        'cp -r',
        '../models/'+models[mhc_class],
        join(outdir, 'class{}'.format(mhc_class+1))
        ]))
    system(' '.join([
        'cp -r',
        '../models/resnet.py',
        join(outdir, 'class{}'.format(mhc_class+1))
        ]))
