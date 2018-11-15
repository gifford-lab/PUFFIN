from __future__ import print_function
import time,numpy as np,sys,h5py,cPickle,argparse,json, pandas as pd, shutil, os
from os.path import join,dirname,basename,exists,realpath
from os import system,chdir,getcwd,makedirs, listdir
from tempfile import mkdtemp
from sklearn.metrics import accuracy_score,roc_auc_score
from pprint import pprint
from time import time

import ray
from ray.tune import Trainable, TrainingResult, register_trainable, \
    run_experiments
from ray.tune.hyperband import HyperBandScheduler
from torch.utils.data import DataLoader

cwd = dirname(realpath(__file__))

def parse_args():
    parser = argparse.ArgumentParser(description="Keras + Hyperband for genomics")
    parser.add_argument("-y", "--hyper", dest="hyper", default=False, action='store_true',help="Perform hyper-parameter tuning")
    parser.add_argument("-f", "--findbest", default=False, action='store_true',help="Identify the best hyper-parameter combination")
    parser.add_argument("-t", "--train", dest="train", default=False, action='store_true',help="Train on the training set with the best hyper-params")
    parser.add_argument("-e", "--eval", dest="eval", default=False, action='store_true',help="Evaluate the model on the test set")
    parser.add_argument("-pb", "--pred_batch", dest="pred_batch", default='', help="Evaluate the model on the test set")
    parser.add_argument("-pbo", "--pred_batch_outdir", default='', help="Evaluate the model on the test set")
    parser.add_argument("-p", "--predit", dest="infile", default='', help="Path to data to predict on (up till batch number)")
    parser.add_argument("-pw", "--preditwith", dest="predictwith", default='best', help="the checkpoint to predict with, best or last")
    parser.add_argument("-d", "--topdir", dest="topdir", required=True, help="The data directory")
    parser.add_argument("-m", "--model", dest="model", default='', help="name of the model")
    parser.add_argument("-o", "--outdir", dest="outdir",default='',help="Output directory for the prediction on new data")
    parser.add_argument("-hi", "--hyperiter", dest="hyperiter", default=20, type=int, help="Num of max iteration for each hyper-param config")
    parser.add_argument("-te", "--trainepoch", default=20, type=int, help="The number of epochs to train for")
    parser.add_argument("-pa", "--patience", default=10000, type=int, help="number of epochs with no improvement after which training will be stopped.")
    parser.add_argument("-bs", "--batchsize", default=100, type=int,help="Batchsize in SGD-based training")
    parser.add_argument("-w", "--weightfile", default=None, help="Weight file for the best model")
    parser.add_argument("-l", "--lweightfile", default=None, help="Weight file after training")
    parser.add_argument("-r", "--retrain", default=False, action='store_true', help="codename for the retrain run")
    parser.add_argument("-rw", "--rweightfile", default='', help="Weight file to load for retraining")
    parser.add_argument("-dm", "--datamode", default='memory', help="whether to load data into memory ('memory') or using a generator('generator')")
    parser.add_argument("-ei", "--evalidx", dest='evalidx', default=0, type=int, help="which output neuron (0-based) to calculate 2-class auROC for")
    parser.add_argument("--tunemethod", default="random")
    parser.add_argument("--metric", default="loss")
    parser.add_argument("--metric_weight", default=1, type=float)
    parser.add_argument("--pred_func", default='pred', type=str)
    parser.add_argument("--pred_save_as", default='h5', type=str)
    parser.add_argument("--pred_h5_batchsize", default=50000, type=int)

    return parser.parse_args()

def load_model(mode='pred', restore=True, pred_prefix=None, restore_from='best'):
    with open(join(best_trial_dir, 'params.json')) as f:
        best_config = json.loads(f.readline())
    print('Best config:', best_config)

    for key, item in mymodel.get_setup()['config'].iteritems():
        if key not in best_config.keys():
            best_config[key] = item

    best_config['mode'] = mode
    if pred_prefix:
        best_config['testset_prefix'] = pred_prefix

    model = mymodel.MyTrainableClass(config=best_config)
    if restore:
        restore_dir = best_model_dir if restore_from == 'best' else last_model_dir
        print('restore model from', restore_dir)
        model._restore(join(restore_dir, 'checkpoint.pt'))

    return model


if __name__ == "__main__":

    args = parse_args()
    if not(os.environ['PYTHONPATH'] or args.model):
        print('Provide a model name or set a valid PYTHONPATH to the model.py file')
        sys.exit(1)
    else:
        model_arch = args.model or basename(os.environ['PYTHONPATH'])

    model_arch = model_arch[:-3] if model_arch[-3:] == '.py' else model_arch

    outdir = join(args.topdir, model_arch)
    if not exists(outdir):
        makedirs(outdir)

    best_model_dir = join(outdir, 'best_model')
    last_model_dir = join(outdir, 'last_model')
    evalout = join(outdir,  'best_model_eval.txt')
    tune_dir = realpath(join(outdir, 'ray_tune_log'))
    best_trial_dir = realpath(join(outdir, 'best_trial_'+args.tunemethod))
    historyfile = realpath(join(outdir, 'train.log'))

    print('Using model.py under {}'.format(realpath(os.environ['PYTHONPATH'])))
    import model as mymodel


    if args.hyper:
        shutil.copy(join(os.environ['PYTHONPATH'], 'model.py'), join(outdir, 'tune_model.py'))

        # Hyper-param tuning
        register_trainable("my_class", mymodel.MyTrainableClass)
        ray.init()

        model_config = mymodel.get_setup()
        for dt in ['train', 'valid', 'test']:
            model_config['config'][dt+'set_prefix'] = realpath(join(args.topdir, dt+'.h5.batch'))
        model_config['local_dir'] = tune_dir
        print('model_config:', model_config)

        if exists(tune_dir):
            shutil.rmtree(tune_dir)
        makedirs(tune_dir)

        with open(join(outdir, 'hypertune.args'), 'w') as f:
            for k,v in vars(args).items():
                f.write('{}\t{}\n'.format(k,v))

        if args.tunemethod == 'hyperband':
            hyperband = HyperBandScheduler(
                time_attr="timesteps_total", reward_attr="neg_mean_loss",
                max_t=args.hyperiter)
            run_experiments({args.tunemethod: model_config}, scheduler=hyperband)
        else:
            model_config['stop'] = {"training_iteration": args.hyperiter}
            run_experiments({args.tunemethod: model_config})

    if args.findbest:
        # Determine the best hyper-parmas
        best_loss = -1
        tune_dir_method = join(tune_dir, args.tunemethod)

        for trial in listdir(tune_dir_method):
            result_file = join(tune_dir_method, trial, 'result.json')
            if exists(result_file):
                with open(result_file) as f:
                    t_loss = [json.loads(x)['mean_loss'] for x in f]
                    if len(t_loss)==0:
                        continue
                    t_best_loss = min(t_loss)
                    if t_best_loss < best_loss or best_loss == -1:
                        best_loss = t_best_loss
                        best_trial = trial

        if exists(best_trial_dir):
            shutil.rmtree(best_trial_dir)
        shutil.copytree(join(tune_dir_method, best_trial, ''), best_trial_dir)

        with open(join(tune_dir_method, best_trial, 'params.json')) as f:
            best_config = json.loads(f.readline())
        print('Best config:', best_config)


    if args.train or args.retrain:
        shutil.copy(join(os.environ['PYTHONPATH'], 'model.py'), join(outdir, 'train_model.py'))

        # Training with the best hyper-params
        if args.train:
            model = load_model(mode='train', restore=False)
            if exists(best_model_dir):
                shutil.rmtree(best_model_dir)
            makedirs(best_model_dir)
            if not exists(last_model_dir):
                makedirs(last_model_dir)
        else:
            model = load_model(mode='train', restore=True, restore_from='last')

        with open(join(outdir, 'train.args'), 'w') as f:
            for k,v in vars(args).items():
                f.write('{}\t{}\n'.format(k,v))

        if args.retrain:
            hist = pd.read_csv(historyfile, sep='\t')
            best_metric = min(np.asarray(hist[args.metric]) * args.metric_weight)
            print('previously best val {}: {}'.format(args.metric, best_metric))

        history = []
        pa_cnt = 0
        for idx in range(args.trainepoch):
            t_train_metrics, t_valid_metrics = model._train()
            valid_metric2use = t_valid_metrics[args.metric] * args.metric_weight
            pa_cnt += 1
            msg = 'Epoch {}:'.format(idx)
            msg += ' mean train '
            for k, t in t_train_metrics.items():
                msg += ' {0}:{1:.3f} '.format(k, t)
            msg += ' mean valid '
            for k, t in t_valid_metrics.items():
                msg += ' {0}:{1:.3f} '.format(k, t)
            print(msg)

            if idx==0 or best_metric > valid_metric2use:
                best_metric = valid_metric2use
                model._save(best_model_dir)
                print('Best epoch so far based on valid {}!'.format(args.metric))
                pa_cnt = 0

            history.append([idx+1] + t_train_metrics.values() + t_valid_metrics.values())
            model._save(last_model_dir)
            if pa_cnt >= args.patience:
                print('No progress after {} epoch. Early stopping!'.format(args.patience))
                break

        # Correct epoch count if retrain
        if args.retrain:
            for i in range(len(history)):
                history[i][0] += int(list(hist['Epoch'])[-1])

        write_mode = 'w' if args.train else 'a'
        pd.DataFrame(history, columns=['Epoch'] + t_train_metrics.keys() + t_valid_metrics.keys()).to_csv(historyfile, sep='\t', index=False, mode=write_mode, header=args.train)


    if args.eval:
        ## Evaluate
        model = load_model(mode='eval', restore_from=args.predictwith)

        pred_test = model._pred(model.testset)

        cnt = 1
        prefix = join(args.topdir, 'test.h5.batch')
        while exists(prefix+str(cnt)):
            dataall = h5py.File(prefix+str(cnt),'r')
            if cnt == 1:
                label = dataall['label'][()]
            else:
                label = np.vstack((label, dataall['label'][()]))
            cnt += 1

        t_auc = roc_auc_score([x[args.evalidx] for x in label], [x[args.evalidx] for x in pred_test ])
        t_acc = accuracy_score([np.argmax(x) for x in label], [np.argmax(x) for x in pred_test])
        print('Test AUC for output neuron {}:'.format(args.evalidx), t_auc)
        print('Test categorical accuracy:', t_acc)
        np.savetxt(evalout, [t_auc, t_acc])


    if args.infile != '':
        ## Predict on new data
        model = load_model(mode='pred', pred_prefix=args.infile, restore_from=args.predictwith)

        outdir = join(dirname(args.infile), '.'.join(['pred', model_arch, basename(args.infile)])) if args.outdir == '' else args.outdir
        if exists(outdir):
            print('Output directory', outdir, 'exists! Overwrite? (yes/no)')
            if raw_input().lower() == 'yes':
                shutil.rmtree(outdir)
            else:
                print('Quit predicting!')
                sys.exit(1)
        makedirs(outdir)

        pred = model._pred(model.testset) if args.pred_func=='pred' else model._embed(model.testset)

        assert(args.pred_save_as in ['h5', 'pickle'])
        if args.pred_save_as == 'pickle':
            for label_dim in range(pred.shape[1]):
                with open(join(outdir, str(label_dim)+'.pkl'), 'wb') as f:
                    cPickle.dump(pred[:, label_dim], f)
        else:
            for idx, pos in enumerate(range(0, len(pred), args.pred_h5_batchsize)):
                with h5py.File(join(outdir, 'h5.batch'+str(idx+1)), 'w') as f:
                    f.create_dataset('pred', data=pred[pos:min(len(pred), pos+args.pred_h5_batchsize)])


    if args.pred_batch != '':
        with open(args.pred_batch) as f:
            files2pred = [x.strip() for x in f]

        assert(args.pred_batch_outdir!='')

        model = load_model(mode='pred', restore_from=args.predictwith)


        for file2pred in files2pred:
            start = time()
            testset = mymodel.HDF5_dataset(file2pred, batch=False, haslabel=False)
            model.testset = DataLoader(testset, batch_size=args.pred_h5_batchsize, shuffle=False, num_workers=2)
            print(file2pred)

            outdir = join(args.pred_batch_outdir, basename(file2pred))
            if exists(outdir):
                system('rm -r '+outdir)
            makedirs(outdir)

            for i, data in enumerate(model.testset, 0):
                pred = model._pred_all(data) if args.pred_func=='pred' else model._embed_all(data)
                with h5py.File(join(outdir, 'h5.batch'+str(i+1)), 'w') as f:
                    f.create_dataset('pred', data=pred)
            print('time elapsed:', time()-start)

    #system('rm -r ' + tmpdir)
