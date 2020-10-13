import argparse
import torch
import torch.nn.functional as F
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model as module_arch
from metrics.evaluate_tDCF_asvspoof19_func import evaluate_tdcf_eer, evaluate_eer
from parse_config import ConfigParser
from pathlib import Path
from collections import defaultdict
from functools import reduce
import numpy as np

torch.manual_seed(1234)  #cpu
torch.cuda.manual_seed(1234) #gpu
np.random.seed(1234) #numpy
# random.seed(1234) #random and transforms

torch.backends.cudnn.benchmark = True

# compute the err for the judge threshold
def compute_det_curve(target_scores, nontarget_scores):

    n_scores = target_scores.size + nontarget_scores.size
    all_scores = np.concatenate((target_scores, nontarget_scores))
    labels = np.concatenate((np.ones(target_scores.size), np.zeros(nontarget_scores.size)))
    
    # Sort labels based on scores
    indices = np.argsort(all_scores, kind='mergesort')
    labels = labels[indices]
    
    # Compute false rejection and false acceptance rates
    tar_trial_sums = np.cumsum(labels)
    nontarget_trial_sums = nontarget_scores.size - (np.arange(1, n_scores + 1) - tar_trial_sums)

    frr = np.concatenate((np.atleast_1d(0), tar_trial_sums / target_scores.size))  # false rejection rates
    far = np.concatenate((np.atleast_1d(1), nontarget_trial_sums / nontarget_scores.size))  # false acceptance rates
    thresholds = np.concatenate((np.atleast_1d(all_scores[indices[0]] - 0.001), all_scores[indices]))  # Thresholds are the sorted scores

    return frr, far, thresholds
def compute_eer(target_scores, nontarget_scores):
    """ Returns equal error rate (EER) and the corresponding threshold. """
    frr, far, thresholds = compute_det_curve(target_scores, nontarget_scores)
    abs_diffs = np.abs(frr - far)
    min_index = np.argmin(abs_diffs)
    eer = np.mean((frr[min_index], far[min_index]))
    return eer, thresholds[min_index]
    
    
def main(config, resume, protocol_file, asv_score_file):
    logger = config.get_logger('develop')

    # setup data_loader instances
    data_loader = getattr(module_data, config['interface_data_loader']['type'])(
        None,
        config['interface_data_loader']['args']['data_dir'],
        batch_size=128,
        shuffle=False,
        validation_split=0.0,
        num_workers=2,
        eval=True,
        read_protocol=True, 
        protocol_file=protocol_file
    )
    print(type(data_loader))
    print(type(data_loader.dataset))
    print(config['interface_data_loader']['args']['data_dir'])

    # build model architecture
    model = config.initialize('arch', module_arch)
    logger.info(model)

    # get function handles of loss and metrics
    # loss_fn = getattr(module_loss, config['loss'])
    loss_fn = config.initialize('loss', module_loss)
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    logger.info('Loading checkpoint: {} ...'.format(resume))
    checkpoint = torch.load(resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    model = model.to(device)
    model.eval()

    # utt2scores = defaultdict(list)
    utt2scores = defaultdict()

    # total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns))

    with torch.no_grad():
        for i, (utt_list, data, target) in enumerate(tqdm(data_loader)):
            data, target = data.to(device), target.to(device)
            output = model(data, eval=True)
            # loss = loss_fn(output, target)
            batch_size = data.shape[0]
            # total_loss += loss.item() * batch_size
            for i, metric in enumerate(metric_fns):
                total_metrics[i] += metric(output, target) * batch_size
            score = output[:, 1] # use the bonafide class for scoring
            # score = F.softmax(output, dim=1)[:, 1]
            # ======= #
            # loglikeli = F.log_softmax(output, dim=1)
            # score = loglikeli[:, 1] - loglikeli[:, 0]
            # ======= #
            
            for index, utt_id in enumerate(utt_list):
                utt2scores[utt_id] = score[index].item()
    
    ''' compute the err for the judge threshold, the threshold is -0.5947539210319519
    cm_score_file = Path(resume).parent / 'cm_score_dev.txt'
    # Load CM scores
    cm_data = np.genfromtxt(cm_score_file, dtype=str)
    cm_utt_id = cm_data[:, 0]
    cm_sources = cm_data[:, 1]
    cm_keys = cm_data[:, 2]
    cm_scores = cm_data[:, 3].astype(np.float)

    # Extract bona fide (real human) and spoof scores from the CM scores
    bona_cm = cm_scores[cm_keys == 'bonafide']
    spoof_cm = cm_scores[cm_keys == 'spoof']

    eer_cm, eer_point = compute_eer(bona_cm, spoof_cm)
    logger.info({"EER_CM": eer_cm, "EER_point": eer_point})
    '''
    threshold=-0.3
    for i in utt2scores:
        if utt2scores[i]>threshold:
            print(i+": bonafide - "+str(utt2scores[i]))
        else:
            print(i+": spoof - "+str(utt2scores[i]))
        
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ASVSpoof2019 project')

    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='path to latest checkpoint (default: None)')
    parser.add_argument('-f', '--protocol_file', default=None, type=str,
                        help='Protocol file: e.g., data/ASVspoof2019.PA.cm.dev.trl.txt')
    parser.add_argument('-a', '--asv_score_file', default=None, type=str,
                        help='Protocol file: e.g., data/ASVspoof2019_PA_dev_asv_scores_v1.txt')    
    parser.add_argument('-d', '--device', default=0, type=str,
                        help='indices of GPUs to enable (default: all)')
    

    args = parser.parse_args()
    config = ConfigParser(args)
    main(config, args.resume, args.protocol_file, args.asv_score_file)
