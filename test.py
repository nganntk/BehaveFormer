"""Main test code"""
import os
import random
import logging
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from model.behaveformer import BehaveFormer
from model.dataset import HUMITestDataset, FETATestDataset
from evaluation.metrics import Metric
from utils.config import Config
from utils.utils import read_pickle


def set_random_seeds(seed: int):
    """
    Set all the random seeds to a fixed value for reproducibility.

    Args:
        seed (int): the seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def create_logger(log_dir: str) -> logging.Logger:
    """Create logger to output to: <log_dir>/training.log

    Args:
        log_dir (str): Folder to save training.log

    Returns:
        logging.Logger: created logger
    """    
    logger = logging.getLogger()
    logging.getLogger('matplotlib.font_manager').disabled = True
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(os.path.join(log_dir, 'test.log'))
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    return logger   

def get_evaluate_results(feature_embeddings, num_enroll_sessions, num_verify_sessions=None, num_users=None, user_session_count=None, dataset=''):
    """Get all evaluation results

    Args:
        feature_embeddings (torch.Tensor): Feature embeddings - output of BehaveFormer
        num_enroll_sessions (int): Number of enrollment sessions
        num_verify_sessions (int, optional): For HuMidb with fixed number of verify sessions. Defaults to None.
        num_users (int, optional): Number of users for FETA. Defaults to None.
        user_session_count (_type_, optional): Number of sessions per user for FETA with varying number of sessions per user. Defaults to None.
        dataset (str, optional): 'humi' or 'feta', used to determine how to calculate EER. Defaults to ''.

    Raises:
        NotImplementedError: _description_

    Returns:
        _type_: _description_
    """    
    _acc = []
    _usability = []
    _tcr = []
    _fawi = []
    _frwi = []
    if dataset == 'humi':
        for i in range(feature_embeddings.shape[0]): 
            all_ver_embeddings = torch.cat([feature_embeddings[i,num_enroll_sessions:], torch.flatten(feature_embeddings[:i,num_enroll_sessions:], start_dim=0, end_dim=1), torch.flatten(feature_embeddings[i+1:,num_enroll_sessions:], start_dim=0, end_dim=1)], dim=0)
            scores = Metric.cal_session_distance_fixed_sessions(all_ver_embeddings, feature_embeddings[i,:num_enroll_sessions])
            periods = get_periods(i, num_enroll_sessions, num_verify_sessions)   #### use num_verify_sessions for period & also skip num_enroll_sessions
            labels = torch.tensor([1] * num_verify_sessions + [0] * (feature_embeddings.shape[0] - 1) * num_verify_sessions)
            acc, threshold = Metric.eer_compute(scores[:num_verify_sessions], scores[num_verify_sessions:])

            usability = Metric.calculate_usability(scores, threshold, periods, labels)
            tcr = Metric.calculate_TCR(scores, threshold, periods, labels)
            frwi = Metric.calculate_FRWI(scores, threshold, periods, labels)
            fawi = Metric.calculate_FAWI(scores, threshold, periods, labels)
            _acc.append(acc)
            _usability.append(usability)
            _tcr.append(tcr)
            _fawi.append(fawi)
            _frwi.append(frwi)
            
        return 100 - np.mean(_acc, axis=0), np.mean(_usability, axis=0), np.mean(_tcr, axis=0), np.mean(_frwi, axis=0) , np.mean(_fawi, axis=0)
    
    elif dataset == 'feta':
        for i in range(num_users):
            start_idx, num_sessions = user_session_count[i]  # different user has different number of session
            
            # Get current user data
            current_user = feature_embeddings[start_idx:(start_idx+num_sessions)]
            enroll_sess = current_user[:num_enroll_sessions]

            # Get verifiation sessions of the current user (genuine)
            verify_sess = current_user[num_enroll_sessions:]
            num_verify_sessions = len(verify_sess)  # number of verify sessions varied depends on the user

            # Get sessions of all other users (imposter)
            other_user = []
            for j in range(num_users):
                if j != i:
                    start_idx_j, num_sessions_j = user_session_count[j]
                    user_j = feature_embeddings[start_idx_j:(start_idx_j + num_sessions_j)]
                    other_user.append(user_j[num_enroll_sessions:])
            all_ver_embeddings = torch.cat([verify_sess] + other_user, dim=0)
            
            scores = Metric.cal_session_distance_vary_sessions(all_ver_embeddings, enroll_sess)
            acc, threshold = Metric.eer_compute(scores[:num_verify_sessions], scores[num_verify_sessions:])  # score_g, score_i

            periods = get_periods(i, num_enroll_sessions, user_session_count=user_session_count, dataset='feta')   #### use num_verify_sessions for period & also skip num_enroll_sessions
            labels = torch.tensor([1] * num_verify_sessions + [0] * (scores.shape[0] - num_verify_sessions))
            acc, threshold = Metric.eer_compute(scores[:num_verify_sessions], scores[num_verify_sessions:])

            usability = Metric.calculate_usability(scores, threshold, periods, labels)
            tcr = Metric.calculate_TCR(scores, threshold, periods, labels)
            frwi = Metric.calculate_FRWI(scores, threshold, periods, labels)
            fawi = Metric.calculate_FAWI(scores, threshold, periods, labels)
            _acc.append(acc)
            _usability.append(usability)
            _tcr.append(tcr)
            _fawi.append(fawi)
            _frwi.append(frwi)
            
        return 100 - np.mean(_acc, axis=0), np.mean(_usability, axis=0), np.mean(_tcr, axis=0), np.mean(_frwi, axis=0), np.mean(_fawi, axis=0)

    else:
        raise NotImplementedError

def get_periods(user_id, num_enroll_sess, num_verify_sess=None, user_session_count=None, dataset='humi'):
    def get_window_time_humi(seqs):
        seq = seqs[0][0]
        start = seq[0][0]
        end = seq[-1][0]
        
        i = -1
        while (end == 0):  # handle zero padding at the end
            end = seq[i-1][0]
            i = i - 1
        return (end - start) / 1000

    def get_window_time_feta(seqs):
        seq = seqs[0].numpy()
        start = seq[0][0]
        end = seq[-1][0]
        
        i = -1
        while (end == 0):  # handle zero padding at the end
            end = seq[i-1][0]
            i = i - 1
        return (end - start) / 1000

    if dataset == 'humi':    
        # Get 2 period from the same user and 2 from different user
        periods = []
        for j in range(num_verify_sess):
            periods.append(get_window_time_humi(test_dataset.data[user_id][num_enroll_sess + j]))
        for i in range(len(test_dataset.data)):
            if (i != user_id):
                for j in range(num_verify_sess):
                    periods.append(get_window_time_humi(test_dataset.data[i][num_enroll_sess + j]))
        return periods
    elif dataset == 'feta':
        _, current_user_num_sessions = user_session_count[user_id]
        current_user_num_verify = current_user_num_sessions - num_enroll_sess

        # Get 2 period from the same user and 2 from different usertm
        periods = []
        for j in range(current_user_num_verify):
            tmp = test_dataset.load_data_with_time(user_id, num_enroll_sess + j, 0)  # first sequence, same as BehaveFormer
            periods.append(get_window_time_feta(tmp))  # tmp: scroll, imu or scroll only
        for i in range(len(user_session_count)):
            if (i != user_id):
                _, user_num_sess_i = user_session_count[i]
                user_num_verify_i = user_num_sess_i - num_enroll_sess
                for j in range(user_num_verify_i):
                    tmp = test_dataset.load_data_with_time(i, num_enroll_sess + j, 0)  # first 
                    periods.append(get_window_time_feta(tmp))
        return periods

def get_args():
    """Argument parser for test code"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataname', help='Dataset name', choices=['humi', 'feta'], required=True)
    parser.add_argument('-c', '--config', help="Config file", required=True)
    parser.add_argument('--metric', help="Metric used for test, options: basic, det, pca", nargs='+', default='basic')
    parser.add_argument('--trained_weights_pt', help="Trained weights in .pt file use for testing")
    parser.add_argument('--test_pickle', help="Test processed data. If not passed, use file in the config", default='')
    parser.add_argument('--work_dirs_subfolder', help="Subfolder in work_dirs, used for ablation study to not clutter main results folder", default='')
    return parser.parse_args()

if __name__ == "__main__":
    # Get input argument
    args = get_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"INFO: Input argument: {str(args)}")
    dataname = args.dataname
    metric_list = args.metric
    print(metric_list)

    # Get output folder from config filename
    config_data = Config(path=args.config).get_config_dict()
    config_name = Path(args.config).stem
    config_output_dir = os.path.join('work_dirs', args.work_dirs_subfolder, config_name)
    latest_timestamp = sorted(os.listdir(config_output_dir))[-1]
    work_dir = os.path.join(config_output_dir, latest_timestamp)
    results_path = os.path.join(work_dir, 'results', timestamp)
    os.makedirs(results_path)

    if(config_data["GPU"] == "True"):
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    
    # Read hyperparamters
    scroll_sequence_len = config_data['data']['scroll_sequence_len']
    imu_sequence_len = config_data['data']['imu_sequence_len']
    action_type = config_data['data']['action_type']

    hyperparams = config_data['hyperparams']
    batch_size = hyperparams['batch_size']
    target_len = hyperparams['target_len']
    gre_k = hyperparams['gre_k']
    scroll_feature_dim = hyperparams['scroll_feature_dim']
    behave_temporal_heads = hyperparams['scroll_temporal_heads']
    behave_channel_heads = hyperparams['scroll_channel_heads']
    imu_temporal_heads = hyperparams['imu_temporal_heads']
    imu_channel_heads = hyperparams['imu_channel_heads']
    imu_type = hyperparams['imu_type']
    if imu_type == 'none':
        assert hyperparams['num_imu'] == 0, "Check config file, num_imu must be 0 when imu_type is none"
    else:
        assert len(imu_type.split('_')) == hyperparams['num_imu'], "Check config file. imu_type and num_imu do not matched"
    imu_feature_dim = hyperparams['num_imu'] * 12
    number_of_enrollment_sessions = hyperparams['number_of_enrollment_sessions']
    print('number_of_enrollment_sessions', number_of_enrollment_sessions)
    num_verify_sessions = hyperparams['number_of_verify_sessions'] if dataname == 'humi' else None  # If None, determined based on number of sessions for each user

    # Set logging
    logger = create_logger(results_path)
    logger.info(f"Input argument: {str(args)}\n")

    if dataname == 'humi':
        splits = read_pickle(os.path.join(config_data['folders']['data_dir'], 'splits.pickle'))
        logger.info(f"#testing users {len(splits['testing'])}")

        test_dataset = HUMITestDataset(action=action_type, 
                                       validation_file=os.path.join(config_data['folders']['data_dir'], 'testing_scroll_imu_data_all.pickle'),
                                       imu_type=imu_type)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    elif dataname == 'feta':
        splits = read_pickle(os.path.join(config_data['folders']['data_dir'], 'splits.pickle'))
        logger.info(f"#testing users {len(splits['testing'])}")

        test_dataset = FETATestDataset(data_root=os.path.join(config_data['folders']['data_dir'], 'processed'),
                                       action=action_type,
                                       user_list=splits['testing'],
                                       imu_type=imu_type)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4)  

    # Create model and load weights
    model = BehaveFormer(scroll_feature_dim, imu_feature_dim, scroll_sequence_len, imu_sequence_len, target_len, gre_k, behave_temporal_heads, behave_channel_heads, imu_temporal_heads, imu_channel_heads, imu_type=imu_type)
    if args.trained_weights_pt is None:
        best_epoch = max([int(i.split('_')[1]) for i in os.listdir(os.path.join(work_dir, 'best_models'))])
        tmp = [i for i in os.listdir(os.path.join(work_dir, 'best_models')) if i.startswith(f'epoch_{best_epoch}_eer_')]
        assert len(tmp) == 1
        args.trained_weights_pt = os.path.join(work_dir, 'best_models', tmp[0])
        print("Taking the weights from the last epoch in best_models folder:", tmp[0])
    if (config_data["GPU"] == "True"):
        model.load_state_dict(torch.load(args.trained_weights_pt))
    else:
        model.load_state_dict(torch.load(args.trained_weights_pt, map_location=torch.device('cpu')))
    
    # Log weights, config
    logger.info(f"CONFIG: {os.path.basename(args.config).split('.')[0]}")
    logger.info(f"WEIGHTS: {os.path.basename(args.trained_weights_pt)}")

    # Get feature embeddings
    model.train(False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        feature_embeddings = []
        for batch_idx, item in enumerate(test_dataloader):
            # print(item[0].shape, item[1].shape)  # (64, 50, 8), (64, 100, 36)
            if imu_type != 'none':
                feature_embeddings.append(model([item[0].to(device).float(), item[1].to(device).float()]))
            else:
                feature_embeddings.append(model(item[0].to(device).float()))

    if dataname == 'humi':
        if "basic" in args.metric:
            res = get_evaluate_results(torch.cat(feature_embeddings, dim=0).view(test_dataset.num_users, test_dataset.num_sessions, test_dataset.num_seqs, target_len), number_of_enrollment_sessions, num_verify_sessions, dataset=dataname)
            res_df = pd.DataFrame([list(res)], columns=["eer", "usability", "tcr", "frwi", "fawi"])
            logger.info(f"\nRESULTS\n{res_df}")
            res_df.to_csv(f"{results_path}/basic.csv", index=False)
        if "det" in args.metric:
            Metric.save_DET_curve(torch.cat(feature_embeddings, dim=0).view(test_dataset.num_users, test_dataset.num_sessions, test_dataset.num_seqs, target_len), number_of_enrollment_sessions, num_verify_sessions=num_verify_sessions, dataset=dataname, results_path=results_path)
        if "pca" in args.metric:
            perplexity = 10
            logger.info(f"tSNE perplexity {perplexity}")
            Metric.save_PCA_curve_fixed_sessions(torch.cat(feature_embeddings, dim=0).view(test_dataset.num_users, test_dataset.num_sessions, test_dataset.num_seqs, target_len), number_of_enrollment_sessions + num_verify_sessions, 10, results_path, perplexity=perplexity)

    elif dataname == 'feta':
        num_users = len(test_dataset.user_list)
        user_session_count = test_dataset.user_session_count
        if "basic" in args.metric:
            # print(torch.cat(feature_embeddings, dim=0).shape)  # (325, 64), 325 = 65 * 5 with 65 = number of subjects, 5 = number of sessions per subjects
            res = get_evaluate_results(torch.cat(feature_embeddings, dim=0), number_of_enrollment_sessions, num_users=num_users, user_session_count=user_session_count, dataset=dataname)
            res_df = pd.DataFrame([list(res)], columns=["eer", "usability", "tcr", "frwi", "fawi"])
            logger.info(f"\nRESULTS\n{res_df}")
            res_df.to_csv(f"{results_path}/basic.csv", index=False)
        if "det" in args.metric:
            Metric.save_DET_curve(torch.cat(feature_embeddings, dim=0), number_of_enrollment_sessions, num_users=num_users, user_session_count=user_session_count, dataset=dataname, results_path=results_path)
        if "pca" in args.metric:
            perplexity = 10
            logger.info(f"tSNE perplexity {perplexity}")
            Metric.save_PCA_curve_vary_sessions(torch.cat(feature_embeddings, dim=0), user_session_count, 10, results_path, perplexity=perplexity)

    else:
        raise NotImplementedError