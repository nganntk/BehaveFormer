"""Main training code"""
import os
import json
import math
import time
import random
import shutil
import logging
import argparse
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader

from model.dataset import HUMITrainDataset, HUMITestDataset
from model.dataset import FETATrainDataset, FETATestDataset
from model.loss import TripletLoss
from model.behaveformer import BehaveFormer
from evaluation.metrics import Metric
from utils.config import Config
from utils.utils import read_pickle, list2txt


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
    fh = logging.FileHandler(os.path.join(log_dir, 'training.log'))
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    return logger    

def evaluate(model, test_dataset, test_dataloader, target_len, number_of_enrollment_sessions, number_of_verify_sessions, imu_type, device, dataname):
    """Evaluation method at the end of each training epoch

    Args:
        model (BehaveFormer): Current model
        test_dataset (torch.utils.data.Dataset): Test dataset
        test_dataloader (torch.utils.data.DataLoader): Test dataloader
        number_of_enrollment_sessions (int): Number of enrollment sessions
        number_of_verify_sessions (int): For HuMidb with fixed number of verify sessions
        imu_type (str): 'all' or 'none', used for getting feature_embeddings 
        device (torch.device): cuda or cpu
        dataname (str): 'humi' or 'feta', used to determine how to calculate EER

    Returns:
        float: EER value
    """    
    model.train(False)

    with torch.no_grad():
        feature_embeddings = []
        for batch_idx, item in enumerate(test_dataloader):
            if imu_type != 'none':
                feature_embeddings.append(model([item[0].to(device).float(), item[1].to(device).float()]))
            else:
                feature_embeddings.append(model(item[0].to(device).float()))
    
    if dataname == 'humi':
        eer = Metric.cal_user_eer_fixed_sessions(torch.cat(feature_embeddings, dim=0).view(test_dataset.num_users, test_dataset.num_sessions, test_dataset.num_seqs, target_len), number_of_enrollment_sessions, number_of_verify_sessions)[0]
       
    elif dataname == 'feta':
        num_users = len(test_dataset.user_list)
        user_session_count = test_dataset.user_session_count
        eer = Metric.cal_user_eer_vary_sessions(torch.cat(feature_embeddings, dim=0), num_users, user_session_count, number_of_enrollment_sessions)[0]

    return eer

def get_args():
    """Argument parser for train code"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataname', help='Dataset name', choices=['humi', 'feta'], required=True)
    parser.add_argument('-c', '--config', help="Config file", required=True)
    parser.add_argument('--mode', choices=['from_scratch', 'resume', 'transfer_learning'], help="Mode", default='from_scratch')
    parser.add_argument('--debug', action='store_true', help="Debug mode, only train for 3 epochs")
    parser.add_argument('--work_dirs_subfolder', help="Subfolder in work_dirs, used for ablation study to not clutter main results folder", default='')
    parser.add_argument('--checkpoint_interval', help="Interval to save checkpoint, if >0, checkpoint will be save for based on the interval. Pass -1 to skip", default=100, type=int)
    return parser.parse_args()

def main(args):
    # Setup: timestamp, seed, config file
    start_time = time.time()
    set_random_seeds(1234)
    print(f"INFO: Input argument: {str(args)}")
    dataname = args.dataname

    # Create output folder and copy config file over
    start_time_fmt = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_config = os.path.basename(args.config).split('.')[0]
    work_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'work_dirs', args.work_dirs_subfolder, model_config, start_time_fmt)
    if args.debug:
        work_dir = os.path.join(os.path.dirname(os.path.dirname(work_dir)), 'debug', start_time_fmt)
    print('work_dir', work_dir)
    os.makedirs(work_dir)
    args.config = shutil.copy2(args.config, os.path.join(work_dir))

    best_model_save_path = os.path.join(work_dir, 'best_models')
    checkpoint_save_path = os.path.join(work_dir, 'checkpoints')
    os.makedirs(best_model_save_path)
    os.makedirs(checkpoint_save_path)

    # Read config, set output folder to config
    config_data = Config(args.config).get_config_dict()
    assert dataname == config_data['dataname'], f"dataname is {dataname} while config's data name is {config_data['dataname']}"
    if(config_data["GPU"] == "True"):
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    # Read hyperparamters
    scroll_sequence_len = config_data['data']['scroll_sequence_len']
    imu_sequence_len = config_data['data']['imu_sequence_len']
    action_type = config_data['data']['action_type']
    
    hyperparams = config_data['hyperparams']
    init_epoch = hyperparams['init_epoch']
    epochs = hyperparams['epochs']
    if args.debug:
        init_epoch = 0
        epochs = 3
    batch_size = hyperparams['batch_size']
    epoch_batch_count = hyperparams['epoch_batch_count']
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
    num_verify_sessions = hyperparams['number_of_verify_sessions'] if dataname == 'humi' else None  # If None, determined based on number of sessions for each user

    learning_rate = hyperparams['learning_rate']
    logger = create_logger(work_dir)
    logger.info(f"Input argument: {str(args)}\n")
    
    # NOTE: Dataset assume column order is acc, gyr, mag by default
    if dataname == 'humi':
        splits = read_pickle(os.path.join(config_data['folders']['data_dir'], 'splits.pickle'))

        train_dataset = HUMITrainDataset(batch_size=batch_size,
                                         epoch_batch_count=epoch_batch_count,
                                         action=action_type,
                                         training_file=os.path.join(config_data['folders']['root_dir'], config_data['folders']['data_dir'], 'training_scroll_imu_data_all.pickle'),
                                         user_list=splits['training'],
                                         imu_type=imu_type)
        logger.info(f"INFO: Training on {train_dataset.dataset_name}")

        val_dataset = HUMITestDataset(action=action_type, 
                                      validation_file=os.path.join(config_data['folders']['root_dir'], config_data['folders']['data_dir'], 'validation_scroll_imu_data_all.pickle'),
                                      imu_type=imu_type)
        # Create dataloaders
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

    elif dataname == 'feta':
        splits = read_pickle(os.path.join(config_data['folders']['data_dir'], 'splits.pickle'))
        logger.info(f"#training {len(splits['training'])}, #validation {len(splits['validation'])}")
        
        train_dataset = FETATrainDataset(data_root=os.path.join(config_data['folders']['data_dir'], 'processed'),
                                         batch_size=batch_size,
                                         epoch_batch_count=epoch_batch_count,
                                         action=action_type,
                                         user_list=splits['training'],
                                         imu_type=imu_type,
                                         data_type='float64')
        logger.info(f"INFO: Training on {train_dataset.dataset_name}")

        val_dataset = FETATestDataset(data_root=os.path.join(config_data['folders']['data_dir'], 'processed'),
                                      action=action_type, 
                                      user_list=splits['validation'],
                                      imu_type=imu_type,
                                      data_type='float64')    
        # Create dataloaders
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4)
   
    logger.info("INFO: Using BehaveFormer")
    print('imu_type', imu_type)
    model = BehaveFormer(scroll_feature_dim, imu_feature_dim, scroll_sequence_len, imu_sequence_len, target_len, gre_k, behave_temporal_heads, behave_channel_heads, imu_temporal_heads, imu_channel_heads, imu_type=imu_type)
    loss_fn = TripletLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.05)

    # Warmup learning rate from base_lr to target_lr (learning_rate) over warmup_epochs            
    base_lr = hyperparams['warmup_baselr']
    warmup_epochs = hyperparams['warmup_epochs']
    warmup_steps = len(train_dataloader) * warmup_epochs  # warmup is trigger after every batch so warmup_steps = #batch * #warmup_epochs
    training_steps = len(train_dataloader) * (epochs - warmup_epochs)
    lr_warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=float(base_lr / learning_rate), total_iters=warmup_steps)
    lr_constant = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1, total_iters=training_steps)
    lr_schedule_list = [lr_warmup, lr_constant]
    lr_scheduler = torch.optim.lr_scheduler.ChainedScheduler(lr_schedule_list)
    
    logger.info(f"Number of epochs {epochs}")
    g_eer = math.inf
    # Either transfer learning (from a different dataset) or resume or train from scratch
    if args.mode == 'transfer_learning':    # Load pretrain weights
        checkpoint_pt = config_data['folders']['transfer_learning_weights']
        logger.info(f"Transfer learning, use pretrained weights from {checkpoint_pt}")
        if (config_data["GPU"] == "True"):
            checkpoint = torch.load(checkpoint_pt)
        else:
            checkpoint = torch.load(checkpoint_pt, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint)
    
    elif args.mode == 'resume':    # Continue training
        assert (init_epoch % 100 == 0), f"Only saved weights every 100 epochs, cannot start from {init_epoch}."
        latest_timestamp = sorted(os.listdir(os.path.dirname(work_dir)))[-2]  # the one at -1 is the current output folder
        checkpoint_pt = os.path.join(os.path.dirname(work_dir), latest_timestamp, 'checkpoints', f'training_{init_epoch}.tar')
        logger.info(f"Resume from {init_epoch} at {latest_timestamp}")

        if (config_data["GPU"] == "True"):
            checkpoint = torch.load(checkpoint_pt)
        else:
            checkpoint = torch.load(checkpoint_pt, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        init_epoch = int(checkpoint['epoch'])
        init_eer = checkpoint['eer']

        epochs = init_epoch + epochs
        g_eer = init_eer

    # MAIN WORK
    history = {'train': {'loss': []}, 
               'val': {'eer': []}, 
               'best_val_eer': None, 
               'best_epoch': None,
               'total_epoch': epochs - init_epoch}
    best_epoch = -1
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for i in range(init_epoch, epochs):
        t_loss = 0.0
        start = time.time()
        model.train(True)
        for batch_idx, item in enumerate(train_dataloader):
            anchor, positive, negative = item
            optimizer.zero_grad()

            if imu_type != 'none':
                anchor_out = model([anchor[0].to(device).float(), anchor[1].to(device).float()])
                positive_out = model([positive[0].to(device).float(), positive[1].to(device).float()])
                negative_out = model([negative[0].to(device).float(), negative[1].to(device).float()])
            else:
                anchor_out = model(anchor[0].to(device).float())
                positive_out = model(positive[0].to(device).float())
                negative_out = model(negative[0].to(device).float())
            loss = loss_fn(anchor_out, positive_out, negative_out)
            loss.backward()
            optimizer.step()
            if lr_scheduler is not None:
                lr_scheduler.step()
            
            t_loss = t_loss + loss.item()
            if batch_idx == len(train_dataloader)-1:
                t_loss = t_loss / len(train_dataloader)
  
        eer = evaluate(model, val_dataset, val_dataloader, target_len, number_of_enrollment_sessions, num_verify_sessions, imu_type, device, dataname)
        end = time.time()
        history['train']['loss'].append(t_loss)
        history['val']['eer'].append(eer)
        # Note: lr here is after 1 epoch (multiple batches) but warmup lr is done after each batch. However, we should still see an increase in lr
        # For warmup case: the lr here is after lr_scheduler.step
        if lr_scheduler is not None:
            logger.info(f"------> Epoch No: {i+1} - LR: {lr_scheduler.get_last_lr()[0]:>7f} - Loss: {t_loss:>7f} - EER: {eer:>7f} - Time: {end-start:>2f}")
        else:
            logger.info(f"------> Epoch No: {i+1} - Loss: {t_loss:>7f} - EER: {eer:>7f} - Time: {end-start:>2f}")
        if (eer < g_eer):
            logger.info(f"EER improved from {g_eer} to {eer}")
            g_eer = eer
            best_epoch = i+1
            torch.save(model.state_dict(), best_model_save_path + f"/epoch_{i+1}_eer_{eer}.pt")
        
        if args.checkpoint_interval > 0 and ((i+1) % args.checkpoint_interval == 0):
            torch.save({
                'epoch': i+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'eer': g_eer
            }, f"{checkpoint_save_path}/training_{i+1}.tar")

    # Finishing: save model, print elapsed time
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'eer': g_eer
    }, f"{checkpoint_save_path}/training_{epochs}.tar")
    end_time = time.time()
    print(f"Total time: {((end_time - start_time)/60):.4f} minutes.")
    logger.info(f"Total time: {((end_time - start_time)/60):.4f} minutes.")
    logger.info('Best validation eer: {:.4f} at epoch {}'.format(g_eer, best_epoch))
    
    # Set best params
    history['best_val_eer'] = g_eer
    history['best_epoch'] = best_epoch

    # Save loss, eer history to files
    train_history_txt = os.path.join(work_dir, 'train_loss.txt')
    logger.info(f"Saved train loss history to {train_history_txt}")
    list2txt(history['train']['loss'], train_history_txt)
    
    val_history_txt = os.path.join(work_dir, 'val_eer.txt')
    logger.info(f"Saved validation eer history to {val_history_txt}")
    list2txt(history['val']['eer'], val_history_txt)

    # Save best params
    with open(os.path.join(work_dir, 'best_params.json'), 'w') as fp:
        json.dump(history, fp, indent=4)
    
    # Remove the current logger handle
        logger.info(f"Total time: {((end_time - start_time)/60):.4f} minutes.")

    # Remove the current logger handle
    while logger.hasHandlers():
        logger.removeHandler(logger.handlers[0])


if __name__ == "__main__":
    main(get_args())