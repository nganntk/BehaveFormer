"""Preprocessing script for HuMIdb dataset"""

import os
import math
import time
import pickle
import logging
import argparse
import numpy as np
import pandas as pd
import multiprocessing
from datetime import datetime
from sklearn.model_selection import train_test_split

from utils.config import Config


def get_filtered_users(dataset_dir):
    """Get users with 5 sessions.

    Args:
        dataset_dir (str): Path to raw data folder

    Returns:
        list: list of users id to keep
    """

    user_keep = []
    for user in sorted(os.listdir(dataset_dir)):
        count = 0
        for session in [i for i in os.listdir(os.path.join(dataset_dir, user)) if i.startswith('Sesion')]:
            has_scroll = []
            for scroll in scroll_type_list:  # ['SCROLLDOWN', 'SCROLLUP']
                scroll_type = 'd' if scroll == 'SCROLLDOWN' else 'u'
                key = pd.read_csv(os.path.join(dataset_dir, user, session, scroll, f'scroll_{scroll_type}_touch.csv'), sep=' ', names=["event_time", "orient", "x", "y", "pressure", "action"])
                has_scroll.append(key.shape[0] > 0)
            if all(has_scroll):
                count += 1
            else:
                logger.info(f"{user}, {session} has no scroll data")

        if count == 5:
            user_keep.append(user)
    assert len(user_keep) == len(set(user_keep))  # Make sure no duplicates in user list

    return sorted(user_keep)

def imu_feature_extract(imu_type_data):
    """Extracting IMU data for 

    Args:
        imu_type_data (_type_): _description_

    Returns:
        _type_: _description_
    """    
    imu_type_data["event_time"] = imu_type_data["event_time"].astype(int)
    if (imu_type_data.isnull().values.any()):
        logger.info("WARNING: IMU datframe contains NaN")
    
    imu_type_data["x"] = imu_type_data["x"].astype(float)
    imu_type_data["y"] = imu_type_data["y"].astype(float)
    imu_type_data["z"] = imu_type_data["z"].astype(float)

    imu_type_data["fft_x"] = np.abs(np.fft.fft(imu_type_data["x"].values))
    imu_type_data["fft_y"] = np.abs(np.fft.fft(imu_type_data["y"].values))
    imu_type_data["fft_z"] = np.abs(np.fft.fft(imu_type_data["z"].values))

    imu_type_data["fd_x"] = np.gradient(imu_type_data["x"].values, edge_order=2)
    imu_type_data["fd_y"] = np.gradient(imu_type_data["y"].values, edge_order=2)
    imu_type_data["fd_z"] = np.gradient(imu_type_data["z"].values, edge_order=2)

    imu_type_data["sd_x"] = np.gradient(imu_type_data["fd_x"].values, edge_order=2)
    imu_type_data["sd_y"] = np.gradient(imu_type_data["fd_y"].values, edge_order=2)
    imu_type_data["sd_z"] = np.gradient(imu_type_data["fd_z"].values, edge_order=2)

    return imu_type_data

def scroll_feature_extract(scroll_type_data, scroll=None):
    """Extracting scroll/swipe features, applicable for both scroll up and scroll down actions.
    Use same features as IMU. References: ...

    Args:
        scroll_type_data (DataFrame): Input scroll data of 1 session, shape (N, 3)
            N: number of timepoints
            3: event_time, x, y
        scroll (str): Scroll type to add as an additional feature. If None, no differentiate between scroll type
            scroll_type = 1 for SCROLLUP and 0 for SCROLLDOWN
    Returns:
        DataFrame: _description_
    """
    assert scroll in ["SCROLLDOWN", "SCROLLUP", None], f"scroll_type is {scroll}. Only accept SCROLLDOWN or SCROLLUP"

    scroll_type_data["event_time"] = scroll_type_data["event_time"].astype(int)
    if (scroll_type_data.isnull().values.any()):
        logger.info("WARNING: IMU datframe contains NaN")
    
    scroll_type_data["x"] = scroll_type_data["x"].astype(float)
    scroll_type_data["y"] = scroll_type_data["y"].astype(float)

    scroll_type_data["fft_x"] = np.abs(np.fft.fft(scroll_type_data["x"].values))
    scroll_type_data["fft_y"] = np.abs(np.fft.fft(scroll_type_data["y"].values))

    scroll_type_data["fd_x"] = np.gradient(scroll_type_data["x"].values, edge_order=2)
    scroll_type_data["fd_y"] = np.gradient(scroll_type_data["y"].values, edge_order=2)

    scroll_type_data["sd_x"] = np.gradient(scroll_type_data["fd_x"].values, edge_order=2)
    scroll_type_data["sd_y"] = np.gradient(scroll_type_data["fd_y"].values, edge_order=2)
    
    scroll_type_data["type"] = 1 if scroll == "SCROLLUP" else 0

    return scroll_type_data

def embed_zero_padding(sequence, sequence_length):
    sample_count = sequence.shape[0]
    missing_sample_count = sequence_length - sample_count
    new_items_df = pd.DataFrame([[0] * sequence.shape[1] for i in range(missing_sample_count)], columns=list(sequence.columns))
    sequence = pd.concat([sequence,new_items_df],axis=0)
    sequence.reset_index(inplace = True, drop = True)

    return sequence

def sync_imu_data(accelerometer_data, gyroscope_data, magnetometer_data, sync_period, imu_sequence_length):
    if (accelerometer_data.isnull().values.any()):
        logger.info("WARNING: Original accelerometer_data datframe contains NaN")
        accelerometer_data.replace(np.nan, 0, inplace=True)
    if (gyroscope_data.isnull().values.any()):
        logger.info("WARNING: Original gyroscope_data datframe contains NaN")
        gyroscope_data.replace(np.nan, 0, inplace=True)
    if (magnetometer_data.isnull().values.any()):
        logger.info("WARNING: Original magnetometer_data datframe contains NaN")
        magnetometer_data.replace(np.nan, 0, inplace=True)

    imu_prefixes = ["a", "g", "m"]
    column_names = ["x", "y", "z", "fft_x", "fft_y", "fft_z", "fd_x", "fd_y", "fd_z", "sd_x", "sd_y", "sd_z"]
    columns = []
    for prefix in imu_prefixes:
        for name in column_names:
            columns.append(f"{prefix}_{name}")

    imu_sequence = pd.DataFrame(columns=columns)

    accelerometer_min = math.inf
    gyroscope_min = math.inf
    magnetometer_min = math.inf

    if accelerometer_data.shape[0] != 0:
        accelerometer_min = accelerometer_data.iloc[0]['event_time']

    if gyroscope_data.shape[0] != 0:
        gyroscope_min = gyroscope_data.iloc[0]['event_time']

    if magnetometer_data.shape[0] != 0:
        magnetometer_min = magnetometer_data.iloc[0]['event_time']
  
    lowest_time = min(accelerometer_min, gyroscope_min, magnetometer_min)

    accelerometer_max = - math.inf
    gyroscope_max = - math.inf
    magnetometer_max = - math.inf

    if accelerometer_data.shape[0] != 0:
        accelerometer_max = accelerometer_data.iloc[accelerometer_data.shape[0] - 1]['event_time'] 

    if gyroscope_data.shape[0] != 0:  
        gyroscope_max = gyroscope_data.iloc[gyroscope_data.shape[0] - 1]['event_time']

    if magnetometer_data.shape[0] != 0:
        magnetometer_max = magnetometer_data.iloc[magnetometer_data.shape[0] - 1]['event_time']

    highest_time = max(accelerometer_max, gyroscope_max, magnetometer_max)

    start_time = lowest_time
    start_times = []
    while start_time < highest_time:
        start_times.append(start_time)
        start_time = start_time + sync_period
    start_times = np.array(start_times)
    end_times = start_times + sync_period

    imu_sequence = []
    accelerometer_np = accelerometer_data.to_numpy()  # event_time, <column_names>: total 13 columns
    gyroscope_np = gyroscope_data.to_numpy()  # event_time, <column_names>: total 13 columns
    magnetometer_np = magnetometer_data.to_numpy()  # event_time, <column_names>: total 13 columns
    for start_time, end_time in zip(start_times, end_times):
        relevant_accelerometer_np = accelerometer_np[(start_time <= accelerometer_np[:, 0]) & (accelerometer_np[:, 0] <= end_time), 1:]  # skip event_time column
        relevant_gyroscope_np = gyroscope_np[(start_time <= gyroscope_np[:, 0]) & (gyroscope_np[:, 0] <= end_time), 1:]  # skip event_time column
        relevant_magnetometer_np = magnetometer_np[(start_time <= magnetometer_np[:, 0]) & (magnetometer_np[:, 0] <= end_time), 1:]  # skip event_time column
        data = []
        for prefix in imu_prefixes:
            if prefix == 'a':
                values = relevant_accelerometer_np.mean(axis=0) if len(relevant_accelerometer_np) > 0 else np.zeros(len(column_names))
            elif prefix == 'g':
                values = relevant_gyroscope_np.mean(axis=0) if len(relevant_gyroscope_np) > 0 else np.zeros(len(column_names))
            elif prefix == 'm':
                values = relevant_magnetometer_np.mean(axis=0) if len(relevant_magnetometer_np) > 0 else np.zeros(len(column_names))
            data.extend(values.tolist())
        imu_sequence.append(data)
    imu_sequence = pd.DataFrame(imu_sequence, columns=columns, dtype='float64')

    if(imu_sequence.shape[0] > imu_sequence_length):
        imu_sequence = imu_sequence.head(imu_sequence_length)
    elif (imu_sequence.shape[0] < imu_sequence_length):
        imu_sequence = embed_zero_padding(imu_sequence, imu_sequence_length)

    return imu_sequence

def pre_process(event_data, event_sequence_length, imu_sequence_length, offset, accelerometer_data, gyroscope_data, magnetometer_data):
    length = event_data.shape[0]  # Event sequence is stored horizontally
    start = 0
    end = start + event_sequence_length
    event_sequences = []
    # Get overlapped windows of event_sequence_length, stride = offset, 
    while start < length:
        if end >= length:
            event_sequences.append(event_data.loc[start: , :])
            break

        sequence = event_data.loc[start:(end - 1), :]
        sequence.reset_index(inplace = True, drop = True)
        event_sequences.append(sequence) 
        start = start + offset
        end = start + event_sequence_length

    imu_sequences = []

    max_imu_sample_count = -math.inf
  
    for sequence in event_sequences:
        start_time = int(sequence.iloc[0]['event_time'])
        end_time = int(sequence.iloc[-1]['event_time'])
        
        relevant_accelerometer_data = accelerometer_data.loc[(accelerometer_data['event_time'] >= start_time) & (accelerometer_data['event_time'] <= end_time)]
        relevant_gyroscope_data = gyroscope_data.loc[(gyroscope_data['event_time'] >= start_time) & (gyroscope_data['event_time'] <= end_time)]
        relevant_magnetometer_data = magnetometer_data.loc[(magnetometer_data['event_time'] >= start_time) & (magnetometer_data['event_time'] <= end_time)]

        sync_period = (end_time - start_time) / imu_sequence_length

        imu_sequence = sync_imu_data(relevant_accelerometer_data, relevant_gyroscope_data, relevant_magnetometer_data, sync_period, imu_sequence_length)

        imu_sequences.append(imu_sequence)
  
    event_sequences[len(event_sequences) - 1] = embed_zero_padding(event_sequences[len(event_sequences) - 1], event_sequence_length)

    return event_sequences, imu_sequences

def read_imu(csv_file):
    """
    Read IMU data, all follows the same file format
    """
    df = pd.read_csv(csv_file, header=None, sep=' ', names=['event_time', 'orientation', 'x', 'y', 'z'])
    if df.shape[0] == 0:  # No sensor data, add 3 row of 0s. Because np.gradient with edge_order 2 needs input with at least 3 values
        df = pd.DataFrame(np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]), columns=['event_time', 'orientation', 'x', 'y', 'z'])
    elif df.shape[0] == 1:
        df = pd.concat([df, df.tail(1), df.tail(1)], axis=0)
    elif df.shape[0] == 2:
        df = pd.concat([df, df.tail(1)], axis=0)
    df.drop(columns = ['orientation'], inplace=True)

    return df

def parse_info_json(json_file):
    """
    info.json is in non json format, need to parse it here
    """
    with open(json_file, 'r') as f:
        data = f.read()

    data = data[1:-1]  # delete {, }
    data_list = [i.strip() for i in data.split(',')]  
    
    data_dict = {}
    for item in data_list:
        k, v = item.split('=')
        
        data_dict[k] = v.replace("'", "")
        data_dict[k] = int(v) if v.isdigit() else v
    return data_dict

def get_user_data(userid):
    logger.info(f">>> Processing {userid}")

    # Read screen size 
    device_info = parse_info_json(os.path.join(data_dir, userid, 'info.json'))
    screen_width, screen_height = device_info['screenWidth'], device_info['screenHeight']

    # Read each session data
    session_data = []
    for session in sorted([i for i in os.listdir(os.path.join(data_dir, userid)) if i.startswith('Sesion')]):
        for scroll in scroll_type_list:  # ['SCROLLDOWN', 'SCROLLUP']
            scroll_dir = os.path.join(data_dir, userid, session, scroll)
            scroll_type = 'u' if scroll == 'SCROLLUP' else 'd'
            scroll_csv_data = pd.read_csv(os.path.join(scroll_dir, f'scroll_{scroll_type}_touch.csv'), sep=' ',
                                             names=['event_time', 'orientation', 'x', 'y', 'pressure', 'action'])
            assert scroll_csv_data.shape[0] != 0

            # Make sure x is in screen_width and y in screen_height
            # Assign x > screen_width to be screen_width, y > screen_height to be screen_height
            if not ((scroll_csv_data['x'].values <= screen_width).all() and (scroll_csv_data['y'].values <= screen_height).all()):
                logger.info(f"WARNING: User {userid} {session} {scroll} has x/y out of bound. Setting them to screenWidth/screenHeight")
                scroll_csv_data.loc[scroll_csv_data['x'] > screen_width, 'x'] = screen_width
                scroll_csv_data.loc[scroll_csv_data['y'] > screen_height, 'y'] = screen_height
            assert ((scroll_csv_data['x'].values <= screen_width).all() and (scroll_csv_data['y'].values <= screen_height).all()), f"ERROR: Scroll x, y has problem @ {userid} {session} {scroll}."
          
            # Only consider portrait (orientation = 1): width and height same as in info.json
            assert np.unique(scroll_csv_data['orientation']) == 1

            # Normalize x, y in scroll data using screen size
            scroll_csv_data['x'] = scroll_csv_data['x'].div(screen_width)
            scroll_csv_data['y'] = scroll_csv_data['y'].div(screen_height)

            scroll_csv_data.drop(columns=['orientation', 'pressure', 'action'], inplace=True)

            # Read accelerometer, gyroscope, magnetometer
            accelerometer_csv_data = read_imu(os.path.join(scroll_dir, "SENSORS", "sensor_lacc.csv"))
            gyroscope_csv_data = read_imu(os.path.join(scroll_dir, "SENSORS", "sensor_gyro.csv"))
            magnetometer_csv_data = read_imu(os.path.join(scroll_dir, "SENSORS", "sensor_magn.csv"))

            # Extract features
            scroll_csv_data = scroll_feature_extract(scroll_csv_data, scroll)
            accelerometer_csv_data = imu_feature_extract(accelerometer_csv_data)
            gyroscope_csv_data = imu_feature_extract(gyroscope_csv_data)
            magnetometer_csv_data = imu_feature_extract(magnetometer_csv_data)
            
            scroll_sequences, imu_sequences = pre_process(scroll_csv_data, scroll_sequence_len, imu_sequence_len, windowing_offset, accelerometer_csv_data, gyroscope_csv_data, magnetometer_csv_data)
            
            sequence_data = []
            for i in range(len(scroll_sequences)):
                temp_scroll = scroll_sequences[i]
                temp_imu = imu_sequences[i]
                sequence_data.append([temp_scroll.to_numpy(dtype=float_format), temp_imu.to_numpy(dtype=float_format)])
            session_data.append(sequence_data)

    logger.info(f"INFO: User {userid} completed")
    
    return session_data

def read_scroll(users_list, ncpus=None):
    if ncpus:
        with multiprocessing.Pool(ncpus) as p:
            scroll_data = p.map(get_user_data, users_list)
        scroll_data = [x for x in scroll_data]
    else:
        scroll_data = []
        for userid in users_list:
            userid_sess = get_user_data(userid)
            scroll_data.append(userid_sess)

    return scroll_data

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help="Path to config file (.yaml)")
    parser.add_argument('--ncpus', help="Number of CPUs to use", default=1, type=int)
    return parser.parse_args()

def main(args):
    # Use global variables
    global scroll_sequence_len, imu_sequence_len, windowing_offset, scroll_type_list, float_format
    global data_dir, working_dir, logger

    NCPUS = args.ncpus

    config = Config(args.config).get_config_dict()
    print(config)

    # Read hyperparameters
    dataname = config['dataname']
    scroll_sequence_len = config['preprocess']['dataset_config']['scroll_sequence_len']
    imu_sequence_len = config['preprocess']['dataset_config']['imu_sequence_len']
    windowing_offset = config['preprocess']['dataset_config']['windowing_offset']
    scroll_type_list = config['preprocess']['dataset_config']['scroll_type_list']
    float_format = config['preprocess']['float_format']
    num_training = int(config['data_split']['training'])
    num_validation = int(config['data_split']['validation'])
    num_testing = int(config['data_split']['testing'])
    
    # Input, output
    data_dir = config['preprocess']['dataset_root']
    output_dir = config['preprocess']['output_dir']
    config_name = os.path.basename(args.config).split('.')[0].split(f'{dataname}_')[-1]
    working_dir = os.path.join(output_dir, config_name)
    os.makedirs(working_dir)
    
    # Set logging
    logger = logging.getLogger()
    logging.getLogger('matplotlib.font_manager').disabled = True
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(os.path.join(working_dir, 'preproc.log'))
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    logger.info(f"START TIME: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Running with {NCPUS} CPUs".upper())
    logger.info(f"CONFIG: scroll_type {scroll_type_list}, scroll_sequence_len {scroll_sequence_len}, imu_sequence_len {imu_sequence_len}, windowing_offset {windowing_offset}")

    # Get user list without missing data
    user_list = get_filtered_users(data_dir)
    logger.info(f">>> Total number of users: {len(user_list)}".upper())
    assert len(user_list) == num_training + num_validation + num_testing, \
        f"Number of users changed. There are {len(user_list)} users, which is not equal to {num_training} + {num_validation} + {num_testing}"
    start_time = time.time()

    # Main work: data split, then preprocess
    if os.path.exists(os.path.join(working_dir, 'splits.pickle')):
        logger.info("WARNING: Reading pre-splited splits")
        with open(os.path.join(working_dir, 'splits.pickle'), 'rb') as f:
            user_lists = pickle.load(f)
    else:
        logger.info("INFO: Start splitting to train, validation, and test")

        # Follow HuMINet: 65 for validation, 65 for testing, the rest for training (=428-65-65=298)
        training_user_list, val_test_user_list = train_test_split(user_list, test_size=(num_validation + num_testing), train_size=num_training, shuffle=True, random_state=1234)
        validation_user_list, testing_user_list = train_test_split(val_test_user_list, test_size=num_testing, train_size=num_validation, shuffle=True, random_state=1234)
        user_lists = {'training': training_user_list, 
                      'validation': validation_user_list,
                      'testing': testing_user_list}
        with open(os.path.join(working_dir, 'splits.pickle'), 'wb') as f:
            pickle.dump(user_lists, f)

    # Set output files
    output_files = {k: f"{working_dir}/{k}_scroll_imu_data_all.pickle" for k in ['training', 'validation', 'testing']}
    for k, v in output_files.items():
        if os.path.exists(v):
            logger.info(f"{k}'s output is already existed at {v}. Skipping")
            continue

        # Read scroll data for each split and save them to the corresponding file
        logger.info(f">>> \t{k.capitalize()}: {len(user_lists[k])} users.")
        logger.info(f">>> {k.capitalize()} is at {v}")
        scroll_imu_data = read_scroll(user_lists[k], NCPUS)
        with open(output_files[k], 'wb') as f:
            pickle.dump(scroll_imu_data, f)

    end_time = time.time()
    logger.info(f">>> Elapsed time {((end_time - start_time)/60):.4f} minutes".upper())
    logger.info(f"END TIME: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Remove the current logger handle
    while logger.hasHandlers():
        logger.removeHandler(logger.handlers[0])


if __name__ == "__main__":
    # Global variable
    logger = logging.getLogger()

    # These will be set in config
    scroll_sequence_len = -1
    imu_sequence_len = -1
    windowing_offset = -1
    scroll_type_list = None
    float_format = ''

    # Folders
    data_dir = None
    working_dir = ''

    main(get_args())