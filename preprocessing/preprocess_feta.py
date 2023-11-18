"""Preprocessing script for FETA dataset"""

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


def get_filtered_users(data_dir, user_csv):
    """Get user list with more than 5 scroll sessions

    Args:
        data_dir (_type_): _description_

    Returns:
        user_list: _description_
        user_map: df containing user id, session id, etc.
    """
    # Iterate through multiple nested folders
    user_data = []
    user_data_count = {}
    for subdir, _, files in os.walk(data_dir):
        for file in files:
            path = os.path.join(subdir, file)
            rel_path = os.path.relpath(path, data_dir)

            try:
                user_id, sess_id, action, repetition_id, data_type, data_file = rel_path.split('/')

                if (data_file == ".DS_Store") or (data_type not in ["touch_data", "gyroscope_data", "accelerometer_data"]) or (action != "scroll"):
                    continue

                user_data.append({
                    'user_id': user_id,
                    'sess_id': sess_id,
                    'action': action,
                    'repetition_id': repetition_id,
                    'data_type': data_type,
                    'data_file': data_file
                })
                if user_id not in user_data_count:
                    user_data_count[user_id] = 1
                else:
                    user_data_count[user_id] += 1

            except ValueError:
                continue
    user_data = pd.DataFrame(user_data)

    # Drop user with less than 5 sessions
    user_drop = set([k for k, v in user_data_count.items() if v < 5])

    if len(user_drop) > 0:
        print(f"Dropping {len(user_drop)} users.")
        user_data = user_data.loc[~(user_data['user_id'].isin(user_drop)), :].copy()

    user_data.to_csv(user_csv, index=False)

    # Get user list
    user_list = user_data['user_id'].unique().tolist()

    return sorted(user_list)

def read_imu(csv_file):
    """
    Read IMU data, all follows the same file format
    """
    df_cols = ['timestamp', 'x', 'y', 'z']
    df = pd.read_csv(csv_file, usecols=df_cols)[df_cols]
    df.rename(columns={'timestamp': 'event_time'}, inplace=True)  # uniform with HuMIdb

    if df.shape[0] == 0:  # No sensor data, add 3 row of 0s. Because np.gradient with edge_order 2 needs input with at least 3 values
        df = pd.DataFrame(np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]), columns=['event_time', 'x', 'y', 'z'])
    elif df.shape[0] == 1:
        df = pd.concat([df, df.tail(1), df.tail(1)], axis=0)
    elif df.shape[0] == 2:
        df = pd.concat([df, df.tail(1)], axis=0)

    return df

def imu_feature_extract(imu_type_data):
    imu_type_data["event_time"] = imu_type_data["event_time"].astype(int)
    if (imu_type_data.isnull().values.any()):
        logger.info("WARNING: IMU dataframe contains NaN")
    
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

def scroll_feature_extract(scroll_type_data):
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
    
    scroll_type_data.insert(len(scroll_type_data.columns)-1, 'scroll_type', scroll_type_data.pop('scroll_type'))  # Move this to the end

    return scroll_type_data

def embed_zero_padding(sequence, sequence_length):
    sample_count = sequence.shape[0]
    missing_sample_count = sequence_length - sample_count
    new_items_df = pd.DataFrame([[0] * sequence.shape[1] for i in range(missing_sample_count)], columns=list(sequence.columns))
    sequence = pd.concat([sequence,new_items_df],axis=0)
    sequence.reset_index(inplace = True, drop = True)

    return sequence

def sync_imu_data(accelerometer_data, gyroscope_data, sync_period, imu_sequence_length):
    """NOTE: FETA only have accelerometer and gyroscope data

    Args:
        accelerometer_data (_type_): _description_
        gyroscope_data (_type_): _description_
        sync_period (_type_): _description_
        imu_sequence_length (_type_): _description_

    Returns:
        _type_: _description_
    """    
    if (accelerometer_data.isnull().values.any()):
        logger.info("WARNING: Original accelerometer_data datframe contains NaN")
        accelerometer_data.replace(np.nan, 0, inplace=True)
    if (gyroscope_data.isnull().values.any()):
        logger.info("WARNING: Original gyroscope_data datframe contains NaN")
        gyroscope_data.replace(np.nan, 0, inplace=True)

    imu_prefixes = ["a", "g"]
    column_names = ["x", "y", "z", "fft_x", "fft_y", "fft_z", "fd_x", "fd_y", "fd_z", "sd_x", "sd_y", "sd_z"]
    columns = []
    for prefix in imu_prefixes:
        for name in column_names:
            columns.append(f"{prefix}_{name}")

    accelerometer_min = math.inf
    gyroscope_min = math.inf
    magnetometer_min = math.inf

    if accelerometer_data.shape[0] != 0:
        accelerometer_min = accelerometer_data.iloc[0]['event_time']

    if gyroscope_data.shape[0] != 0:
        gyroscope_min = gyroscope_data.iloc[0]['event_time']

    lowest_time = min(accelerometer_min, gyroscope_min, magnetometer_min)

    accelerometer_max = - math.inf
    gyroscope_max = - math.inf
    magnetometer_max = - math.inf

    if accelerometer_data.shape[0] != 0:
        accelerometer_max = accelerometer_data.iloc[accelerometer_data.shape[0] - 1]['event_time'] 

    if gyroscope_data.shape[0] != 0:  
        gyroscope_max = gyroscope_data.iloc[gyroscope_data.shape[0] - 1]['event_time']

    highest_time = max(accelerometer_max, gyroscope_max, magnetometer_max)

    # Generate start time, don't use np.arange
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
    for start_time, end_time in zip(start_times, end_times):
        relevant_accelerometer_np = accelerometer_np[(start_time <= accelerometer_np[:, 0]) & (accelerometer_np[:, 0] <= end_time), 1:]  # skip event_time column
        relevant_gyroscope_np = gyroscope_np[(start_time <= gyroscope_np[:, 0]) & (gyroscope_np[:, 0] <= end_time), 1:]  # skip event_time column
        data = []
        for prefix in imu_prefixes:
            if prefix == 'a':
                values = relevant_accelerometer_np.mean(axis=0) if len(relevant_accelerometer_np) > 0 else np.zeros(len(column_names))
            elif prefix == 'g':
                values = relevant_gyroscope_np.mean(axis=0) if len(relevant_gyroscope_np) > 0 else np.zeros(len(column_names))
            data.extend(values.tolist())
        imu_sequence.append(data)
    imu_sequence = pd.DataFrame(imu_sequence, columns=columns, dtype='float64')

    if(imu_sequence.shape[0] > imu_sequence_length):
        imu_sequence = imu_sequence.head(imu_sequence_length)
    elif (imu_sequence.shape[0] < imu_sequence_length):
        imu_sequence = embed_zero_padding(imu_sequence, imu_sequence_length)

    return imu_sequence

def pre_process(event_data, event_sequence_length, imu_sequence_length, offset, accelerometer_data, gyroscope_data):
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

        sync_period = (end_time - start_time) / imu_sequence_length

        imu_sequence = sync_imu_data(relevant_accelerometer_data, relevant_gyroscope_data, sync_period, imu_sequence_length)

        imu_sequences.append(imu_sequence)
  
    event_sequences[len(event_sequences) - 1] = embed_zero_padding(event_sequences[len(event_sequences) - 1], event_sequence_length)

    return event_sequences, imu_sequences

def split_by_direction(scroll_data):
    """Find direction for scrolls: down or up.
    TODO: More efficent way

    Args:
        scroll_data (_type_): _description_
    """
    startY, endY = -1, -1
    index_list = []
    index_list_down, index_list_up = [], []

    # If first 2 rows is 0 => cannot start a scroll down or scroll up -> drop first row
    if scroll_data['type'].values[0] == 0  and scroll_data['type'].values[1] == 0:
        scroll_data.drop(scroll_data.head(1).index, inplace=True)

    # If last row is 0 => cannot start a scroll down or scroll up -> drop last row
    if scroll_data['type'].values[-1] == 0:
        scroll_data.drop(scroll_data.tail(1).index, inplace=True)

    # type need to have pattern 0 1..1 2. If 2 is missing, assign the last 1 as 2
    ## Create a column of next row value in case of missing 2
    scroll_data['type_next'] = scroll_data['type'].shift(-1)
    for index, row in scroll_data.iterrows():
        if int(row['type']) == 0:
            startY = row['y']
            index_list = [index]  # restart
        
        elif int(row['type']) == 1:
            if int(row['type_next']) == 0:  # this row should be type = 2 but was wrong, do the find type logic here
                index_list.append(index)
                endY = row['y']
                if startY - endY < 0:  # SCROLLDOWN
                    index_list_down.extend(index_list)
                else:  # SCROLLUP
                    index_list_up.extend(index_list)

                # clear
                startY, endY = -1, -1
                index_list = []
            
            else:
                index_list.append(index)
        
        elif int(row['type']) == 2:
            endY = row['y']
            index_list.append(index)

            if startY - endY < 0:
                index_list_down.extend(index_list)
            else:
                index_list_up.extend(index_list)

            # clear
            startY, endY = -1, -1
            index_list = []  

        else:
            raise ValueError
    # Assign type to 0 for SCROLLDOWN or 1 for SCROLLUP
    scroll_data.drop(columns=['type', 'type_next'], inplace=True)
    scroll_data.loc[index_list_down, 'scroll_type'] = 0
    scroll_data.loc[index_list_up, 'scroll_type'] = 1

    scroll_down = scroll_data.loc[index_list_down, :].copy().reset_index(drop=True)
    scroll_up = scroll_data.loc[index_list_up, :].copy().reset_index(drop=True)


    if len(scroll_data) > 0:
        assert all(x in [0, 1] for x in [int(i) for i in scroll_data['scroll_type'].unique()]), f"{sorted(scroll_data['scroll_type'].unique())}"
    if len(scroll_down) > 0:
        assert [int(i) for i in scroll_down['scroll_type'].unique()] == [0], f"{scroll_down['scroll_type'].unique()}"
    if len(scroll_up) > 0:
        assert [int(i) for i in scroll_up['scroll_type'].unique()] == [1], f"{scroll_up['scroll_type'].unique()}"

    return scroll_down, scroll_up, scroll_data

def get_screen_size(user_id: str):
    """Get screen size for the current user id based on phone model

    Args:
        user_id (_type_): _description_

    Returns:
        _type_: _description_
    """    
    model = data_metadata.loc[data_metadata["uuid"] == user_id]["phone_model"].values[0]

    return device_width_height[model]

def get_user_data(userid: str):
    logger.info(f">>> Processing {userid}")
    user_dir = os.path.join(working_dir, userid)  # output
    if len(sorted(os.listdir(os.path.join(data_dir, userid)))) < 5:
        logger.info(f"Skipping {userid}, less than 5 sessions")
        return

    # Read screen size 
    screen_width, screen_height = get_screen_size(userid)

    # Read each session data
    session_data_count = 0
    session_limit = False
    session_list = sorted([i for i in os.listdir(os.path.join(data_dir, userid)) if i != '.DS_Store'])
    for session in session_list:
        repetition_scrolldown, repetition_imudown = [], []
        repetition_scrollup, repetition_imuup = [], []

        repetition_list = sorted([i for i in os.listdir(os.path.join(data_dir, userid, session, 'scroll')) if i != '.DS_Store'])
        for repetition in repetition_list:
            # 1. Read touch data
            scroll_dir = os.path.join(data_dir, userid, session, 'scroll', repetition, 'touch_data')
            scroll_csv = [i for i in os.listdir(scroll_dir) if i.endswith('.csv')]
            assert len(scroll_csv) == 1, "In touch_csv. Something wrong, each repetition should only have 1 .csv file"
            scroll_cols = ['timestamp', 'x', 'y', 'type']  # ['timestamp', 'x', 'y', 'pressure', 'area', 'type']
            scroll = pd.read_csv(os.path.join(scroll_dir, scroll_csv[0]), usecols=scroll_cols)[scroll_cols]
            assert len(scroll) > 0
            scroll.rename(columns={'timestamp': 'event_time'}, inplace=True)  # uniform with HuMIdb

            # Assign x > screen_width to be screen_width, y > screen_height to be screen_height
            if not ((scroll['x'].values <= screen_width).all() and (scroll['y'].values <= screen_height).all()):
                logger.info(f"WARNING: User {userid} {session} {repetition} has x/y out of bound. Setting them to screenWidth/screenHeight")
                scroll.loc[scroll['x'] > screen_width, 'x'] = screen_width
                scroll.loc[scroll['y'] > screen_height, 'y'] = screen_height
            assert ((scroll['x'].values <= screen_width).all() and (scroll['y'].values <= screen_height).all()), f"ERROR: Scroll x, y has problem @ {userid} {session} {repetition}."

            # Normalize x, y in scroll data using screen size
            scroll['x'] = scroll['x'].div(screen_width)
            scroll['y'] = scroll['y'].div(screen_height)

            scroll_down, scroll_up, scroll_downup = split_by_direction(scroll)  # add scroll_type column: down (0) or up (1)

            # 2. Read accelerometer data
            accel_dir = os.path.join(data_dir, userid, session, 'scroll', repetition, 'accelerometer_data')
            accel_csv = [i for i in os.listdir(accel_dir) if i.endswith('.csv')]
            assert len(accel_csv) == 1, "In accel_csv. Something wrong, each repetition should only have 1 .csv file"
            accelerometer = read_imu(os.path.join(accel_dir, accel_csv[0]))
            accelerometer = imu_feature_extract(accelerometer)

            # 3. Read gyroscope data
            gyro_dir = os.path.join(data_dir, userid, session, 'scroll', repetition, 'gyroscope_data')
            gyro_csv = [i for i in os.listdir(gyro_dir) if i.endswith('.csv')]
            assert len(gyro_csv) == 1, "In gyro_csv. Something wrong, each repetition should only have 1 .csv file"
            gyroscope = read_imu(os.path.join(gyro_dir, gyro_csv[0]))
            gyroscope = imu_feature_extract(gyroscope)

            if len(scroll_down) >= 3 and len(scroll_up) >= 3 and len(scroll_downup) >= 3:
                scroll_down = scroll_feature_extract(scroll_down)
                scroll_down_sequences, imu_down_sequences = pre_process(scroll_down, scroll_sequence_len, imu_sequence_len, windowing_offset, accelerometer, gyroscope)
                repetition_scrolldown.extend(scroll_down_sequences)
                repetition_imudown.extend(imu_down_sequences)

                scroll_up = scroll_feature_extract(scroll_up)
                scroll_up_sequences, imu_up_sequences = pre_process(scroll_up, scroll_sequence_len, imu_sequence_len, windowing_offset, accelerometer, gyroscope)
                repetition_scrollup.extend(scroll_up_sequences)
                repetition_imuup.extend(imu_up_sequences)

                logger.info(f"INFO: User {userid} session {session} found a repetition")
                break  # just need to find 1 repetition
        
        # Save data within 1 session
        if len(repetition_scrolldown) > 0 and len(repetition_scrollup) > 0 and len(repetition_scrolldown) > 0:
            for i_seq, (temp_scroll, temp_imu) in enumerate(zip(repetition_scrolldown, repetition_imudown)):
                curr_outdir = os.path.join(user_dir, 'down', str(session_data_count), str(i_seq))
                os.makedirs(curr_outdir)
                with open(os.path.join(curr_outdir, f'scroll_imu.pickle'), 'wb') as f:
                    pickle.dump([temp_scroll.to_numpy(dtype=float_format), temp_imu.to_numpy(dtype=float_format)], f)

            for i_seq, (temp_scroll, temp_imu) in enumerate(zip(repetition_scrollup, repetition_imuup)):
                curr_outdir = os.path.join(user_dir, 'up', str(session_data_count), str(i_seq))
                os.makedirs(curr_outdir)
                with open(os.path.join(curr_outdir, f'scroll_imu.pickle'), 'wb') as f:
                    pickle.dump([temp_scroll.to_numpy(dtype=float_format), temp_imu.to_numpy(dtype=float_format)], f)

            session_data_count += 1
            session_limit = (session_data_count >= 10)

        if session_limit:
            break
    logger.info(f"INFO: User {userid} completed")

def read_scroll(users_list, ncpus=None):
    if ncpus > 1:
        with multiprocessing.Pool(processes=ncpus, maxtasksperchild=3) as p:  # Restart often to avoid increasing memory
            p.map(get_user_data, users_list, chunksize=3)
    else:
        for userid in users_list:
            get_user_data(userid)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help="Path to config file (.yaml)")
    parser.add_argument('--mode', help="Mode: preprocess, split", choices=['preprocess', 'split'])
    parser.add_argument('--ncpus', help="Number of CPUs to use", default=1, type=int)
    return parser.parse_args()

def main(args):
    # Use global variables
    global scroll_sequence_len, imu_sequence_len, windowing_offset, data_metadata, float_format
    global data_dir, working_dir, logger

    NCPUS = args.ncpus
    
    config = Config(args.config).get_config_dict()
    print(config)

    dataname = config['dataname']
    scroll_sequence_len = config['preprocess']['dataset_config']['scroll_sequence_len']
    imu_sequence_len = config['preprocess']['dataset_config']['imu_sequence_len']
    windowing_offset = config['preprocess']['dataset_config']['windowing_offset']
    float_format = config['preprocess']['float_format']

    # Input, output
    data_root_dir = config['preprocess']['dataset_root']
    data_dir = os.path.join(data_root_dir, 'data_files')
    output_dir = config['preprocess']['output_dir']
    config_name = os.path.basename(args.config).split('.')[0].split(f'{dataname}_')[-1]

    if args.mode == 'preprocess':
        print(f"Processing {dataname.upper()} using {NCPUS} CPUs")

        # Input, output
        working_dir = os.path.join(output_dir, config_name, 'processed')
        os.makedirs(working_dir)
        
        # Set logging
        logging.getLogger('matplotlib.font_manager').disabled = True
        logger.setLevel(logging.DEBUG)
        fh = logging.FileHandler(os.path.join(os.path.dirname(working_dir), 'preproc.log'))
        fh.setLevel(logging.DEBUG)
        logger.addHandler(fh)
        logger.info(f"START TIME: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Running with {NCPUS} CPUs".upper())
        logger.info(f"CONFIG: scroll_type downup, scroll_sequence_len {scroll_sequence_len}, imu_sequence_len {imu_sequence_len}, windowing_offset {windowing_offset}")

        # Get device information
        data_metadata = pd.read_csv(os.path.join(data_root_dir, 'tables', 'userdata.csv'))
        
        # Get user list, only keep user with more than 5 sessions
        user_list = get_filtered_users(data_dir, user_csv=os.path.join(os.path.dirname(working_dir), 'users_all.csv'))
        logger.info(f">>> Total number of users: {len(user_list)}".upper())
        start_time = time.time()

        # Process all users
        read_scroll(user_list, NCPUS)

        end_time = time.time()
        logger.info(f">>> Elapsed time {((end_time - start_time)/60):.4f} minutes".upper())
        logger.info(f"END TIME: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Remove the current logger handle
        while logger.hasHandlers():
            logger.removeHandler(logger.handlers[0])

    elif args.mode == 'split':
        print(f"Data split for {dataname.upper()}")

        num_training = int(config['data_split']['training'])
        num_validation = int(config['data_split']['validation'])
        num_testing = int(config['data_split']['testing'])

        # Input, output
        working_dir = os.path.join(output_dir, config_name)
        data_dir = os.path.join(working_dir, 'processed')

        # Check if data is processed
        assert os.path.exists(data_dir), "processed folder does not exists. Please process data first."

        # Set logging
        logging.getLogger('matplotlib.font_manager').disabled = True
        logger.setLevel(logging.DEBUG)
        fh = logging.FileHandler(os.path.join(os.path.dirname(working_dir), 'data_split.log'), mode='a')
        fh.setLevel(logging.DEBUG)
        logger.addHandler(fh)
        logger.info("="*30)
        logger.info("Performing data split")
        logger.info(f"START TIME: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Get user list, only keep user with more than 5 sessions
        user_list_full = sorted(pd.read_csv(os.path.join(working_dir, 'users_all.csv'), usecols=['user_id'])['user_id'].unique().tolist())
        print(f">>> Total number of users: {len(user_list_full)}".upper())
        start_time = time.time()

        # Take only users with more than 5 down and 5 up session
        user_5up = []
        user_5down = []
        for user in user_list_full:
            if not os.path.isdir(os.path.join(data_dir, user, 'up')) or not os.path.isdir(os.path.join(data_dir, user, 'down')):
                continue
            if len(os.listdir(os.path.join(data_dir, user, 'up'))) >= 5:
                user_5up.append(user)
            if len(os.listdir(os.path.join(data_dir, user, 'down'))) >= 5:
                user_5down.append(user)
        user_list = [i for i in user_5down if i in user_5up]
        print("#user with more than 5 down", len(user_5down))
        print("#user with more than 5 up", len(user_5up))
        print("#user with more than 5 down & 5 up", len(user_list))
        assert len(user_list) == num_training + num_validation + num_testing, f"Number of users are change. There are {len(user_list)} users, which is not equal to {num_training} + {num_validation} + {num_testing}"

        # Follow HuMINet: In the FETA paper, data split is considered for each user. So for us, we split the same ratio as in the huminet and duonet: train:val:test = 70:15:15
        training_user_list, val_test_user_list = train_test_split(user_list, test_size=(num_validation + num_testing), train_size=num_training, shuffle=True, random_state=1234)
        validation_user_list, testing_user_list = train_test_split(val_test_user_list, test_size=num_testing, train_size=num_validation, shuffle=True, random_state=1234)
        user_lists = {'training': training_user_list, 
                    'validation': validation_user_list,
                    'testing': testing_user_list}
        assert not os.path.exists(os.path.join(working_dir, 'splits.pickle')), "Data splits already exists. Stopping."
        with open(os.path.join(working_dir, 'splits.pickle'), 'wb') as f:
            pickle.dump(user_lists, f)
        for k in ['training', 'validation', 'testing']:
            with open(os.path.join(working_dir, f'{k}_scroll_imu_data_all.pickle'), 'wb') as f:
                pickle.dump(user_lists[k], f)
        
        end_time = time.time()
        logger.info(f">>> Elapsed time {((end_time - start_time)/60):.4f} minutes".upper())
        logger.info(f"END TIME: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Remove the current logger handle
        while logger.hasHandlers():
            logger.removeHandler(logger.handlers[0])

    else:
        raise NotImplementedError


if __name__ == "__main__":
    # Global variable
    logger = logging.getLogger()
    data_metadata = None
    device_width_height = {"iPhone 6s Plus": (1080, 1920),
                           "iPhone 7 Plus": (1080, 1920),
                           "iPhone 8 Plus": (1080, 1920),
                           "iPhone 6s": (750, 1334),
                           "iPhone 7": (750, 1334),
                           "iPhone 8": (750, 1334),
                           "iPhone X": (1125, 2436),
                           "iPhone XS": (1125, 2436),
                           "iPhone XS Max": (1242, 2688)}
    user_map = None  # Mapping between userid and session id, etc.

    # These will be set in config
    scroll_sequence_len = -1
    imu_sequence_len = -1
    windowing_offset = -1
    float_format = ''

    # Folders
    data_dir = None
    working_dir = ''

    main(get_args())