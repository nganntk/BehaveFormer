"""Datasets classes
BaseTrainDataset and BaseTestDataset are the 2 base clases for training and testing, used for triplet loss and BehaveFormer framework.

For a new dataset, you should inheritted from the 2 base class and implement your logic to load data in these functions:
- BaseTrainDataset: implement create_user_sess_seq, load_data
- BaseTestDataset: implement load_data
"""
import os
import math
import pickle
import numpy as np
from torch.utils.data import Dataset
import torch

class BaseTrainDataset(Dataset):
    """Train dataset that loads data during training"""
    def __init__(self, 
                 user_list: list = None, 
                 user_sess_seq: list[list[int]] = None, 
                 batch_size: int = -1, 
                 epoch_batch_count: int = -1, 
                 data_type='float64'):
        self.user_list = user_list
        self.user_sess_seq = user_sess_seq  # Use to get random user later
        # Format of user_sess_seq: nested list, same format as processed data but only store the number of sequences, not the sequence itself. 
        # This is needed because each user may have a different number of sessions and sequences
        assert len(user_list) == len(user_sess_seq)
        self.data_type = data_type

        self.batch_size = batch_size
        self.epoch_batch_count = epoch_batch_count

    def __len__(self):
        return self.batch_size * self.epoch_batch_count

    def __getitem__(self, idx):
        # Get 2 random users
        genuine_user_idx = np.random.randint(0, len(self.user_sess_seq))
        imposter_user_idx = np.random.randint(0, len(self.user_sess_seq))
        while (imposter_user_idx == genuine_user_idx):
            imposter_user_idx = np.random.randint(0, len(self.user_sess_seq))
        
        # Genuine: get 2 random sessions, imposter: get 1 random sessions
        genuine_sess_1 = np.random.randint(0, len(self.user_sess_seq[genuine_user_idx]))
        genuine_sess_2 = np.random.randint(0, len(self.user_sess_seq[genuine_user_idx]))
        while (genuine_sess_2 == genuine_sess_1):
            genuine_sess_2 = np.random.randint(0, len(self.user_sess_seq[genuine_user_idx]))
        imposter_sess = np.random.randint(0, len(self.user_sess_seq[imposter_user_idx]))
        
        genuine_seq_1 = np.random.randint(0, self.user_sess_seq[genuine_user_idx][genuine_sess_1])
        genuine_seq_2 = np.random.randint(0, self.user_sess_seq[genuine_user_idx][genuine_sess_2])
        imposter_seq = np.random.randint(0, self.user_sess_seq[imposter_user_idx][imposter_sess])

        anchor = self.load_data(genuine_user_idx, genuine_sess_1, genuine_seq_1)
        positive = self.load_data(genuine_user_idx, genuine_sess_2, genuine_seq_2)
        negative = self.load_data(imposter_user_idx, imposter_sess, imposter_seq)

        assert anchor[0].shape[1] == 8 and positive[0].shape[1] == 8 and negative[0].shape[1] == 8, f"scroll data must have dim 8, input shape is {anchor[0].shape}"

        return anchor, positive, negative

    def convert_type(self, single_sequence: list[np.array, np.array]):
        """Convert data to required data format, e.g., float64"""
        for i in range(len(single_sequence)):
            if single_sequence[i].dtype != self.data_type:
                single_sequence[i] = single_sequence[i].astype(self.data_type)
        return single_sequence

    def scale(self, single_sequence: list[np.array, np.array]):
        """Scaling input data"""
        raise NotImplementedError

    def create_user_sess_seq(self):
        """Creat indicies nested list here for each dataset"""
        raise NotImplementedError
    
    def load_data(self, user_idx: int, sess_idx: int, seq_idx: int):
        """Load data given user_idx, sess_idx, seq_idx. 
        This may be different from dataset to dataset."""
        raise NotImplementedError

class BaseTestDataset(Dataset):
    """Test dataset that loads data during training"""
    def __init__(self,
                 num_users: int,
                 num_sessions: int,
                 num_seqs: int,
                 data_type: str='float64', 
                 ):
        self.data_type = data_type

        self.num_users = num_users
        self.num_sessions = num_sessions
        self.num_seqs = num_seqs

    def __len__(self):
        return math.ceil(self.num_users * self.num_sessions * self.num_seqs)

    def __getitem__(self, idx):
        t_session = idx // self.num_seqs
        user_idx = t_session // self.num_sessions
        session_idx = t_session % self.num_sessions
        seq_idx = idx % self.num_seqs

        return self.load_data(user_idx, session_idx, seq_idx)
   
    def convert_type(self, single_sequence: list[np.array, np.array]):
        """Convert data to required data format, e.g., float64"""
        for i in range(len(single_sequence)):
            if single_sequence[i].dtype != self.data_type:
                single_sequence[i] = single_sequence[i].astype(self.data_type)
        return single_sequence
    
    def scale(self, single_sequence: list[np.array, np.array]):
        """Scaling input data"""
        raise NotImplementedError
    
    def load_data(self):
        """Load data given user_idx, sess_idx, seq_idx. 
        This may be different from dataset to dataset."""
        raise NotImplementedError

class HUMITrainDataset(BaseTrainDataset):
    """NOTE: For HuMIdb, down and up session are store consecutively: s1down-s1up-s2down-s2up
    For HuMIdb: load all data in at the beginning
    The procesed data stores both down and up action so this function reformat the input data according to action_type so that we can use the same dataset function for all type of action_type.
    This is handled using self.session_idx
    
    Data format note:
        input_list (nested List): All input data
            - 1st dim: user
            - 2nd dim: sessions  (for case with both down and up, there will be 10 sessions: s1down-s1up-s2down-s2up-...)
            - 3nd dim: sequences
        action_type (str): Can be 'down', 'up', 'downup', 'downupcode'
            down: use only scroll down action
            up: use only scroll up action
            downup: use both scroll down and scroll up (order of sessions: down-up-down-up-...)
        num_sequences: Number of sequence to use for each session. This is used to reduce run time for ML models. Pass -1 to take all sequences
    """
    IMU_COLS = {'acc_gyr_mag': list(range(36)),
                'acc_gyr': list(range(24)),
                'acc_mag': list(range(12)) + list(range(24, 36)),
                'gyr_mag': list(range(12, 36)),
                'acc': list(range(12)),
                'gyr': list(range(12, 24)),
                'mag': list(range(24, 36))}

    def __init__(self, 
                 batch_size: int, 
                 epoch_batch_count: int, 
                 action: str, 
                 training_file: str,
                 user_list: list, 
                 user_sess_seq: list[list[int]] = None, 
                 imu_type: str = 'none', 
                 imu_cols: list = None,
                 data_type: str='float64'):
        self.training_file = training_file
        if action == 'down':
            self.session_idx = [(i * 2) for i in range(5)]
        elif action == 'up':
            self.session_idx = [(i * 2 + 1) for i in range(5)]
        else:
            self.session_idx = list(range(10))
        if user_sess_seq is None:
            self.create_user_sess_seq()
        self.dataset_name = 'HuMIdb'
        
        super().__init__(user_list, self.user_sess_seq, batch_size, epoch_batch_count, data_type)

        self.data = None
        self.load_data_all()
        
        self.imu_type = imu_type
        if self.imu_type != 'none' and imu_cols is None:
            self.imu_cols = self.IMU_COLS[self.imu_type]

    def create_user_sess_seq(self):
        """Creat indicies nested list here for each dataset"""
        self.user_sess_seq = []
        with open(self.training_file, 'rb') as f:
            data = pickle.load(f)
        for user in data:   # humidb: down and up are consecutive
            self.user_sess_seq.append([len(user[sess_idx])for sess_idx in self.session_idx])
    
    def load_data(self, user_idx, sess_idx, seq_idx):
        """Load data given user_idx, sess_idx, seq_idx. 
        This may be different from dataset to dataset."""
        # Retrieve the session idx in the down-up consecutive
        _sess_idx = self.session_idx[sess_idx]

        assert len(self.data[user_idx][_sess_idx][seq_idx]) == 2, "Data must contain 2 elements: action, imu"
        
        ret_scroll = self.data[user_idx][_sess_idx][seq_idx][0][:, 1:-1]
        if self.imu_type == 'none':
            return [ret_scroll]
        else: # Take imu columns corresponding to imu type
            ret_imu = self.data[user_idx][_sess_idx][seq_idx][1][:, self.imu_cols]
            return [ret_scroll, ret_imu]

    def load_data_all(self):
        # For HuMIdb: Load all data in
        with open(self.training_file, 'rb') as f:
            self.data = pickle.load(f)

        # Convert and scale all sessions and sequence data
        for user in range(len(self.data)):
            for session in range(len(self.data[user])):
                for sequence in range(len(self.data[user][session])):
                    self.data[user][session][sequence] = self.scale(self.convert_type(self.data[user][session][sequence]))

    def scale(self, single_sequence: list[np.array, np.array]):
        """Scaling, use the same way as original BehaveFormer"""
        # For scroll, divide fft by 100 to scale to the same range as others
        for j in [3, 4]:  # time, x, y, fx, fy, x', y', x'', y''
            single_sequence[0][:, j] = single_sequence[0][:, j] / 100

        # Normalize IMU follows BehaveFormer
        # Remove very large values from fft values
        for j in [16, 17, 27, 28, 29]:   
            for k in range(100):
                if (single_sequence[1][k][j] >= 1000000):
                    single_sequence[1][k][j] = 0.0
                
        # IMU scaling: Same as BehaveFormer
        for j in range(36):
            if (j in [0,1,2]):
                single_sequence[1][:, j] = single_sequence[1][:, j] / 10
            elif (j in [3,4,5,24,25,26,27,28,29]):
                single_sequence[1][:, j] = single_sequence[1][:, j] / 1000
            elif (j in [15,16,17]):
                single_sequence[1][:, j] = single_sequence[1][:, j] / 1000
        return single_sequence


class HUMITestDataset(BaseTestDataset):
    """All data is loaded at the beginning"""
    IMU_COLS = {'acc_gyr_mag': list(range(36)),
                'acc_gyr': list(range(24)),
                'acc_mag': list(range(12)) + list(range(24, 36)),
                'gyr_mag': list(range(12, 36)),
                'acc': list(range(12)),
                'gyr': list(range(12, 24)),
                'mag': list(range(24, 36))}

    def __init__(self, 
                 action: str, 
                 validation_file: str,
                 imu_type: str='none',
                 imu_cols: list=None,
                 data_type: str = 'float64',
                 use_time: bool=False):
        self.action = action
        self.validation_file = validation_file

        self.dataset_name = 'HuMIdb'
        self.data_type = data_type
        self.use_time = use_time  # use to compute CA metrics
        
        self.data = None
        self.load_data_all()
        super().__init__(num_users=len(self.data), 
                         num_sessions=len(self.data[0]), 
                         num_seqs=len(self.data[0][0]), 
                         data_type=data_type)
        self.imu_type = imu_type
        if self.imu_type != 'none' and imu_cols is None:
            self.imu_cols = self.IMU_COLS[self.imu_type]

    def scale(self, single_sequence: list[np.array, np.array]):
        """Scaling, use the same way as original BehaveFormer"""
        # For scroll, divide fft by 100 to scale to the same range as others
        for j in [3, 4]:  # time, x, y, fx, fy, x', y', x'', y''
            single_sequence[0][:, j] = single_sequence[0][:, j] / 100

        # Normalize IMU follows BehaveFormer
        # Remove very large values from fft values
        for j in [16, 17, 27, 28, 29]:   
            for k in range(100):
                if (single_sequence[1][k][j] >= 1000000):
                    single_sequence[1][k][j] = 0.0
                
        # IMU scaling: Same as BehaveFormer
        for j in range(36):
            if (j in [0,1,2]):
                single_sequence[1][:, j] = single_sequence[1][:, j] / 10
            elif (j in [3,4,5,24,25,26,27,28,29]):
                single_sequence[1][:, j] = single_sequence[1][:, j] / 1000
            elif (j in [15,16,17]):
                single_sequence[1][:, j] = single_sequence[1][:, j] / 1000
        return single_sequence

    def load_data(self, user_idx, sess_idx, seq_idx):
        assert len(self.data[user_idx][sess_idx][seq_idx]) == 2, "Data must contain 2 elements: action, imu"

        ret_scroll = self.data[user_idx][sess_idx][seq_idx][0][:, 1:-1]
        if self.imu_type == 'none':
            return [ret_scroll]
        else:  # Take imu columns corresponding to imu type            
            ret_imu = self.data[user_idx][sess_idx][seq_idx][1][:, self.imu_cols]
            return [ret_scroll, ret_imu]

    def load_data_with_time(self, user_idx, sess_idx, seq_idx):
        assert len(self.data[user_idx][sess_idx][seq_idx]) == 2, "Data must contain 2 elements: action, imu"

        ret_scroll = self.data[user_idx][sess_idx][seq_idx][0][:, :-1]
        if self.imu_type == 'none':
            return [ret_scroll]
        else:
            ret_imu = self.data[user_idx][sess_idx][seq_idx][1][:, self.imu_cols]  # Take imu columns corresponding to imu type
            return [ret_scroll, ret_imu]

    def load_data_all(self):
        # For HuMIdb: Load all data in
        with open(self.validation_file, 'rb') as f:
            self.data = pickle.load(f)
               
        # Convert and scale all sessions and sequence data
        for user in range(len(self.data)):
            for session in range(len(self.data[user])):
                for sequence in range(len(self.data[user][session])):
                    self.data[user][session][sequence] = self.scale(self.convert_type(self.data[user][session][sequence]))
        
        if self.action == 'down':
            action_session = [(i * 2) for i in range(5)]
        elif self.action == 'up':
            action_session = [(i * 2 + 1) for i in range(5)]
        else:
            action_session = list(range(10))
        for user_idx, user in enumerate(self.data):
            self.data[user_idx] = [user[i] for i in action_session]
            for idx, session in enumerate(self.data[user_idx]):
                self.data[user_idx][idx] = session[:1]

class FETATrainDataset(BaseTrainDataset):
    DATASET_NAME = 'FETA'
    IMU_COLS = {'acc_gyr': list(range(24)),
                'acc': list(range(12)),
                'gyr': list(range(12, 24))}
    def __init__(self, 
                 data_root: str, 
                 batch_size: int, 
                 epoch_batch_count: int,
                 action: str,
                 user_list: list, 
                 user_sess_seq: list[list[int]] = None, 
                 imu_type: str='none', 
                 imu_cols: list = None,
                 data_type: str='float64'):
        self.dataset_name = self.DATASET_NAME
        
        self.data_root = data_root   # Folder contains user processed data, each user is in a folder with the userid
        self.action = action
        self.user_list = user_list
        self.user_sess_seq = user_sess_seq

        if self.user_sess_seq is None:
            self.create_user_sess_seq()
        self.imu_type = imu_type
        if self.imu_type != 'none' and imu_cols is None:
            self.imu_cols = self.IMU_COLS[self.imu_type]

        super().__init__(user_list, self.user_sess_seq, batch_size, epoch_batch_count, data_type)

    def create_user_sess_seq(self):
        """Creat indicies nested list here for each dataset"""
        self.user_sess_seq = []
        for user in self.user_list:
            session_list = sorted([int(i) for i in os.listdir(os.path.join(self.data_root, user, self.action))])
            sess_seq = [len(os.listdir(os.path.join(self.data_root, user, self.action, str(i)))) for i in session_list]

            self.user_sess_seq.append(sess_seq) 

    def scale(self, single_sequence: list[np.array, np.array]):
        """Scaling, use the same way as original BehaveFormer.
        Only use colums for acc and gyr because FETA only have acc and gyr"""
        # For scroll, divide fft by 100 to scale to the same range as others
        for j in [3, 4]:  # time, x, y, fx, fy, x', y', x'', y''
            single_sequence[0][:, j] = single_sequence[0][:, j] / 100

        # Normalize IMU follows BehaveFormer
        # Remove very large values from fft values
        for j in [16, 17]:   
            for k in range(100):
                if (single_sequence[1][k][j] >= 1000000):
                    single_sequence[1][k][j] = 0.0
                
        # IMU scaling: Same as BehaveFormer
        for j in range(36):
            if (j in [0,1,2]):
                single_sequence[1][:, j] = single_sequence[1][:, j] / 10
            elif (j in [3,4,5]):
                single_sequence[1][:, j] = single_sequence[1][:, j] / 1000
            elif (j in [15,16,17]):
                single_sequence[1][:, j] = single_sequence[1][:, j] / 1000
        return single_sequence
    
    def load_data(self, user_id, sess_idx, seq_idx):
        """Load data given user_idx, sess_idx, seq_idx. 
        This may be different from dataset to dataset.
        """
        with open(os.path.join(self.data_root, self.user_list[user_id], self.action, str(sess_idx), str(seq_idx), 'scroll_imu.pickle'), 'rb') as f:
            data = pickle.load(f)  # scroll, imu
        assert len(data) == 2, "Data must contain 2 elements: action, imu"
        
        ret = self.scale(self.convert_type(data))
        ret[0] = torch.from_numpy(ret[0][:, 1:-1])
        if self.imu_type == 'none':
            return [ret[0]]
        else: # Take imu columns corresponding to imu type
            ret[1] = torch.from_numpy(ret[1][:, self.imu_cols])
            return ret

class FETATestDataset(BaseTestDataset):
    """NOTE: FETATestDataset: each user has varying number of sessions. Meanwhile in Humidb, each user has 5 sessions"""
    DATASET_NAME = 'FETA'
    IMU_COLS = {'acc_gyr': list(range(24)),
                'acc': list(range(12)),
                'gyr': list(range(12, 24))}
    
    def __init__(self, 
                 data_root: str, 
                 action: str, 
                 user_list: list[str],
                 imu_type: str='none',
                 imu_cols: list=None,
                 data_type: str = 'float64',
                 ):
        self.dataset_name = self.DATASET_NAME
        self.data_root = data_root
        self.action = action
        self.data_type = data_type
        self.user_list = user_list
        self.data_len = 0
        self.idx_user_session = []
        self.user_session_count = []
        self.get_session_map()

        self.imu_type = imu_type
        if self.imu_type != 'none' and imu_cols is None:
            self.imu_cols = self.IMU_COLS[self.imu_type]

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        user_idx, session_idx = self.idx_user_session[idx]
        
        # use first seq_idx follows BehaveFormer
        return self.load_data(user_idx, session_idx, seq_idx=0)

    def get_session_map(self):
        """Subjects have different number of session so we need to do this
        e.g, 2 subjects, first one has 2 sessions, 2nd one has 3 sessions
        -> [00, 01, 10, 11, 12]
        """
        self.idx_user_session = []
        self.user_session_count = []   # [start index, number of session]
        for i_user, user in enumerate(self.user_list):
            num_sessions = len(os.listdir(os.path.join(self.data_root, user, self.action)))
            self.user_session_count.append([len(self.idx_user_session), num_sessions])
            self.idx_user_session.extend([(i_user, i_sess) for i_sess in range(num_sessions)])
        self.data_len = len(self.idx_user_session)

    def scale(self, single_sequence: list[np.array, np.array]):
        """Scaling, use the same way as original BehaveFormer.
        Only use colums for acc and gyr because FETA only have acc and gyr"""
        # For scroll, divide fft by 100 to scale to the same range as others
        for j in [3, 4]:  # time, x, y, fx, fy, x', y', x'', y''
            single_sequence[0][:, j] = single_sequence[0][:, j] / 100

        # Normalize IMU follows BehaveFormer
        # Remove very large values from fft values
        for j in [16, 17]:   
            for k in range(100):
                if (single_sequence[1][k][j] >= 1000000):
                    single_sequence[1][k][j] = 0.0
                
        # IMU scaling: Same as BehaveFormer
        for j in range(36):
            if (j in [0,1,2]):
                single_sequence[1][:, j] = single_sequence[1][:, j] / 10
            elif (j in [3,4,5]):
                single_sequence[1][:, j] = single_sequence[1][:, j] / 1000
            elif (j in [15,16,17]):
                single_sequence[1][:, j] = single_sequence[1][:, j] / 1000
        return single_sequence

    def load_data(self, user_idx: int, sess_idx: int, seq_idx: int):
        """FETA folder structure: user data in a folder named user_idx, each folder contains down.pickle, up.pickle, downup.pickle

        Args:
            user_idx (_type_): _description_
            sess_idx (_type_): _description_
            seq (_type_): _description_
            idx (_type_): _description_

        Returns:
            _type_: _description_
        """
        with open(os.path.join(self.data_root, self.user_list[user_idx], self.action, str(sess_idx), str(seq_idx), 'scroll_imu.pickle'), 'rb') as f:
            data = pickle.load(f)  # scroll, imu
        assert len(data) == 2, "Data must contain 2 elements: action, imu"
       
        ret = self.scale(self.convert_type(data))
        ret[0] = torch.from_numpy(ret[0][:, 1:-1])
        if self.imu_type == 'none':
            return [ret[0]]
        else: # Take imu columns corresponding to imu type
            ret[1] = torch.from_numpy(ret[1][:, self.imu_cols])
            return ret

    def load_data_with_time(self, user_idx: int, sess_idx: int, seq_idx: int):
        """FETA folder structure: user data in a folder named user_idx, each folder contains down.pickle, up.pickle, downup.pickle
        This function is similar to load_data, but return event_time as the first column, used to computed Continuous Authentication metrics

        Args:
            user_idx (_type_): _description_
            sess_idx (_type_): _description_
            seq (_type_): _description_
            idx (_type_): _description_

        Returns:
            _type_: _description_
        """
        with open(os.path.join(self.data_root, self.user_list[user_idx], self.action, str(sess_idx), str(seq_idx), 'scroll_imu.pickle'), 'rb') as f:
            data = pickle.load(f)  # scroll, imu
        assert len(data) == 2, "Data must contain 2 elements: action, imu"
       
        ret = self.scale(self.convert_type(data))
        ret[0] = torch.from_numpy(ret[0][:, :-1])
        if self.imu_type == 'none':
            return [ret[0]]
        else: # Take imu columns corresponding to imu type
            ret[1] = torch.from_numpy(ret[1][:, self.imu_cols])
            return ret