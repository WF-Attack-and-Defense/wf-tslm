import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from utils.timefeatures import time_features
from data_provider.m4 import M4Dataset, M4Meta
import warnings

warnings.filterwarnings('ignore')


class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', percent=100,
                 seasonal_patterns=None):
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.percent = percent
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        # self.percent = percent
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

        self.enc_in = self.data_x.shape[-1]
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.set_type == 0:
            border2 = (border2 - self.seq_len) * self.percent // 100 + self.seq_len

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp


    def __getitem__(self, index):
        feat_id = index // self.tot_len
        s_begin = index % self.tot_len

        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data_x[s_begin:s_end, feat_id:feat_id + 1]
        seq_y = self.data_y[r_begin:r_end, feat_id:feat_id + 1]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return (len(self.data_x) - self.seq_len - self.pred_len + 1) * self.enc_in

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t', percent=100,
                 seasonal_patterns=None):
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.percent = percent
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

        self.enc_in = self.data_x.shape[-1]
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.set_type == 0:
            border2 = (border2 - self.seq_len) * self.percent // 100 + self.seq_len

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        feat_id = index // self.tot_len
        s_begin = index % self.tot_len

        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data_x[s_begin:s_end, feat_id:feat_id + 1]
        seq_y = self.data_y[r_begin:r_end, feat_id:feat_id + 1]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return (len(self.data_x) - self.seq_len - self.pred_len + 1) * self.enc_in

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', percent=100,
                 seasonal_patterns=None):
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.percent = percent

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

        self.enc_in = self.data_x.shape[-1]
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.set_type == 0:
            border2 = (border2 - self.seq_len) * self.percent // 100 + self.seq_len

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        feat_id = index // self.tot_len
        s_begin = index % self.tot_len

        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data_x[s_begin:s_end, feat_id:feat_id + 1]
        seq_y = self.data_y[r_begin:r_end, feat_id:feat_id + 1]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return (len(self.data_x) - self.seq_len - self.pred_len + 1) * self.enc_in

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_M4(Dataset):
    def __init__(self, root_path, flag='pred', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=False, inverse=False, timeenc=0, freq='15min',
                 seasonal_patterns='Yearly'):
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.root_path = root_path

        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]

        self.seasonal_patterns = seasonal_patterns
        self.history_size = M4Meta.history_size[seasonal_patterns]
        self.window_sampling_limit = int(self.history_size * self.pred_len)
        self.flag = flag

        self.__read_data__()

    def __read_data__(self):
        # M4Dataset.initialize()
        if self.flag == 'train':
            dataset = M4Dataset.load(training=True, dataset_file=self.root_path)
        else:
            dataset = M4Dataset.load(training=False, dataset_file=self.root_path)
        training_values = np.array(
            [v[~np.isnan(v)] for v in
             dataset.values[dataset.groups == self.seasonal_patterns]])  # split different frequencies
        self.ids = np.array([i for i in dataset.ids[dataset.groups == self.seasonal_patterns]])
        self.timeseries = [ts for ts in training_values]

    def __getitem__(self, index):
        insample = np.zeros((self.seq_len, 1))
        insample_mask = np.zeros((self.seq_len, 1))
        outsample = np.zeros((self.pred_len + self.label_len, 1))
        outsample_mask = np.zeros((self.pred_len + self.label_len, 1))  # m4 dataset

        sampled_timeseries = self.timeseries[index]
        cut_point = np.random.randint(low=max(1, len(sampled_timeseries) - self.window_sampling_limit),
                                      high=len(sampled_timeseries),
                                      size=1)[0]

        insample_window = sampled_timeseries[max(0, cut_point - self.seq_len):cut_point]
        insample[-len(insample_window):, 0] = insample_window
        insample_mask[-len(insample_window):, 0] = 1.0
        outsample_window = sampled_timeseries[
                           cut_point - self.label_len:min(len(sampled_timeseries), cut_point + self.pred_len)]
        outsample[:len(outsample_window), 0] = outsample_window
        outsample_mask[:len(outsample_window), 0] = 1.0
        return insample, outsample, insample_mask, outsample_mask

    def __len__(self):
        return len(self.timeseries)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    def last_insample_window(self):
        """
        The last window of insample size of all timeseries.
        This function does not support batching and does not reshuffle timeseries.

        :return: Last insample window of all timeseries. Shape "timeseries, insample size"
        """
        insample = np.zeros((len(self.timeseries), self.seq_len))
        insample_mask = np.zeros((len(self.timeseries), self.seq_len))
        for i, ts in enumerate(self.timeseries):
            ts_last_window = ts[-self.seq_len:]
            insample[i, -len(ts):] = ts_last_window
            insample_mask[i, -len(ts):] = 1.0
        return insample, insample_mask


class Dataset_Traffic_Classification(Dataset):
    """
    Dataset for Tor network traffic classification.
    CSV format: First column is class label, rest columns are traffic data (+1 for incoming, -1 for outgoing)
    Each row represents one sample with class label and traffic features.
    """
    @staticmethod
    def extract_traffic_direction_features(data):
        """
        Extract meaningful features from packet direction sequences.
        
        Args:
            data: numpy array of shape (n_samples, n_features) with values -1 (outgoing) or +1 (incoming)
        
        Returns:
            numpy array of shape (n_samples, n_extracted_features) with extracted features:
            - direction_changes: Number of transitions between incoming/outgoing
            - incoming_bursts: Number of consecutive incoming packet bursts
            - outgoing_bursts: Number of consecutive outgoing packet bursts
            - incoming_ratio: Ratio of incoming packets to total packets
            - outgoing_ratio: Ratio of outgoing packets to total packets
            - avg_incoming_burst: Average length of incoming bursts
            - avg_outgoing_burst: Average length of outgoing bursts
            - change_frequency: Frequency of direction changes
            - max_incoming_burst: Maximum length of incoming burst
            - max_outgoing_burst: Maximum length of outgoing burst
        """
        features = []
        for row in data:
            # Ensure values are exactly -1 or +1
            row = np.where(row > 0, 1, -1)
            
            # Count direction changes (transitions between incoming/outgoing)
            direction_changes = np.sum(np.diff(row) != 0)
            
            # Create masks for incoming and outgoing packets
            incoming_mask = (row == 1).astype(int)
            outgoing_mask = (row == -1).astype(int)
            
            # Count burst starts (transitions from 0 to 1 in mask)
            incoming_bursts = np.sum(np.diff(np.concatenate(([0], incoming_mask, [0]))) == 1)
            outgoing_bursts = np.sum(np.diff(np.concatenate(([0], outgoing_mask, [0]))) == 1)
            
            # Packet direction ratios
            total_packets = len(row)
            incoming_count = np.sum(incoming_mask)
            outgoing_count = np.sum(outgoing_mask)
            incoming_ratio = incoming_count / total_packets if total_packets > 0 else 0
            outgoing_ratio = outgoing_count / total_packets if total_packets > 0 else 0
            
            # Calculate burst lengths
            incoming_burst_lengths = []
            outgoing_burst_lengths = []
            current_incoming = 0
            current_outgoing = 0
            
            for val in row:
                if val == 1:  # Incoming
                    current_incoming += 1
                    if current_outgoing > 0:
                        outgoing_burst_lengths.append(current_outgoing)
                        current_outgoing = 0
                elif val == -1:  # Outgoing
                    current_outgoing += 1
                    if current_incoming > 0:
                        incoming_burst_lengths.append(current_incoming)
                        current_incoming = 0
            
            # Add final bursts
            if current_incoming > 0:
                incoming_burst_lengths.append(current_incoming)
            if current_outgoing > 0:
                outgoing_burst_lengths.append(current_outgoing)
            
            # Calculate average and max burst lengths
            avg_incoming_burst = np.mean(incoming_burst_lengths) if incoming_burst_lengths else 0
            avg_outgoing_burst = np.mean(outgoing_burst_lengths) if outgoing_burst_lengths else 0
            max_incoming_burst = np.max(incoming_burst_lengths) if incoming_burst_lengths else 0
            max_outgoing_burst = np.max(outgoing_burst_lengths) if outgoing_burst_lengths else 0
            
            # Direction change frequency
            change_frequency = direction_changes / total_packets if total_packets > 0 else 0
            
            # Additional features: variance of burst lengths
            var_incoming_burst = np.var(incoming_burst_lengths) if len(incoming_burst_lengths) > 1 else 0
            var_outgoing_burst = np.var(outgoing_burst_lengths) if len(outgoing_burst_lengths) > 1 else 0
            
            features.append([
                direction_changes,
                incoming_bursts,
                outgoing_bursts,
                incoming_ratio,
                outgoing_ratio,
                avg_incoming_burst,
                avg_outgoing_burst,
                change_frequency,
                max_incoming_burst,
                max_outgoing_burst,
                var_incoming_burst,
                var_outgoing_burst
            ])
        
        return np.array(features)
    
    def __init__(self, root_path, flag='train', size=None,
                 features='M', data_path='traffic.csv',
                 scale=True, timeenc=0, freq='h', percent=100,
                 seasonal_patterns=None, use_extracted_features=True):
        if size == None:
            self.seq_len = 512
            self.label_len = 48
            self.pred_len = 96
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.percent = percent
        self.use_extracted_features = use_extracted_features

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

        self.enc_in = self.data_x.shape[-1]

    def __read_data__(self):
        # No scaler needed - we preserve packet direction values (-1/+1)
        csv_path = os.path.join(self.root_path, self.data_path)
        
        # Check if file exists
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}. Please ensure the file exists and contains data.")
        
        # Check if file is empty
        file_size = os.path.getsize(csv_path)
        if file_size == 0:
            raise ValueError(
                f"CSV file is empty (0 bytes): {csv_path}\n"
                f"Please populate the file with traffic classification data.\n"
                f"Expected format:\n"
                f"  - First column: class labels\n"
                f"  - Remaining columns: traffic features (+1 for incoming, -1 for outgoing)\n"
                f"  - Each row represents one sample"
            )
        
        try:
            df_raw = pd.read_csv(csv_path)
        except pd.errors.EmptyDataError:
            raise ValueError(f"CSV file has no data: {csv_path}")
        
        # Check if dataframe is empty or has no columns
        if df_raw.empty:
            raise ValueError(f"CSV file is empty or has no data rows: {csv_path}")
        
        if len(df_raw.columns) < 2:
            raise ValueError(f"CSV file must have at least 2 columns (class label + features). Found {len(df_raw.columns)} columns in {csv_path}")

        # First column is class label, rest are traffic features
        class_column = df_raw.columns[0]
        feature_columns = df_raw.columns[1:]
        
        # Extract class labels and convert to numeric
        labels_raw = df_raw[class_column].values
        # Create label mapping if labels are strings
        unique_labels = np.unique(labels_raw)
        self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        self.num_classes = len(unique_labels)
        label_indices = np.array([self.label_to_idx[label] for label in labels_raw])
        
        # Extract traffic features (should be +1 or -1)
        df_data = df_raw[feature_columns]
        
        # Use stratified splitting to ensure each class is represented in train/val/test
        # First split: 70% train, 30% temp (val + test)
        X_train, X_temp, y_train, y_temp = train_test_split(
            df_data.values, label_indices,
            test_size=0.3,
            stratify=label_indices,
            random_state=42
        )
        
        # Second split: split temp into 50% val and 50% test (15% each of total)
        # This gives us: 70% train, 15% val, 15% test
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=0.5,
            stratify=y_temp,
            random_state=42
        )
        
        # Select the appropriate split based on set_type
        if self.set_type == 0:  # train
            X_split = X_train
            y_split = y_train
            # Apply percent for training data
            if self.percent < 100:
                num_samples = int(len(X_train) * self.percent / 100)
                X_split = X_train[:num_samples]
                y_split = y_train[:num_samples]
        elif self.set_type == 1:  # val
            X_split = X_val
            y_split = y_val
        else:  # test
            X_split = X_test
            y_split = y_test
        
        # Preserve packet direction values (-1/+1) instead of scaling
        # Clip values to ensure they are in [-1, 1] range and round to preserve -1/+1
        if self.scale:
            # Ensure values are in [-1, 1] range and preserve packet directions
            data = np.clip(X_split, -1, 1)
            # Round to exactly -1 or +1 to preserve packet direction semantics
            # Values > 0 become +1 (incoming), values <= 0 become -1 (outgoing)
            data = np.where(data > 0, 1, -1)
        else:
            # Keep raw values but ensure they're in valid range
            data = np.clip(X_split, -1, 1)
            data = np.where(data > 0, 1, -1)

        # Extract meaningful features from packet direction sequences
        if self.use_extracted_features:
            extracted_features = self.extract_traffic_direction_features(data)
            print(f"Extracted {extracted_features.shape[1]} traffic direction features from {len(data)} samples")
        else:
            extracted_features = None

        # Each row is a sample: reshape features to (seq_len, num_features)
        # If num_features < seq_len, we pad. If num_features > seq_len, we truncate or reshape
        num_features = data.shape[1]
        
        # Reshape each row's features into a sequence
        # Strategy: treat each feature as a time step, so we get (seq_len, num_channels)
        # If we have more features than seq_len, we can either truncate or reshape
        # If extracted features are used, we can add them as additional channels
        self.data_x = []
        self.data_y = []
        self.data_stamp = []
        
        # Determine number of channels (1 for raw data, +1 if using extracted features)
        num_channels = 1
        if self.use_extracted_features and extracted_features is not None:
            num_channels = 2  # Raw sequence + extracted features as additional channel
        
        for i in range(len(data)):
            features_row = data[i]  # Shape: (num_features,)
            label = y_split[i]
            
            # Reshape features to (seq_len, 1)
            # If num_features < seq_len, pad with zeros
            # If num_features > seq_len, truncate
            if num_features < self.seq_len:
                seq = np.zeros((self.seq_len, 1))
                seq[:num_features, 0] = features_row
            elif num_features > self.seq_len:
                seq = features_row[:self.seq_len].reshape(-1, 1)
            else:
                seq = features_row.reshape(-1, 1)
            
            # If using extracted features, add them as additional channel
            if self.use_extracted_features and extracted_features is not None:
                # Normalize extracted features to [-1, 1] range for consistency
                extracted_feat = extracted_features[i]  # Shape: (n_extracted_features,)
                
                # Normalize each feature to [-1, 1] range
                feat_min = extracted_feat.min()
                feat_max = extracted_feat.max()
                if feat_max > feat_min:
                    extracted_feat_normalized = 2 * (extracted_feat - feat_min) / (feat_max - feat_min) - 1
                else:
                    extracted_feat_normalized = extracted_feat
                
                # Repeat extracted features to match sequence length
                # Or use average/aggregation to create a single value per time step
                # Option 1: Repeat the mean of extracted features across sequence
                extracted_channel = np.full((self.seq_len, 1), np.mean(extracted_feat_normalized))
                
                # Option 2: Use first extracted feature (direction_changes) as additional channel
                # This provides direction change information at each time step
                # extracted_channel = np.full((self.seq_len, 1), extracted_feat_normalized[0])
                
                # Concatenate raw sequence and extracted features channel
                seq = np.concatenate([seq, extracted_channel], axis=1)  # Shape: (seq_len, 2)
            
            self.data_x.append(seq)
            self.data_y.append(label)
            # Create dummy time stamps (not used for classification but required by interface)
            self.data_stamp.append(np.zeros((self.seq_len, 4)))
        
        self.data_x = np.array(self.data_x)  # Shape: (num_samples, seq_len, num_channels)
        self.data_y = np.array(self.data_y)  # Shape: (num_samples,)
        self.data_stamp = np.array(self.data_stamp)  # Shape: (num_samples, seq_len, 4)
        
        # Store extracted features for potential use
        self.extracted_features = extracted_features if self.use_extracted_features else None

    def __getitem__(self, index):
        seq_x = self.data_x[index]  # (seq_len, 1)
        seq_y = self.data_y[index]  # class label (scalar)
        seq_x_mark = self.data_stamp[index]  # (seq_len, 4) dummy timestamps
        seq_y_mark = self.data_stamp[index]  # dummy timestamps
        
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x)

    def inverse_transform(self, data):
        # No inverse transform needed - data is already in original -1/+1 format
        # Return data as-is since we're not scaling packet direction values
        return data

