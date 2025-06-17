import os
import random
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

#########################################################
#    DATASET LOADING                               #
#########################################################

# ---                   IOTBOTNET relevant features/attribute mappings CONSTANTS                   --- #
SAMPLE_SIZE = 1
DATASET_DIRECTORY_BASE = '/root/datasets/IOTBOTNET2020'

# RELEVANT_ATTRIBUTES_IOTBOTNET = [
#         'Src_Port', 'Pkt_Size_Avg', 'Bwd_Pkts/s', 'Pkt_Len_Mean', 'Dst_Port', 'Bwd_IAT_Max', 'Flow_IAT_Mean',
#         'ACK_Flag_Cnt', 'Flow_Duration', 'Flow_IAT_Max', 'Flow_Pkts/s', 'Fwd_Pkts/s', 'Bwd_IAT_Tot', 'Bwd_Header_Len',
#         'Bwd_IAT_Mean', 'Bwd_Seg_Size_Avg', 'Label'
#     ]

RELEVANT_FEATURES_IOTBOTNET = [
    'Src_Port', 'Pkt_Size_Avg', 'Bwd_Pkts/s', 'Pkt_Len_Mean', 'Dst_Port', 'Bwd_IAT_Max', 'Flow_IAT_Mean',
    'ACK_Flag_Cnt', 'Flow_Duration', 'Flow_IAT_Max', 'Flow_Pkts/s', 'Fwd_Pkts/s', 'Bwd_IAT_Tot', 'Bwd_Header_Len',
    'Bwd_IAT_Mean', 'Bwd_Seg_Size_Avg'
]
RELEVANT_ATTRIBUTES_IOTBOTNET = RELEVANT_FEATURES_IOTBOTNET + ['Label']

ATTACK_RATIO = 0.25  # Desired imbalance ratio in test set


# ---                 Helper  Functions                   --- #

def load_files_from_directory(directory, file_extension=".csv", sample_size=None):
    """Loads files from a directory with optional sampling."""
    if not os.path.exists(directory):
        print(f"Directory '{directory}' does not exist.")
        return []

    all_files = [os.path.join(root, file)
                 for root, _, files in os.walk(directory)
                 for file in files if file.endswith(file_extension)]

    if sample_size is not None:
        random.shuffle(all_files)
        all_files = all_files[:sample_size]
        print("Sample(s) Selected:", all_files)

    dataframes = [pd.read_csv(file_path) for file_path in all_files]
    return dataframes


def split_train_test(dataframe, test_size=0.2):
    """Splits a DataFrame into training and testing sets."""
    return train_test_split(dataframe, test_size=test_size)


def combine_general_attacks(ddos_dataframes, dos_dataframes, scan_dataframes, theft_dataframes):
    """Combines subcategories into general classes."""
    return (
        pd.concat(ddos_dataframes, ignore_index=True),
        pd.concat(dos_dataframes, ignore_index=True),
        pd.concat(scan_dataframes, ignore_index=True),
        pd.concat(theft_dataframes, ignore_index=True)
    )


def combine_all_attacks(dataframes):
    """Combines all attack dataframes into a single DataFrame."""
    return pd.concat(dataframes, ignore_index=True)


def balance_data(dataframe, label_column='Label'):
    """Balances dataset via undersampling."""
    label_counts = dataframe[label_column].value_counts()
    min_count = label_counts.min()

    balanced_dataframes = [
        dataframe[dataframe[label_column] == label].sample(min_count, random_state=47)
        for label in label_counts.index
    ]

    balanced_dataframe = pd.concat(balanced_dataframes)
    return shuffle(balanced_dataframe, random_state=47)


def reduce_attack_samples(data, attack_ratio):
    """Reduces attack samples to achieve desired imbalance."""
    attack_samples = data[data['Label'] == 'Anomaly']
    benign_samples = data[data['Label'] == 'Normal']
    reduced_attack_samples = attack_samples.sample(frac=attack_ratio, random_state=47)
    return pd.concat([benign_samples, reduced_attack_samples])


###################################################################################
#               Load Config for IOTBOTNET 2020 Dataset                            #
###################################################################################
def loadIOTBOTNET(poisonedDataType=None):

    # Select dataset directory based on poisoned data type
    DATASET_DIRECTORY = f"{DATASET_DIRECTORY_BASE}_POISONED{poisonedDataType}" if poisonedDataType else DATASET_DIRECTORY_BASE

    # --- Load Each Attack Dataset --- #

    print("\nLoading Each IOT Network Attack Class...")
    print("Loading DDOS Data...")
    ddos_udp_dataframes = load_files_from_directory(f"{DATASET_DIRECTORY}/ddos/ddos_udp", sample_size=SAMPLE_SIZE)
    ddos_tcp_dataframes = load_files_from_directory(f"{DATASET_DIRECTORY}/ddos/ddos_tcp", sample_size=SAMPLE_SIZE)
    ddos_http_dataframes = load_files_from_directory(f"{DATASET_DIRECTORY}/ddos/ddos_http")

    print("Loading DOS Data...")
    dos_udp_dataframes = load_files_from_directory(f"{DATASET_DIRECTORY}/dos/dos_udp", sample_size=SAMPLE_SIZE)
    dos_tcp_dataframes = load_files_from_directory(f"{DATASET_DIRECTORY}/dos/dos_tcp", sample_size=SAMPLE_SIZE)
    dos_http_dataframes = load_files_from_directory(f"{DATASET_DIRECTORY}/dos/dos_http")

    print("Loading SCAN Data...")
    scan_os_dataframes = load_files_from_directory(f"{DATASET_DIRECTORY}/scan/os")
    scan_service_dataframes = load_files_from_directory(f"{DATASET_DIRECTORY}/scan/service")

    print("Loading THEFT Data...")
    theft_data_exfiltration_dataframes = load_files_from_directory(f"{DATASET_DIRECTORY}/theft/data_exfiltration")
    theft_keylogging_dataframes = load_files_from_directory(f"{DATASET_DIRECTORY}/theft/keylogging")

    print("Loading Finished...")

    # --- Combine All Classes --- #

    print("\nCombining Attack Data...")
    ddos_combined, dos_combined, scan_combined, theft_combined = combine_general_attacks(
        [pd.concat(ddos_udp_dataframes, ignore_index=True),
         pd.concat(ddos_tcp_dataframes, ignore_index=True),
         pd.concat(ddos_http_dataframes, ignore_index=True)],
        [pd.concat(dos_udp_dataframes, ignore_index=True),
         pd.concat(dos_tcp_dataframes, ignore_index=True),
         pd.concat(dos_http_dataframes, ignore_index=True)],
        [pd.concat(scan_os_dataframes, ignore_index=True),
         pd.concat(scan_service_dataframes, ignore_index=True)],
        [pd.concat(theft_data_exfiltration_dataframes, ignore_index=True),
         pd.concat(theft_keylogging_dataframes, ignore_index=True)]
    )

    all_attacks_combined = combine_all_attacks([ddos_combined, dos_combined, scan_combined, theft_combined])
    print("Attack Data Loaded & Combined...")

    # --- Preprocessing --- #

    print("\nCleaning Dataset...")
    all_attacks_combined.replace([float('inf'), -float('inf')], float('nan'), inplace=True)
    all_attacks_combined.dropna(inplace=True)

    # --- Train Test Split --- #

    print("\nTrain Test Split...")
    all_attacks_train, all_attacks_test = split_train_test(all_attacks_combined)

    print("Balance Training Dataset...")
    all_attacks_train = balance_data(all_attacks_train, label_column='Label')

    print("Balance Testing Dataset...")
    all_attacks_test = balance_data(all_attacks_test, label_column='Label')

    print("Adjusting Test Set Representation...")
    all_attacks_test = reduce_attack_samples(all_attacks_test, ATTACK_RATIO)

    return all_attacks_train, all_attacks_test, RELEVANT_FEATURES_IOTBOTNET
