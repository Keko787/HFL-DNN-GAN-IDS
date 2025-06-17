import os
import random
import pandas as pd
from sklearn.utils import shuffle

#---                 Constants               ---#

NUM_COLS = ['flow_duration', 'Duration', 'Header_Length', 'Rate', 'Drate', 'IAT']

CAT_COLS = [
        'Protocol Type', 'fin flag number', 'syn flag number', 'psh flag number',
        'HTTP', 'HTTPS', 'DNS', 'Telnet', 'SMTP', 'SSH', 'IRC',
        'TCP', 'UDP', 'DHCP', 'ARP', 'ICMP', 'IPv'
    ]

# Categorical Features
categorical_features = [
    "Protocol type",
    "fin_flag_number",
    "syn_flag_number",
    "psh_flag_number",
    "TCP",
    "UDP",
    "HTTP",
    "HTTPS",
    "DNS",
    "Telnet",
    "SMTP",
    "SSH",
    "IRC",
    "DHCP",
    "ARP",
    "ICMP",
    "IPv",
]

# Numerical Features
numerical_features = [
    "flow_duration",
    "Header_Length",
    "Duration",
    "Rate",
    "Drate",
    "IAT"
]

IRRELEVANT_FEATURES = ["Header_Length"]

# select the num cols that are relevant
RElEVANT_NUM_COLS = [col for col in NUM_COLS if col not in IRRELEVANT_FEATURES]


# ---                 Helper  Functions                   --- #

# def load_and_balance_data(file_path, label_class_dict, current_benign_size, benign_size_limit):
#
#     data = pd.read_csv(file_path)
#
#     data = map_labels(data, label_class_dict)
#
#     attack_samples = data[data['label'] == 'Attack']
#     benign_samples = data[data['label'] == 'Benign']
#
#     remaining_benign_quota = benign_size_limit - current_benign_size
#     if len(benign_samples) > remaining_benign_quota:
#         benign_samples = benign_samples.sample(remaining_benign_quota, random_state=47)
#
#     min_samples = min(len(attack_samples), len(benign_samples))
#     balanced_data = pd.concat([attack_samples.sample(min_samples, random_state=47),
#                                benign_samples.sample(min_samples, random_state=47)])
#
#     return balanced_data, len(benign_samples)


def load_data(file_path, current_size=None, data_limit=None):
    """
    Loads data from a CSV file and optionally samples a specified number of rows.

    Parameters:
        file_path (str): Path to the CSV file.
        data_limit (int, optional): Number of rows to sample from the data.
                                    If None, the entire dataset is returned.

    Returns:
        pd.DataFrame: The loaded (and possibly sampled) data.
        int: A count of the total number of samples in a file.
    """
    data = pd.read_csv(file_path)

    if data_limit is not None:
        remaining_quota = data_limit - current_size

        if len(data) > remaining_quota:
            data = data.sample(n=remaining_quota, random_state=47)

    return data, len(data)

###################################################################################
#               Load Config for CICIOT 2023 Dataset                            #
###################################################################################
def loadLiveCaptureData(verbose=True, sample_size=1, dataset_size=1000):

    # INIT
    DATASET_DIRECTORY = '../../datasets/LIVEDATA/CSV'

    # --- File Paths for samples --- #
    if verbose:
        print("\n === Loading Network Traffic Data Files ===\n")

    # List and sorts the files in the dataset directory
    csv_filepaths = sorted([filename for filename in os.listdir(DATASET_DIRECTORY) if filename.endswith('.csv')])
    print(csv_filepaths)
    liveData_files = random.sample(csv_filepaths, sample_size)

    if verbose:
        print("\nLive Data File Sets:\n", liveData_files, "\n")

    # --- Load Train Data Samples from files --- #
    if verbose:
        print("\n-- Loading Live Data --\n")

    live_data = pd.DataFrame()
    dataset_size_count = 0

    for file in liveData_files:

        # break loop once limit is reached
        if dataset_size_count >= dataset_size:
            break

        if verbose:
            print(f"\nLive dataset sample: {file}")

        data, data_count = load_data(os.path.join(DATASET_DIRECTORY, file), dataset_size_count, dataset_size)

        live_data = pd.concat([live_data, data])

        dataset_size_count += data_count

        if verbose:
            print(
                f"Live Data Samples | Samples in File: {data_count} | Total: {dataset_size_count} | LIMIT: {dataset_size}")

    return live_data, IRRELEVANT_FEATURES
