import os
import pandas as pd
import random

from sklearn.model_selection import train_test_split


# Function to load all files from a given directory
def load_files_from_directory(directory, file_extension=".csv", sample_size=None):
    dataframes = []
    all_files = []

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(file_extension):
                file_path = os.path.join(root, file)
                all_files.append(file_path)

    # Shuffle and sample files if sample_size is specified
    if sample_size is not None:
        random.shuffle(all_files)
        all_files = all_files[:sample_size]
        print("Sample Selected:", all_files)

    for file_path in all_files:
        df = pd.read_csv(file_path)  # Modify this line if files are in a different format
        dataframes.append(df)

    print("Files Loaded...")
    return dataframes

# Function to split a DataFrame into train and test sets
def split_train_test(dataframe, test_size=0.2):
    train_df, test_df = train_test_split(dataframe, test_size=test_size)
    return train_df, test_df

# Function to combine subcategories into general classes
def combine_general_attacks(ddos_dataframes, dos_dataframes, scan_dataframes, theft_dataframes):
    ddos_combined = pd.concat(ddos_dataframes, ignore_index=True)
    dos_combined = pd.concat(dos_dataframes, ignore_index=True)
    scan_combined = pd.concat(scan_dataframes, ignore_index=True)
    theft_combined = pd.concat(theft_dataframes, ignore_index=True)
    return ddos_combined, dos_combined, scan_combined, theft_combined

# Function to combine all dataframes into one
def combine_all_attacks(dataframes):
    combined_df = pd.concat(dataframes, ignore_index=True)
    return combined_df

sample_size = 1

# Load DDoS UDP files
ddos_udp_directory = '../iotbotnet2020_archive/ddos/DDOS_UDP'
ddos_udp_dataframes = load_files_from_directory(ddos_udp_directory, sample_size=sample_size)

# Load DDoS TCP files
ddos_tcp_directory = '../iotbotnet2020_archive/ddos/DDOS_TCP'
ddos_tcp_dataframes = load_files_from_directory(ddos_tcp_directory, sample_size=sample_size)

# Load DDoS HTTP files
ddos_http_directory = '../iotbotnet2020_archive/ddos/DDOS_HTTP'
ddos_http_dataframes = load_files_from_directory(ddos_http_directory)

# Load DoS UDP files
dos_udp_directory = '../iotbotnet2020_archive/dos/dos_udp'
dos_udp_dataframes = load_files_from_directory(dos_udp_directory, sample_size=sample_size)

# Load DDoS TCP files
dos_tcp_directory = '../iotbotnet2020_archive/dos/dos_tcp'
dos_tcp_dataframes = load_files_from_directory(dos_tcp_directory, sample_size=sample_size)

# Load DDoS HTTP files
dos_http_directory = '../iotbotnet2020_archive/dos/dos_http'
dos_http_dataframes = load_files_from_directory(dos_http_directory)

# Load scan_os files
scan_os_directory = '../iotbotnet2020_archive/scan/os'
scan_os_dataframes = load_files_from_directory(scan_os_directory)

# Load scan_service files
scan_service_directory = '../iotbotnet2020_archive/scan/service'
scan_service_dataframes = load_files_from_directory(scan_service_directory)

# Load theft_data_exfiltration files
theft_data_exfiltration_directory = '../iotbotnet2020_archive/theft/data_exfiltration'
theft_data_exfiltration_dataframes = load_files_from_directory(theft_data_exfiltration_directory)

# Load theft_keylogging files
theft_keylogging_directory = '../iotbotnet2020_archive/theft/keylogging'
theft_keylogging_dataframes = load_files_from_directory(theft_keylogging_directory)

# Optionally, concatenate all dataframes if needed
ddos_udp_data = pd.concat(ddos_udp_dataframes, ignore_index=True)
ddos_tcp_data = pd.concat(ddos_tcp_dataframes, ignore_index=True)
ddos_http_data = pd.concat(ddos_http_dataframes, ignore_index=True)
dos_udp_data = pd.concat(dos_udp_dataframes, ignore_index=True)
dos_tcp_data = pd.concat(dos_tcp_dataframes, ignore_index=True)
dos_http_data = pd.concat(dos_http_dataframes, ignore_index=True)
scan_os_data = pd.concat(scan_os_dataframes, ignore_index=True)
scan_service_data = pd.concat(scan_service_dataframes, ignore_index=True)
theft_data_exfiltration_data = pd.concat(theft_data_exfiltration_dataframes, ignore_index=True)
theft_keylogging_data = pd.concat(theft_keylogging_dataframes, ignore_index=True)

# Combine subcategories into general classes
ddos_combined, dos_combined, scan_combined, theft_combined = combine_general_attacks(
    [ddos_udp_data, ddos_tcp_data, ddos_http_data],
    [dos_udp_data, dos_tcp_data, dos_http_data],
    [scan_os_data, scan_service_data],
    [theft_data_exfiltration_data, theft_keylogging_data]
)

# Combine all attacks into one DataFrame
all_attacks_combined = combine_all_attacks([
    ddos_combined, dos_combined, scan_combined, theft_combined
])

# Split each combined DataFrame into train and test sets
ddos_train, ddos_test = split_train_test(ddos_combined)
dos_train, dos_test = split_train_test(dos_combined)
scan_train, scan_test = split_train_test(scan_combined)
theft_train, theft_test = split_train_test(theft_combined)
all_attacks_train, all_attacks_test = split_train_test(all_attacks_combined)

# Display the first few rows of the combined DataFrames
# print("DDoS UDP Data:")
# print(ddos_udp_data.head())
#
# print("DDoS TCP Data:")
# print(ddos_tcp_data.head())
#
# print("DDoS HTTP Data:")
# print(ddos_http_data.head())
#
# print("DoS UDP Data:")
# print(dos_udp_data.head())
#
# print("DoS TCP Data:")
# print(dos_tcp_data.head())
#
# print("DoS HTTP Data:")
# print(dos_http_data.head())
#
# print("Scan OS Data:")
# print(scan_os_data.head())
#
# print("Scan Service Data:")
# print(scan_service_data.head())
#
# print("Theft Data Exfiltration Data:")
# print(theft_data_exfiltration_data.head())
#
# print("Theft Keylogging Data:")
# print(theft_keylogging_data.head())

# Display the first few rows of each combined DataFrame
print("DDoS Combined Data (Train):")
print(ddos_train.head())

print("DDoS Combined Data (Test):")
print(ddos_test.head())

print("DoS Combined Data (Train):")
print(dos_train.head())

print("DoS Combined Data (Test):")
print(dos_test.head())

print("Scan Combined Data (Train):")
print(scan_train.head())

print("Scan Combined Data (Test):")
print(scan_test.head())

print("Theft Combined Data (Train):")
print(theft_train.head())

print("Theft Combined Data (Test):")
print(theft_test.head())

print("All Attacks Combined Data (Train):")
print(all_attacks_train.head())

print("All Attacks Combined Data (Test):")
print(all_attacks_test.head())

# Save the combined DataFrames to new CSV files if needed
# all_ddos_udp_data.to_csv('combined_ddos_udp_data.csv', index=False)
# all_dos_udp_data.to_csv('combined_dos_udp_data.csv', index=False)