import os
import random

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
# Step 1: Load and Preprocess the Data
DATASET_DIRECTORY = '../ciciot2023_archive/'          # If your dataset is within your python project directory, change this to the relative path to your dataset

### Inputs ###
ciciot_sample_size = 2  # input: 2 at minimum
# label classes 33+1 7+1 1+1
ciciot_label_class = "1+1"

# List the files in the dataset
csv_filepaths = [filename for filename in os.listdir(DATASET_DIRECTORY) if filename.endswith('.csv')]
print(csv_filepaths)

# If there are more than X CSV files, randomly select X files from the list
if len(csv_filepaths) > ciciot_sample_size:
    csv_filepaths = random.sample(csv_filepaths, ciciot_sample_size)
    print(csv_filepaths)
csv_filepaths.sort()
split_index = int(len(csv_filepaths) * 0.5)

training_data_sets = csv_filepaths[:split_index]
test_data_sets = csv_filepaths[split_index:]

print("Training Sets:\n", training_data_sets, "\n")
print("Test Sets:\n", test_data_sets)

# Mapping Features
num_cols = [
    'flow_duration', 'Header_Length', 'Duration',
    'Rate', 'Srate', 'ack_count', 'syn_count',
    'fin_count', 'urg_count', 'rst_count', 'Tot sum',
    'Min', 'Max', 'AVG', 'Std', 'Tot size', 'IAT', 'Number',
    'Magnitue', 'Radius', 'Covariance', 'Variance', 'Weight',
]

cat_cols = [
    'Protocol Type', 'Drate', 'fin_flag_number', 'syn_flag_number', 'rst_flag_number',
    'psh_flag_number', 'ack_flag_number', 'ece_flag_number',
    'cwr_flag_number', 'HTTP', 'HTTPS', 'DNS', 'Telnet',
    'SMTP', 'SSH', 'IRC', 'TCP', 'UDP', 'DHCP', 'ARP',
    'ICMP', 'IPv', 'LLC', 'label'
]

irrelevant_features = ['ack_flag_number', 'ece_flag_number', 'cwr_flag_number', 'Magnitue', 'Radius', 'Covariance',
                       'Variance', 'flow_duration', 'Header_Length', 'urg_count', 'rst_count', 'Tot sum', 'Min',
                       'Max', 'AVG', 'Std']

# Mapping Labels
dict_7classes = {'DDoS-RSTFINFlood': 'DDoS', 'DDoS-PSHACK_Flood': 'DDoS', 'DDoS-SYN_Flood': 'DDoS',
                 'DDoS-UDP_Flood': 'DDoS', 'DDoS-TCP_Flood': 'DDoS', 'DDoS-ICMP_Flood': 'DDoS',
                 'DDoS-SynonymousIP_Flood': 'DDoS', 'DDoS-ACK_Fragmentation': 'DDoS', 'DDoS-UDP_Fragmentation': 'DDoS',
                 'DDoS-ICMP_Fragmentation': 'DDoS', 'DDoS-SlowLoris': 'DDoS', 'DDoS-HTTP_Flood': 'DDoS',
                 'DoS-UDP_Flood': 'DoS', 'DoS-SYN_Flood': 'DoS', 'DoS-TCP_Flood': 'DoS', 'DoS-HTTP_Flood': 'DoS',
                 'Mirai-greeth_flood': 'Mirai', 'Mirai-greip_flood': 'Mirai', 'Mirai-udpplain': 'Mirai',
                 'Recon-PingSweep': 'Recon', 'Recon-OSScan': 'Recon', 'Recon-PortScan': 'Recon',
                 'VulnerabilityScan': 'Recon', 'Recon-HostDiscovery': 'Recon', 'DNS_Spoofing': 'Spoofing',
                 'MITM-ArpSpoofing': 'Spoofing', 'BenignTraffic': 'Benign', 'BrowserHijacking': 'Web',
                 'Backdoor_Malware': 'Web', 'XSS': 'Web', 'Uploading_Attack': 'Web', 'SqlInjection': 'Web',
                 'CommandInjection': 'Web', 'DictionaryBruteForce': 'BruteForce'}

dict_2classes = {'DDoS-RSTFINFlood': 'Attack', 'DDoS-PSHACK_Flood': 'Attack', 'DDoS-SYN_Flood': 'Attack',
                 'DDoS-UDP_Flood': 'Attack', 'DDoS-TCP_Flood': 'Attack', 'DDoS-ICMP_Flood': 'Attack',
                 'DDoS-SynonymousIP_Flood': 'Attack', 'DDoS-ACK_Fragmentation': 'Attack',
                 'DDoS-UDP_Fragmentation': 'Attack', 'DDoS-ICMP_Fragmentation': 'Attack', 'DDoS-SlowLoris': 'Attack',
                 'DDoS-HTTP_Flood': 'Attack', 'DoS-UDP_Flood': 'Attack', 'DoS-SYN_Flood': 'Attack',
                 'DoS-TCP_Flood': 'Attack', 'DoS-HTTP_Flood': 'Attack', 'Mirai-greeth_flood': 'Attack',
                 'Mirai-greip_flood': 'Attack', 'Mirai-udpplain': 'Attack', 'Recon-PingSweep': 'Attack',
                 'Recon-OSScan': 'Attack', 'Recon-PortScan': 'Attack', 'VulnerabilityScan': 'Attack',
                 'Recon-HostDiscovery': 'Attack', 'DNS_Spoofing': 'Attack', 'MITM-ArpSpoofing': 'Attack',
                 'BenignTraffic': 'Benign', 'BrowserHijacking': 'Attack', 'Backdoor_Malware': 'Attack', 'XSS': 'Attack',
                 'Uploading_Attack': 'Attack', 'SqlInjection': 'Attack', 'CommandInjection': 'Attack',
                 'DictionaryBruteForce': 'Attack'}

# Extracting data from csv to input into data frame
ciciot_train_data = pd.DataFrame()
for data_set in training_data_sets:
    print(f"data set {data_set} out of {len(training_data_sets)} \n")
    data_path = os.path.join(DATASET_DIRECTORY, data_set)
    df = pd.read_csv(data_path)
    ciciot_train_data = pd.concat([ciciot_train_data, df])  # dataframe to manipulate

ciciot_test_data = pd.DataFrame()
for test_set in test_data_sets:
    print(f"Testing set {test_set} out of {len(test_data_sets)} \n")
    data_path = os.path.join(DATASET_DIRECTORY, test_set)
    df = pd.read_csv(data_path)
    ciciot_test_data = pd.concat([df])

#########################################################
#    Process Dataset For CICIOT 2023                    #
#########################################################

## Remapping for other Classifications ##

if ciciot_label_class == "7+1":
    # Relabel the 'label' column using dict_7classes
    ciciot_train_data['label'] = ciciot_train_data['label'].map(dict_7classes)
    ciciot_test_data['label'] = ciciot_test_data['label'].map(dict_7classes)

if ciciot_label_class == "1+1":
    # Relabel the 'label' column using dict_2classes
    ciciot_train_data['label'] = ciciot_train_data['label'].map(dict_2classes)
    ciciot_test_data['label'] = ciciot_test_data['label'].map(dict_2classes)

# Drop the irrelevant features
# ciciot_train_data = ciciot_train_data.drop(columns=irrelevant_features)
# ciciot_test_data = ciciot_test_data.drop(columns=irrelevant_features)

# Shuffle data
ciciot_train_data = shuffle(ciciot_train_data, random_state=47)

# prints an instance of each class
print("Before Encoding and Scaling:")
unique_labels = ciciot_train_data['label'].unique()
for label in unique_labels:
    print(f"First instance of {label}:")
    print(ciciot_train_data[ciciot_train_data['label'] == label].iloc[0])


# ---                   Labeling                     --- #

# Assuming 'label' is the column name for the labels in the DataFrame `synth_data`
unique_labels = ciciot_train_data['label'].nunique()

# Print the number of unique labels
print(f"There are {unique_labels} unique labels in the dataset.")

# print the amount of instances for each label
class_counts = ciciot_train_data['label'].value_counts()
print(class_counts)

# Display the first few entries to verify the changes
print(ciciot_train_data.head())

# Encodes the training label
label_encoder = LabelEncoder()
ciciot_train_data['label'] = label_encoder.fit_transform(ciciot_train_data['label'])
ciciot_test_data['label'] = label_encoder.fit_transform(ciciot_test_data['label'])

# Store label mappings
label_mapping = {index: label for index, label in enumerate(label_encoder.classes_)}
print("Label mappings:", label_mapping)

# Retrieve the numeric codes for classes
class_codes = {label: label_encoder.transform([label])[0] for label in label_encoder.classes_}

# Print specific instances after label encoding
print("Real data After Encoding:")
for label, code in class_codes.items():
    # Print the first instance of each class
    print(f"First instance of {label} (code {code}):")
    print(ciciot_train_data[ciciot_train_data['label'] == code].iloc[0])
print(ciciot_train_data.head(), "\n")

# Feature / Label Split (X y split)
X_train_data = ciciot_train_data.drop(columns=['label'])
y_train_data = ciciot_train_data['label']

X_test_data = ciciot_test_data.drop(columns=['label'])
y_test_data = ciciot_test_data['label']

# Print the shapes of the resulting splits
print("X_train shape:", X_train_data.shape)
print("y_train shape:", y_train_data.shape)


# Step 2: Remove Correlated Features
correlation_matrix = X_train_data.corr().abs()
upper_triangle = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.70)]

X_reduced = X_train_data.drop(to_drop, axis=1)
print(f"Removed features: {to_drop}")

# Step 3: Apply RFE with Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
rfe = RFE(estimator=model, n_features_to_select=16, step=1)
rfe.fit(X_reduced, y_train_data)

selected_features = X_reduced.columns[rfe.support_]
print(f"Selected features: {selected_features}")

# Step 4: Evaluate the Selected Features
# Using 10-fold cross-validation to evaluate the performance
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Evaluate using the full feature set
full_feature_scores = cross_val_score(model, X_train_data, y_train_data, cv=cv, scoring='accuracy')
print(f"Full feature set accuracy: {full_feature_scores.mean()}")

# Evaluate using the reduced feature set
reduced_feature_scores = cross_val_score(model, X_reduced[selected_features], y_train_data, cv=cv, scoring='accuracy')
print(f"Reduced feature set accuracy: {reduced_feature_scores.mean()}")



def evaluate_model(model, X, y, cv):
    precision = cross_val_score(model, X, y, cv=cv, scoring='precision_macro').mean()
    recall = cross_val_score(model, X, y, cv=cv, scoring='recall_macro').mean()
    f1 = cross_val_score(model, X, y, cv=cv, scoring='f1_macro').mean()
    return precision, recall, f1

# Full feature set
precision_full, recall_full, f1_full = evaluate_model(model, X_train_data, y_train_data, cv)
print(f"Full feature set - Precision: {precision_full}, Recall: {recall_full}, F1 Score: {f1_full}")

# Reduced feature set
precision_reduced, recall_reduced, f1_reduced = evaluate_model(model, X_reduced[selected_features], y_train_data, cv)
print(f"Reduced feature set - Precision: {precision_reduced}, Recall: {recall_reduced}, F1 Score: {f1_reduced}")
