#########################################################
#    Import / Set up                                   #
#########################################################
import os
import random
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, mutual_info_classif
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.utils import shuffle

#########################################################
#   Step 1A: Load Data                                  #
#########################################################
DATASET_DIRECTORY = '../ciciot2023_archive/'

### Inputs ###
ciciot_sample_size = 2  # input: 2 at minimum
ciciot_label_class = "1+1"

# List the files in the dataset
csv_filepaths = [filename for filename in os.listdir(DATASET_DIRECTORY) if filename.endswith('.csv')]
print(csv_filepaths)

# Randomly select X files from the list if necessary
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

###############################################################
# Step 1B: Process Labels and Classes for Dataset CICIOT 2023 #
###############################################################

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

# --- Labeling --- #

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

#########################################################
# Step 2: Remove Correlated Features                    #
#########################################################

correlation_matrix = X_train_data.corr().abs()
upper_triangle = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.70)]

X_reduced = X_train_data.drop(to_drop, axis=1)
print(f"Removed correlated features: {to_drop}")

#########################################################
# Step 3A: Apply Mutual Information                      #
#########################################################

mi = mutual_info_classif(X_reduced, y_train_data, random_state=42)
mi_series = pd.Series(mi, index=X_reduced.columns)

top_features_mi = mi_series.sort_values(ascending=False).head(30).index
print(f"Top features by mutual information: {top_features_mi}")

#########################################################
# Step 1C: Scale the Features                            #
#########################################################

# Setting up Scaler for Features
scaler = MinMaxScaler(feature_range=(0, 1))

# Scale the numeric features present in X_reduced
scaled_num_cols = [col for col in num_cols if col in X_reduced.columns]
X_reduced[scaled_num_cols] = scaler.fit_transform(X_reduced[scaled_num_cols])

# prove if the data is loaded properly
print("Real data After Scaling (TRAIN):")
print(X_reduced.head())
print(X_reduced.shape)

#########################################################
# Step 3B: Tree-Based Feature Importance                 #
#########################################################

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_reduced, y_train_data)

importances = model.feature_importances_
feature_importances = pd.Series(importances, index=X_reduced.columns)

top_features_rf = feature_importances.sort_values(ascending=False).head(30).index
print(f"Top features by Random Forest importance: {top_features_rf}")

#########################################################
# Step 3C: Apply RFE with Random Forest                  #
#########################################################

rfe = RFE(estimator=model, n_features_to_select=16, step=1)
rfe.fit(X_reduced, y_train_data)

top_features_rfe = X_reduced.columns[rfe.support_]
print(f"Top features by RFE: {top_features_rfe}")

#########################################################
# Step 3D: Combine features from different methods               #
#########################################################

combined_features = list(set(top_features_mi) | set(top_features_rf) | set(top_features_rfe))
print(f"Combined top features: {combined_features}")

X_selected = X_reduced[combined_features]

#########################################################
# Step 4: Evaluate the Selected Features                #
#########################################################

# Define the model
final_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Using 10-fold cross-validation to evaluate the performance
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

def evaluate_model(model, X, y, cv):
    accuracy = cross_val_score(model, X, y, cv=cv, scoring='accuracy').mean()
    precision = cross_val_score(model, X, y, cv=cv, scoring='precision_macro').mean()
    recall = cross_val_score(model, X, y, cv=cv, scoring='recall_macro').mean()
    f1 = cross_val_score(model, X, y, cv=cv, scoring='f1_macro').mean()
    return accuracy, precision, recall, f1

# Full feature set
accuracy_full, precision_full, recall_full, f1_full = evaluate_model(final_model, X_train_data, y_train_data, cv)
print(f"Full feature set - Accuracy: {accuracy_full}, Precision: {precision_full}, Recall: {recall_full}, F1 Score: {f1_full}")

# Reduced feature set by RFE
accuracy_rfe, precision_rfe, recall_rfe, f1_rfe = evaluate_model(final_model, X_reduced[top_features_rfe], y_train_data, cv)
print(f"Reduced feature set RFE - Accuracy: {accuracy_rfe}, Precision: {precision_rfe}, Recall: {recall_rfe}, F1 Score: {f1_rfe}")

# Reduced feature set by Random Forest
accuracy_rf, precision_rf, recall_rf, f1_rf = evaluate_model(final_model, X_reduced[top_features_rf], y_train_data, cv)
print(f"Reduced feature set RF - Accuracy: {accuracy_rf}, Precision: {precision_rf}, Recall: {recall_rf}, F1 Score: {f1_rf}")

# Reduced feature set by Mutual Information
accuracy_mi, precision_mi, recall_mi, f1_mi = evaluate_model(final_model, X_reduced[top_features_mi], y_train_data, cv)
print(f"Reduced feature set MI - Accuracy: {accuracy_mi}, Precision: {precision_mi}, Recall: {recall_mi}, F1 Score: {f1_mi}")

# Combined feature set
accuracy_combined, precision_combined, recall_combined, f1_combined = evaluate_model(final_model, X_selected, y_train_data, cv)
print(f"Combined feature set - Accuracy: {accuracy_combined}, Precision: {precision_combined}, Recall: {recall_combined}, F1 Score: {f1_combined}")
