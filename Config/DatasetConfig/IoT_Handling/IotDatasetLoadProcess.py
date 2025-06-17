# -Bishwas
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def load_and_preprocess_data(file_path, classes):
    """Load dataset and preprocess it by mapping attack types and selecting features."""
    df = pd.read_csv(file_path, low_memory=False)
    df.drop(columns=['Unnamed: 0'], inplace=True)
    # df = df.iloc[:, 2:].reset_index(drop=True)
    label = 'Attack_label' if classes == 'Binary' else 'Attack_type'

    # Mapping attack types to numerical labels
    attacks = {'Normal': 0, 'MITM': 1, 'Uploading': 2, 'Ransomware': 3, 'SQL_injection': 4,
               'DDoS_HTTP': 5, 'DDoS_TCP': 6, 'Password': 7, 'Port_Scanning': 8,
               'Vulnerability_scanner': 9, 'Backdoor': 10, 'XSS': 11, 'Fingerprinting': 12,
               'DDoS_UDP': 13, 'DDoS_ICMP': 14}
    df['Attack_type'] = df['Attack_type'].map(attacks)

    X = df.drop(columns=['Attack_label', 'Attack_type'])
    y = df[label]

    return X, y


def feature_selection(X, y):
    """Select best features using Chi-Squared test."""
    chi_selector = SelectKBest(chi2, k='all')
    chi_selector.fit_transform(X, y)

    chi_scores = pd.DataFrame({'feature': X.columns, 'score': chi_selector.scores_}).dropna()
    chi_scores = chi_scores.sort_values(by='score', ascending=False)
    selected_features = chi_scores['feature'].tolist()

    return selected_features


def prepare_data(X, y, selected_features):
    """Scale and split data into training, validation, and test sets."""
    scaler = StandardScaler().fit(X[selected_features])
    X_train, X_test, y_train, y_test = train_test_split(X[selected_features], y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.125, random_state=42)

    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    return X_train, X_val, X_test, y_train, y_val, y_test, scaler

def prepare_data_min_max(X, y, selected_features):
    """Scale and split data into training, validation, and test sets."""
    scaler = MinMaxScaler(feature_range=(0, 1)).fit(X[selected_features])
    X_train, X_test, y_train, y_test = train_test_split(X[selected_features], y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.125, random_state=42)

    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    return X_train, X_val, X_test, y_train, y_val, y_test, scaler