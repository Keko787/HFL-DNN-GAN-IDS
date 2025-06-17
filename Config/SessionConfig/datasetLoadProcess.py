from DatasetConfig.CICIOT2023_Sampling.ciciot2023DatasetLoadV2 import (loadCICIOT)
from DatasetConfig.IOTBotNet2020_Sampling.iotbotnet2020DatasetLoad import loadIOTBOTNET
from DatasetConfig.IoT_Handling.IotDatasetLoadProcess import load_and_preprocess_data, feature_selection, prepare_data, prepare_data_min_max
from DatasetConfig.LiveData_Handling.loadLiveData import loadLiveCaptureData
from DatasetConfig.Dataset_Preprocessing.datasetPreprocess import preprocess_dataset, preprocess_AC_dataset, preprocess_live_dataset


# --- Load Data ---#

def datasetLoadProcess(dataset_used, dataset_preprocessing):

    # --- 1 LOAD DATASET ---#
    # Initiate CICIOT
    ciciot_train_data = None
    ciciot_test_data = None
    irrelevant_features_ciciot = None

    # Initiate iotbonet
    all_attacks_train = None
    all_attacks_test = None
    relevant_features_iotbotnet = None

    # Initiate IoT
    X = None
    y = None

    # Initiate LiveData
    live_data = None
    irrelevant_features_live = None

    # load ciciot data if selected
    if dataset_used == "CICIOT":
        print("Loading CICIOT")
        # Load CICIOT data
        # ciciot_train_data, ciciot_test_data, irrelevant_features_ciciot = loadCICIOT(train_sample_size=130, test_sample_size=30,
        #        training_dataset_size=800000, testing_dataset_size=200000, attack_eval_samples_ratio=0.3)
        ciciot_train_data, ciciot_test_data, irrelevant_features_ciciot = loadCICIOT(train_sample_size=50,
                                                                                     test_sample_size=15,
                                                                                     training_dataset_size=400000,
                                                                                     testing_dataset_size=80000,
                                                                                     attack_eval_samples_ratio=1.0,
                                                                                     random_seed=110)
        # ciciot_train_data, ciciot_test_data, irrelevant_features_ciciot = loadCICIOT(train_sample_size=25,
        #                                                                              test_sample_size=10,
        #                                                                              training_dataset_size=100000,
        #                                                                              testing_dataset_size=20000,
        #                                                                              attack_eval_samples_ratio=0.3)

        # ciciot_train_data, ciciot_test_data, irrelevant_features_ciciot = loadCICIOT(train_sample_size=30,
        #                                                                              test_sample_size=15,
        #                                                                              training_dataset_size=200000,
        #                                                                              testing_dataset_size=40000,
        #                                                                              attack_eval_samples_ratio=0.3)

    # load iotbotnet data if selected
    elif dataset_used == "IOTBOTNET":
        # Load IOTbotnet data
        all_attacks_train, all_attacks_test, relevant_features_iotbotnet = loadIOTBOTNET()

    elif dataset_used == "IOT":
        '''Load IOT -- Bishwas's Dataset loading function'''
        # -- file path to preprocessed dataset
        file_path = 'datasets/50000_5000_IOT112andAllfields_Preprocessed.csv'
        # file_path = 'datasets/combined_edgeIIot_500k_custom_DDos.csv'
        X, y = load_and_preprocess_data(file_path)

    elif dataset_used == "LIVEDATA":
        # Load Live Packet Capture Data
        live_data, irrelevant_features_live = loadLiveCaptureData()

    # --- 2 PREPROCESS DATASET ---#
    if dataset_preprocessing == "Default":
        X_train_data, X_val_data, y_train_data, y_val_data, X_test_data, y_test_data = preprocess_dataset(
            dataset_used, ciciot_train_data, ciciot_test_data, all_attacks_train, all_attacks_test,
            irrelevant_features_ciciot, relevant_features_iotbotnet)
        return X_train_data, X_val_data, y_train_data, y_val_data, X_test_data, y_test_data

    elif dataset_preprocessing == "LIVEDATA":
        # Handle data passing through Initializing empty values
        print("Warning: ONLY X_VALIDATION AND X_TEST AVAILABLE FOR LIVEDATA DATASET")
        X_test_data = preprocess_live_dataset(live_data, irrelevant_features_live)
        return X_test_data

    elif dataset_preprocessing == "MM[-1,1]":
        X_train_data, X_val_data, y_train_data, y_val_data, X_test_data, y_test_data = preprocess_dataset(
            dataset_used, dataset_preprocessing, ciciot_train_data, ciciot_test_data, all_attacks_train, all_attacks_test,
            irrelevant_features_ciciot, relevant_features_iotbotnet)
        return X_train_data, X_val_data, y_train_data, y_val_data, X_test_data, y_test_data

    elif dataset_preprocessing == "AC-GAN":
        X_train_data, X_val_data, y_train_categorical, y_val_categorical, X_test_data, y_test_categorical = preprocess_AC_dataset(
            dataset_used, ciciot_train_data, ciciot_test_data, all_attacks_train, all_attacks_test,
            irrelevant_features_ciciot, relevant_features_iotbotnet)
        return X_train_data, X_val_data, y_train_categorical, y_val_categorical, X_test_data, y_test_categorical

    elif dataset_preprocessing == "IOT":
        selected_features = feature_selection(X, y)
        X_train, X_val, X_test, y_train, y_val, y_test, scaler = prepare_data(X, y, selected_features)

        return X_train, X_val, X_test, y_train, y_val, y_test

    elif dataset_preprocessing == "IOT-MinMax":
        selected_features = feature_selection(X, y)
        X_train, X_val, X_test, y_train, y_val, y_test, scaler = prepare_data_min_max(X, y, selected_features)

        return X_train, X_val, X_test, y_train, y_val, y_test
