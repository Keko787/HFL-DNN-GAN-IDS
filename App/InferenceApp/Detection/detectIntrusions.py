import sys
import os
import argparse
import numpy as np
import tensorflow as tf
import logging
from datetime import datetime
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
sys.path.append(os.path.abspath('..'))

# Import preprocessing and dataset functions
from DatasetConfig.CICIOT2023_Sampling.ciciot2023DatasetLoadV2 import loadCICIOT
from DatasetConfig.IOTBotNet2020_Sampling.iotbotnet2020DatasetLoad import loadIOTBOTNET
from DatasetConfig.Dataset_Preprocessing.datasetPreprocess import preprocess_dataset

# Load the trained model
def load_model(pretrained_model):
    """ Loads the pre-trained NIDS model. """
    try:
        if pretrained_model:
            print(f"📥 Loading pretrained model from {pretrained_model}...")
            with tf.keras.utils.custom_object_scope({'LogCosh': tf.keras.losses.LogCosh}):
                model = tf.keras.models.load_model(pretrained_model)
            print("✅ Model successfully loaded!")
            return model
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        sys.exit(1)

# Predict and take action
def predict_and_act(model, X_test, y_test=None, threshold=0.5):
    """
    Runs predictions on test data, logs results, and takes necessary actions.

    Args:
        model: Trained NIDS model.
        X_test: Processed feature dataset for testing.
        y_test: Optional ground truth labels for evaluation.
        threshold: Probability threshold for classifying intrusions.
    """

    # Get predictions
    y_pred_probs = model.predict(X_test)

    # Convert probabilities to binary intrusion classification
    y_pred = (y_pred_probs < threshold).astype("int32")

    # Get indices of detected intrusions
    intrusion_indices = np.where(y_pred == 1)[0]

    # If ground truth labels exist, evaluate performance
    if y_test is not None:
        # Test the model
        loss, accuracy, precision, recall, auc, logcosh = model.evaluate(X_test, y_test)
        f1 = f1_score(y_test, y_pred, zero_division=1)

        # Compute class-wise precision & recall
        class_report = classification_report(y_test, y_pred, target_names=["Benign", "Attack"])

        print(f"\n📊 Model Performance:")
        print(f"🔹 Precision: {precision:.4f} | Recall: {recall:.4f} | F1-Score: {f1:.4f}")
        print(class_report)
        logging.info(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1-Score: {f1:.4f}")

    # Log intrusions
    if len(intrusion_indices) > 0:
        print(f"🚨 {len(intrusion_indices)} Intrusions Detected!")
        logging.warning(f"🚨 {len(intrusion_indices)} Intrusions Detected!")
        # Pass y_pred to take_action for True Negative calculation
        metrics = take_action(intrusion_indices, y_test, y_pred)
        return metrics
    else:
        print("✅ No intrusions detected.")
        # Still calculate metrics even when no intrusions detected
        if y_test is not None:
            metrics = take_action(intrusion_indices, y_test, y_pred)
            return metrics

# Take action on detected intrusions
# def take_action(intrusion_indices, y_test):
#     """
#     Executes system response when intrusions are detected.
#
#     Args:
#         intrusion_indices: List of intrusion instances detected.
#     """
#     for idx in intrusion_indices:
#         print(f"🛑 Blocking network traffic from index {idx}...")
#     logging.info(f"⚠️ Secured the system from {len(intrusion_indices)} intrusions.")


def take_action(intrusion_indices, y_test, y_pred):
    """
    Executes system response when intrusions are detected and compares predictions to actual labels.
    Now includes True Negatives calculation.

    Args:
        intrusion_indices: List of positional indices where the model predicted an intrusion.
        y_test: Actual labels from the test dataset (0=Attack, 1=Benign).
        y_pred: Model predictions (0=Benign, 1=Attack/Intrusion).

    Returns:
        None
    """
    y_test = np.array(y_test)  # Ensure indexing works correctly
    y_pred = np.array(y_pred)  # Ensure indexing works correctly

    true_positives = 0
    false_positives = 0
    false_negatives = 0
    true_negatives = 0

    # Check predicted intrusions
    for idx in intrusion_indices:
        if idx >= len(y_test):  # Prevent out-of-bounds errors
            print(f"⚠️ Skipping invalid index {idx} (out of range)")
            continue

        actual_label = y_test[idx]  # Get actual label from NumPy array

        if actual_label == 0:  # True Positive (Attack correctly flagged as attack)
            print(f"✅ True Positive: Attack traffic at index {idx} was correctly flagged!")
            true_positives += 1
        else:  # False Positive (Benign incorrectly flagged as attack)
            print(f"⚠️ False Positive: Benign traffic at index {idx} was incorrectly flagged!")
            false_positives += 1

    # Calculate False Negatives (missed attacks)
    all_attack_indices = np.where(y_test == 1)[0]  # Find actual attack positions
    missed_attacks = set(all_attack_indices) - set(intrusion_indices)
    false_negatives = len(missed_attacks)

    # Calculate True Negatives (benign correctly identified as benign)
    all_benign_indices = np.where(y_test == 0)[0]  # Find actual benign positions
    correctly_identified_benign = set(all_benign_indices) - set(intrusion_indices)
    true_negatives = len(correctly_identified_benign)

    # Alternative calculation using predictions array:
    # true_negatives = np.sum((y_pred == 0) & (y_test == 1))

    # Log results
    logging.info(f"🚨 Intrusion Detection Report 🚨")
    logging.info(f"✔️ True Positives: {true_positives}")
    logging.info(f"✔️ True Negatives: {true_negatives}")
    logging.info(f"⚠️ False Positives: {false_positives}")
    logging.info(f"❌ False Negatives (Missed Attacks): {false_negatives}")
    logging.info(f"Total Intrusions Detected: {len(intrusion_indices)}")

    # Print Summary
    print("\n📊 Intrusion Detection Summary:")
    print(f"✔️ True Positives (Attacks Correctly Detected): {true_positives}")
    print(f"✔️ True Negatives (Benign Correctly Identified): {true_negatives}")
    print(f"⚠️ False Positives (Benign Wrongly Flagged): {false_positives}")
    print(f"❌ False Negatives (Attacks Missed): {false_negatives}")

    # Calculate additional metrics
    total_samples = len(y_test)
    accuracy = (true_positives + true_negatives) / total_samples
    specificity = true_negatives / (true_negatives + false_positives) if (true_negatives + false_positives) > 0 else 0

    print(f"\n📈 Additional Metrics:")
    print(f"🎯 Accuracy: {accuracy:.4f}")
    print(f"🔍 Specificity (True Negative Rate): {specificity:.4f}")

    if false_negatives > 0:
        print(f"\n❌ WARNING: {false_negatives} attacks were NOT detected!")

    print("🛑 System response executed for detected intrusions.\n")

    return {
        'true_positives': true_positives,
        'true_negatives': true_negatives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'accuracy': accuracy,
        'specificity': specificity
    }


# Main script execution
if __name__ == "__main__":
    print("\n============================")
    print("🔍 Intrusion Detection System")
    print("============================\n")

    # Argument Parsing
    parser = argparse.ArgumentParser(description='Detect intrusions using a trained NIDS model.')
    parser.add_argument('--dataset', type=str, choices=["CICIOT", "IOTBOTNET", "Live"], default="CICIOT",
                        help='Dataset to use: CICIOT or IOTBOTNET')
    parser.add_argument('--pretrained_model', type=str, required=True, help="Path to the trained model.")
    parser.add_argument('--threshold', type=float, default=0.5, help="Threshold for classifying intrusions.")

    args = parser.parse_args()

    # Load Dataset
    print(f"📡 Loading {args.dataset} dataset...")
    if args.dataset == "CICIOT":
        train_data, test_data, irrelevant_features = loadCICIOT()
    if args.dataset == "IOTBOTNET":
        train_data, test_data, irrelevant_features = loadIOTBOTNET()
    # else: # for live data packet captures
    #      train_data, test_data, irrelevant_features = liveData()

    # Preprocess Dataset
    print("🔄 Preprocessing dataset...")
    X_train, X_val, y_train, y_val, X_test, y_test = preprocess_dataset(
        args.dataset, train_data, test_data, None, None, irrelevant_features, None
    )

    # Load Model
    model = load_model(args.pretrained_model)

    # Predict and Take Action
    predict_and_act(model, X_test, y_test, threshold=args.threshold)
