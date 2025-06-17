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
from datasetHandling.datasetLoadProcess import datasetLoadProcess


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
def predict_and_act(model, X_data, y_data=None, threshold=0.5):
    """
    Runs predictions on data, logs results, and takes necessary actions.
    Works with either labeled test data or unlabeled live data.

    Args:
        model: Trained NIDS model.
        X_data: Processed feature dataset.
        y_data: Optional ground truth labels for evaluation (None for live data).
        threshold: Probability threshold for classifying intrusions.
    """
    # Set up logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"intrusion_alerts_{timestamp}.log"
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Get predictions
    y_pred_probs = model.predict(X_data)

    # Convert probabilities to binary intrusion classification
    # Assuming 0 = Attack, 1 = Benign (adjust based on your model's output)
    y_pred = (y_pred_probs < threshold).astype("int32")

    # Get indices of detected intrusions
    intrusion_indices = np.where(y_pred == 0)[0]  # Adjust based on your model's definition of attack class

    # If ground truth labels exist, evaluate performance (test mode)
    if y_data is not None:
        # Test the model
        loss, accuracy, precision, recall, auc, logcosh = model.evaluate(X_data, y_data)
        f1 = f1_score(y_data, y_pred, zero_division=1)
        class_report = classification_report(y_data, y_pred, target_names=["Attack", "Benign"])

        print(f"\n📊 Model Performance:")
        print(f"🔹 Precision: {precision:.4f} | Recall: {recall:.4f} | F1-Score: {f1:.4f}")
        print(class_report)
        logging.info(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1-Score: {f1:.4f}")

        # Take action with ground truth for validation
        take_action_with_validation(intrusion_indices, y_data)
    else:
        # Live data mode - no ground truth available
        take_action_live(intrusion_indices, X_data)


def take_action_with_validation(intrusion_indices, y_test):
    """
    Executes system response when intrusions are detected and compares predictions to actual labels.
    Used for testing and validation purposes.

    Args:
        intrusion_indices: List of positional indices where the model predicted an intrusion.
        y_test: Actual labels from the test dataset.
    """
    y_test = np.array(y_test)  # Ensure indexing works correctly

    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for idx in intrusion_indices:
        if idx >= len(y_test):  # Prevent out-of-bounds errors
            print(f"⚠️ Skipping invalid index {idx} (out of range)")
            continue

        actual_label = y_test[idx]  # Get actual label from NumPy array

        if actual_label == 1:  # False Positive (Benign flagged as attack)
            print(f"⚠️ False Positive: Non-intrusion traffic at index {idx} was incorrectly flagged!")
            false_positives += 1
        else:  # True Positive (Correct detection of attack)
            print(f"✅ Correct Detection: Attack traffic at index {idx} was correctly flagged!")
            true_positives += 1

    # Calculate False Negatives (missed attacks)
    all_attack_indices = np.where(y_test == 0)[0]  # Find actual attack positions
    missed_attacks = set(all_attack_indices) - set(intrusion_indices)
    false_negatives = len(missed_attacks)

    # Log results
    logging.info(f"🚨 Intrusion Detection Report 🚨")
    logging.info(f"✔️ True Positives: {true_positives}")
    logging.info(f"⚠️ False Positives: {false_positives}")
    logging.info(f"❌ False Negatives (Missed Attacks): {false_negatives}")
    logging.info(f"Total Intrusions Detected: {len(intrusion_indices)}")

    # Print Summary
    print("\n📊 Intrusion Detection Summary:")
    print(f"✔️ True Positives (Correct Detections): {true_positives}")
    print(f"⚠️ False Positives (Wrongly Flagged Benign): {false_positives}")
    print(f"❌ False Negatives (Missed Attacks): {false_negatives}")

    if false_negatives > 0:
        print(f"❌ WARNING: {false_negatives} attacks were NOT detected!")

    print("🛑 System response executed for detected intrusions.\n")


def take_action_live(intrusion_indices, X_data):
    """
    Executes system response when intrusions are detected in live data.
    No ground truth labels are available, so actions are based solely on predictions.

    Args:
        intrusion_indices: List of positional indices where the model predicted an intrusion.
        X_data: The feature data that was analyzed (for reference purposes).
    """
    # Log the total number of detected intrusions
    if len(intrusion_indices) > 0:
        print(f"🚨 ALERT: {len(intrusion_indices)} potential intrusions detected in live traffic!")
        logging.warning(f"🚨 {len(intrusion_indices)} potential intrusions detected")

        # Process each detected intrusion
        for idx in intrusion_indices:
            # Extract relevant features for the intrusion (can be customized based on your feature set)
            # This is an example - adjust according to your specific X_data structure
            alert_id = f"ALERT-{datetime.now().strftime('%H%M%S')}-{idx}"

            # Log specific details about the intrusion
            print(f"🛑 Intrusion {alert_id}: Blocking suspicious traffic at index {idx}")
            logging.warning(f"Intrusion {alert_id}: Suspicious activity detected at index {idx}")

            # Here you would implement your actual response actions
            # Examples:
            # - Block IP address
            # - Terminate session
            # - Quarantine affected system
            # - Notify security team
            # simulate_block_traffic(X_data[idx])  # Placeholder for actual implementation

        print(f"⚠️ Security measures have been activated for {len(intrusion_indices)} suspicious traffic patterns")
        logging.info(f"Security measures activated for {len(intrusion_indices)} potential intrusions")
    else:
        print("✅ No intrusions detected in current data stream.")
        logging.info("Traffic scan completed: No intrusions detected")


# Example of a function that would implement an actual security response
def simulate_block_traffic(traffic_data):
    """
    Simulates blocking malicious traffic.
    In a real system, this would interface with network security controls.

    Args:
        traffic_data: The specific traffic data to block
    """
    # This is a placeholder - replace with actual implementation
    # Example: Extract source IP from traffic_data and add to firewall block list
    pass


# Main script execution
if __name__ == "__main__":
    print("\n============================")
    print("🔍 Intrusion Detection System")
    print("============================\n")

    # Argument Parsing
    parser = argparse.ArgumentParser(description='Detect intrusions using a trained NIDS model.')
    parser.add_argument('--pretrained_model', type=str, required=True, help="Path to the trained model.")
    parser.add_argument('--threshold', type=float, default=0.5, help="Threshold for classifying intrusions.")
    parser.add_argument('--mode', type=str, choices=['test', 'live'], default='live',
                        help="'test' mode uses labeled data for validation, 'live' mode for unlabeled data")

    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(
        filename=f"intrusion_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Load Model
    model = load_model(args.pretrained_model)

    if args.mode == 'test':
        # Test mode with labeled data
        dataset_used = "CICIOT"  # or whatever dataset you want to use for testing
        dataset_processing = "Default"
        print(f"📡 Loading Test Dataset for validation...")
        X_train_data, X_val_data, y_train_data, y_val_data, X_test_data, y_test_data = datasetLoadProcess(
            dataset_used, dataset_processing)

        # Predict and Take Action with validation
        predict_and_act(model, X_test_data, y_test_data, threshold=args.threshold)
    else:
        # Live mode with unlabeled data
        dataset_used = "LIVEDATA"  # Adjust based on your actual live data source
        dataset_processing = "LIVEDATA"
        print(f"📡 Loading Live Data Stream...")

        # This would need to be modified based on how you actually get live data
        # For now, we're assuming the datasetLoadProcess function can handle live data
        X_test_data = datasetLoadProcess(dataset_used, dataset_processing)

        # Predict and Take Action without ground truth
        predict_and_act(model, X_test_data, y_data=None, threshold=args.threshold)
