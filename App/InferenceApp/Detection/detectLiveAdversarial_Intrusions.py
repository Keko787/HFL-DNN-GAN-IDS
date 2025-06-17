import sys
import os
import argparse
import numpy as np
import tensorflow as tf
import logging
from datetime import datetime
from collections import Counter
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

sys.path.append(os.path.abspath('..'))

# Import preprocessing and dataset functions
from datasetHandling.datasetLoadProcess import datasetLoadProcess


# Load the trained models (NIDS and discriminator)
def load_models(nids_model_path, discriminator_model_path=None):
    """
    Loads the pre-trained NIDS model and optionally a discriminator model.

    Args:
        nids_model_path: Path to the traditional NIDS model
        discriminator_model_path: Optional path to the discriminator model

    Returns:
        Dictionary of loaded models
    """
    models = {}

    try:
        # Load NIDS model
        if nids_model_path:
            print(f"📥 Loading NIDS model from {nids_model_path}...")
            with tf.keras.utils.custom_object_scope({'LogCosh': tf.keras.losses.LogCosh}):
                models['nids'] = tf.keras.models.load_model(nids_model_path)
            print("✅ NIDS model successfully loaded!")

        # Load discriminator model (if provided)
        if discriminator_model_path:
            print(f"📥 Loading discriminator model from {discriminator_model_path}...")
            models['discriminator'] = tf.keras.models.load_model(discriminator_model_path)
            print("✅ Discriminator model successfully loaded!")

            # Check if it's an AC-GAN discriminator (has dual outputs)
            if len(models['discriminator'].output_names) > 1:
                print("🔍 AC-GAN discriminator detected with multiple outputs!")
                models['discriminator_type'] = 'acgan'
            else:
                models['discriminator_type'] = 'standard'

        return models

    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        sys.exit(1)


# Probabilistic fusion for AC-GAN discriminator predictions
def probabilistic_fusion(validity_scores, class_predictions):
    """
    Apply probabilistic fusion to combine validity and class predictions.

    Args:
        validity_scores: Array of validity scores from discriminator
        class_predictions: Array of class probabilities from discriminator

    Returns:
        List of classification results with probabilities
    """
    total_samples = len(validity_scores)
    results = []

    for i in range(total_samples):
        # Validity probabilities: P(valid) and P(invalid)
        p_valid = validity_scores[i][0]  # Probability of being valid/real
        p_invalid = 1 - p_valid  # Probability of being invalid/fake

        # Class probabilities: 2 classes (benign=0, attack=1)
        p_benign = class_predictions[i][0]  # Probability of being benign
        p_attack = class_predictions[i][1]  # Probability of being attack

        # Calculate joint probabilities for all combinations
        p_valid_benign = p_valid * p_benign
        p_valid_attack = p_valid * p_attack
        p_invalid_benign = p_invalid * p_benign
        p_invalid_attack = p_invalid * p_attack

        # Store all probabilities in a dictionary
        probabilities = {
            "valid_benign": p_valid_benign,
            "valid_attack": p_valid_attack,
            "invalid_benign": p_invalid_benign,
            "invalid_attack": p_invalid_attack
        }

        # Find the most likely outcome
        most_likely = max(probabilities, key=probabilities.get)

        # For analysis, add the actual probabilities alongside the classification
        result = {
            "classification": most_likely,
            "probabilities": probabilities
        }

        results.append(result)

    return results


# Predict and take action using all available models
def predict_and_act(models, X_data, y_data=None, threshold=0.5, ensemble_method='voting'):
    """
    Runs predictions using all available models, logs results, and takes necessary actions.

    Args:
        models: Dictionary of loaded models (NIDS, discriminator)
        X_data: Processed feature dataset
        y_data: Optional ground truth labels for evaluation (None for live data)
        threshold: Probability threshold for classifying intrusions
        ensemble_method: How to combine predictions ('voting', 'weighted', or 'fusion')
    """
    # Set up logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"intrusion_alerts_{timestamp}.log"
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Dictionary to store predictions from each model
    all_predictions = {}

    # === NIDS Model Predictions ===
    if 'nids' in models:
        nids_model = models['nids']

        # Get predictions
        nids_pred_probs = nids_model.predict(X_data)

        # Convert probabilities to binary intrusion classification
        # Assuming 0 = Attack, 1 = Benign
        nids_pred = (nids_pred_probs < threshold).astype("int32")
        all_predictions['nids'] = nids_pred

        # Log NIDS confidence scores
        logging.info(f"NIDS prediction confidence breakdown:")
        high_confidence = np.sum((nids_pred_probs < 0.2) | (nids_pred_probs > 0.8))
        logging.info(f"  High confidence predictions: {high_confidence / len(nids_pred_probs) * 100:.2f}%")

    # === Discriminator Model Predictions ===
    disc_fusion_results = None
    if 'discriminator' in models:
        disc_model = models['discriminator']
        disc_type = models.get('discriminator_type', 'standard')

        if disc_type == 'acgan':
            # For AC-GAN discriminator with dual outputs
            validity_scores, class_predictions = disc_model.predict(X_data)

            # Apply probabilistic fusion
            disc_fusion_results = probabilistic_fusion(validity_scores, class_predictions)

            # Extract classifications
            disc_classifications = [result["classification"] for result in disc_fusion_results]

            # Mark as attack if classification contains "attack" or "invalid"
            disc_pred = np.array([
                0 if ("attack" in result["classification"] or "invalid" in result["classification"]) else 1
                for result in disc_fusion_results
            ])

            all_predictions['discriminator'] = disc_pred

            # Log discriminator classification distribution
            class_distribution = Counter(disc_classifications)
            logging.info(f"Discriminator classification distribution: {dict(class_distribution)}")

        else:
            # For standard discriminator with single output
            disc_pred_probs = disc_model.predict(X_data)
            disc_pred = (disc_pred_probs > 0.5).astype("int32")  # 1 if fake/generated (attack), 0 if real (benign)
            all_predictions['discriminator'] = disc_pred

    # === Ensemble Predictions ===
    if len(all_predictions) > 1:
        if ensemble_method == 'voting':
            # Simple majority voting
            ensemble_pred = np.zeros(len(X_data), dtype=int)
            for model_name, preds in all_predictions.items():
                ensemble_pred += preds
            # If more models predict benign (1) than attack (0), classify as benign
            ensemble_pred = (ensemble_pred >= len(all_predictions) / 2).astype(int)

        elif ensemble_method == 'weighted':
            # Weighted voting (NIDS has higher weight than discriminator)
            weights = {'nids': 0.7, 'discriminator': 0.3}
            ensemble_pred = np.zeros(len(X_data))
            for model_name, preds in all_predictions.items():
                ensemble_pred += weights.get(model_name, 0.5) * preds
            # Threshold at 0.5 of weighted sum
            ensemble_pred = (ensemble_pred >= 0.5).astype(int)

        elif ensemble_method == 'fusion' and disc_fusion_results:
            # Advanced fusion using discriminator probabilities and NIDS predictions
            ensemble_pred = np.zeros(len(X_data), dtype=int)
            for i in range(len(X_data)):
                nids_vote = all_predictions['nids'][i] if 'nids' in all_predictions else 1

                # Extract probabilities from fusion results
                fusion_result = disc_fusion_results[i]

                # Determine if there's a strong attack signal
                attack_signal = (
                        fusion_result['probabilities']['valid_attack'] > 0.4 or
                        fusion_result['probabilities']['invalid_attack'] > 0.3 or
                        fusion_result['probabilities']['invalid_benign'] > 0.5
                )

                # If either model detects an attack with strong confidence, mark as attack
                if nids_vote == 0 or attack_signal:
                    ensemble_pred[i] = 0  # Attack
                else:
                    ensemble_pred[i] = 1  # Benign
        else:
            # Default to NIDS if ensemble method isn't properly specified or applicable
            ensemble_pred = all_predictions.get('nids', np.ones(len(X_data), dtype=int))
    else:
        # Only one model available
        ensemble_pred = next(iter(all_predictions.values()))

    # Get indices of detected intrusions
    intrusion_indices = np.where(ensemble_pred == 0)[0]  # 0 = Attack

    # === Model Evaluations ===
    # For test mode with ground truth labels
    if y_data is not None:
        # Basic evaluation for each model
        for model_name, preds in all_predictions.items():
            acc = np.mean(preds == y_data)
            precision = precision_score(y_data, preds, zero_division=1)
            recall = recall_score(y_data, preds, zero_division=1)
            f1 = f1_score(y_data, preds, zero_division=1)

            print(f"\n📊 {model_name.upper()} Model Performance:")
            print(f"🔹 Accuracy: {acc:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1-Score: {f1:.4f}")
            logging.info(
                f"{model_name.upper()} Metrics - Accuracy: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

        # Ensemble model evaluation
        ens_acc = np.mean(ensemble_pred == y_data)
        ens_precision = precision_score(y_data, ensemble_pred, zero_division=1)
        ens_recall = recall_score(y_data, ensemble_pred, zero_division=1)
        ens_f1 = f1_score(y_data, ensemble_pred, zero_division=1)
        ens_report = classification_report(y_data, ensemble_pred, target_names=["Attack", "Benign"])

        print(f"\n📊 ENSEMBLE Model Performance:")
        print(
            f"🔹 Accuracy: {ens_acc:.4f} | Precision: {ens_precision:.4f} | Recall: {ens_recall:.4f} | F1-Score: {ens_f1:.4f}")
        print(ens_report)
        logging.info(
            f"ENSEMBLE Metrics - Accuracy: {ens_acc:.4f}, Precision: {ens_precision:.4f}, Recall: {ens_recall:.4f}, F1: {ens_f1:.4f}")

    # === Take Action ===
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
        intrusion_indices: List of positional indices where the model predicted an intrusion
        y_test: Actual labels from the test dataset
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
        intrusion_indices: List of positional indices where the model predicted an intrusion
        X_data: The feature data that was analyzed (for reference purposes)
    """
    # Log the total number of detected intrusions
    if len(intrusion_indices) > 0:
        print(f"🚨 ALERT: {len(intrusion_indices)} potential intrusions detected in live traffic!")
        logging.warning(f"🚨 {len(intrusion_indices)} potential intrusions detected")

        # Process each detected intrusion
        for idx in intrusion_indices:
            # Generate an alert ID with timestamp
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
    print("🔍 Enhanced Intrusion Detection System")
    print("============================\n")

    # Argument Parsing
    parser = argparse.ArgumentParser(description='Detect intrusions using multiple detection models.')
    parser.add_argument('--nids_model', type=str, help="Path to the traditional NIDS model.")
    parser.add_argument('--discriminator_model', type=str, help="Path to the discriminator model (optional).")
    parser.add_argument('--threshold', type=float, default=0.5, help="Threshold for classifying intrusions.")
    parser.add_argument('--mode', type=str, choices=['test', 'live'], default='live',
                        help="'test' mode uses labeled data for validation, 'live' mode for unlabeled data")
    parser.add_argument('--ensemble', type=str, choices=['voting', 'weighted', 'fusion'], default='fusion',
                        help="Method to combine multiple model predictions")

    args = parser.parse_args()

    # Ensure at least one model is provided
    if not args.nids_model and not args.discriminator_model:
        print("❌ Error: At least one model (NIDS or discriminator) must be provided!")
        parser.print_help()
        sys.exit(1)

    # Set up logging
    logging.basicConfig(
        filename=f"intrusion_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Load Models
    models = load_models(args.nids_model, args.discriminator_model)

    if args.mode == 'test':
        # Test mode with labeled data
        dataset_used = "CICIOT"  # or whatever dataset you want to use for testing
        dataset_processing = "Default"
        print(f"📡 Loading Test Dataset for validation...")
        X_train_data, X_val_data, y_train_data, y_val_data, X_test_data, y_test_data = datasetLoadProcess(
            dataset_used, dataset_processing)

        # Predict and Take Action with validation
        predict_and_act(models, X_test_data, y_test_data, threshold=args.threshold, ensemble_method=args.ensemble)
    else:
        # Live mode with unlabeled data
        dataset_used = "LIVEDATA"  # Adjust based on your actual live data source
        dataset_processing = "LIVEDATA"
        print(f"📡 Loading Live Data Stream...")

        # This would need to be modified based on how you actually get live data
        X_test_data = datasetLoadProcess(dataset_used, dataset_processing)

        # Predict and Take Action without ground truth
        predict_and_act(models, X_test_data, y_data=None, threshold=args.threshold, ensemble_method=args.ensemble)
