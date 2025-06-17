import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix


class NidsIoTBinaryCentralConfig:
    def __init__(self, model, nids, x_train, x_val, y_train, y_val, x_test, y_test, BATCH_SIZE,
                 noise_dim, epochs, steps_per_epoch, learning_rate):

        self.nids = nids

    def plot_metrics(self, history, state):
        """Plot training and validation accuracy and loss."""
        plt.figure()
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.savefig(f'../results/centralized/binary/{state}/accuracy_plot.jpg')
        plt.close()

        plt.figure()
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.savefig(f'../results/centralized/binary/{state}/loss_plot.jpg')
        plt.close()

    def save(self):
        self.nids.save('cnn_lstm_gru_model_binary_working.h5')

    def train_and_evaluate(self, model, X_train, y_train, X_val, y_val, X_test, y_test):
        """Train the model and evaluate its performance."""
        start_time = time.time()
        history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=6, batch_size=32)
        train_time = time.time() - start_time

        start_time = time.time()
        loss, accuracy = model.evaluate(X_test, y_test, batch_size=32)
        test_time = time.time() - start_time

        print(f'Test Loss: {loss:.4f}')
        print(f'Test Accuracy: {accuracy:.4f}')
        print(f'Training time: {train_time:.2f} seconds')
        print(f'Testing time: {test_time:.2f} seconds')

        return history, model

    def evaluate_model(self, model, X_test, y_test, state):
        """Evaluate the model and generate a confusion matrix."""
        y_pred = model.predict(np.expand_dims(X_test, axis=2))
        if state == 'test':
            y_pred_binary = (y_pred > 0.5).astype(int)
            print(classification_report(y_test, y_pred_binary, target_names=['No Intrusion', 'Intrusion']))

            conf_mat = confusion_matrix(y_test, y_pred_binary)
        else:
            print(classification_report(y_test, np.round(y_pred), target_names=['No Intrusion', 'Intrusion']))
            conf_mat = confusion_matrix(y_test, np.round(y_pred))
        sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', xticklabels=['No Intrusion', 'Intrusion'],
                    yticklabels=['No Intrusion', 'Intrusion'])
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')
        plt.savefig(f'../results/centralized/binary/{state}/confusion_matrix.jpg')
        plt.close()

        cm_norm = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]

        # Step 5: Plot the normalized confusion matrix as percentages
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm_norm, annot=True, cmap='Blues', xticklabels=['No Intrusion', 'Intrusion'],
                    yticklabels=['No Intrusion', 'Intrusion'], fmt='.2%')

        # Set the plot labels and title for the normalized confusion matrix
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title('Normalized Confusion Matrix (Percentages)')
        # Optional: Save the normalized confusion matrix plot
        plt.savefig(f'../results/centralized/binary/{state}/normalized_confusion_matrix.jpg')
        plt.close()

    def fit(self, model, X_train, y_train, X_val, y_val, X_test, y_test):
        history, model = self.train_and_evaluate(model, X_train, y_train, X_val, y_val, X_test, y_test)
        self.plot_metrics(history, 'train')
        self.evaluate_model(model, X_test, y_test, 'train')
