import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, LeakyReLU, Input, Add, BatchNormalization, LSTM, Conv1D, MaxPooling1D, GRU, Reshape, SeparableConv1D, ReLU
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model

# Hyperparameters
l2_alpha = 0.01  # L2 regularization factor
input_dim = 22  # Number of features
time_steps = 50  # Length of time-series window
dropout_rate = 0.4


# ---                 Default / CICIOT Models                   --- #

def create_high_performance_nids(input_dim=21):
    """
    High-performance CNN-RNN hybrid for optimal detection accuracy.
    Uses CNN feature extraction followed by GRU→LSTM for temporal analysis.

    Estimated size: ~200-250K parameters
    """
    # Input layer for tabular data
    inputs = Input(shape=(input_dim,))

    # Reshape for CNN processing
    x = Reshape((input_dim, 1))(inputs)

    # CNN feature extraction
    x = Conv1D(64, kernel_size=3, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling1D(pool_size=2)(x)

    x = Conv1D(128, kernel_size=3, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)

    # GRU → LSTM sequence (as recommended)
    x = GRU(64, return_sequences=True)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    x = LSTM(48, return_sequences=False)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    # Classification head
    x = Dense(32, kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dropout(0.3)(x)

    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(inputs, outputs)

    # model.compile(
    #     optimizer='adam',
    #     loss='binary_crossentropy',
    #     metrics=['accuracy', 'AUC', 'Precision', 'Recall']
    # )

    return model


def create_balanced_nids(input_dim=21):
    """
    Balanced model with efficient CNN feature extraction and GRU for temporal analysis.
    Good trade-off between detection capability and resource usage.

    Estimated size: ~50-80K parameters
    """

    # Input and reshape
    inputs = Input(shape=(input_dim,))
    x = Reshape((input_dim, 1))(inputs)

    # Efficient CNN with separable convolution
    x = SeparableConv1D(32, kernel_size=3, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling1D(pool_size=2)(x)

    # Single efficient GRU layer
    x = GRU(40, activation='tanh', recurrent_activation='sigmoid')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    # Simple classification head
    x = Dense(24)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Dropout(0.2)(x)

    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(inputs, outputs)
    # model.compile(
    #     optimizer='adam',
    #     loss='binary_crossentropy',
    #     metrics=['accuracy', 'AUC']
    # )

    return model


def create_lightweight_nids(input_dim=21):
    """
    Extremely lightweight model for resource-constrained devices.
    Uses a single GRU layer with minimal dense processing.

    Estimated size: ~8-12K parameters
    """
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, GRU, BatchNormalization

    model = Sequential([
        # Direct GRU processing of tabular data
        GRU(24, input_shape=(input_dim, 1), activation='tanh',
            recurrent_activation='sigmoid', return_sequences=False),
        BatchNormalization(),

        # Minimal classification layer
        Dense(16, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])

    # Use a more efficient optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model


def create_optimized_NIDS_model(input_dim, l2_alpha=0.0001):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),

        # First block - wider with less regularization
        Dense(128, activation='leaky_relu', kernel_regularizer=l2(l2_alpha * 0.5)),
        BatchNormalization(),
        Dropout(0.2),

        # Second block
        Dense(64, activation='leaky_relu', kernel_regularizer=l2(l2_alpha * 0.75)),
        Dropout(0.3),

        # Third block
        Dense(32, activation='leaky_relu', kernel_regularizer=l2(l2_alpha)),
        BatchNormalization(),
        Dropout(0.3),

        # Fourth block
        Dense(16, activation='leaky_relu', kernel_regularizer=l2(l2_alpha)),
        Dropout(0.4),

        # Output
        Dense(1, activation='sigmoid')
    ])
    return model


def create_optimized_model(input_dim, l2_alpha=0.0001, dropout_rate=0.2):
    """
    Optimized deep learning model:
    - Uses LeakyReLU instead of ReLU.
    - Reduces dropout rate for better feature retention.
    - Adds residual (skip) connections for stability.
    - Keeps L2 regularization for weight decay.
    """

    inputs = Input(shape=(input_dim,))

    x = Dense(128, kernel_regularizer=l2(l2_alpha))(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dropout(dropout_rate)(x)

    residual = Dense(64, kernel_regularizer=l2(l2_alpha))(x)  # Residual path
    residual = BatchNormalization()(residual)
    residual = LeakyReLU(alpha=0.1)(residual)

    x = Dense(64, kernel_regularizer=l2(l2_alpha))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dropout(dropout_rate)(x)

    x = Add()([x, residual])  # Adding residual connection

    x = Dense(32, kernel_regularizer=l2(l2_alpha))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dropout(dropout_rate)(x)

    x = Dense(16, kernel_regularizer=l2(l2_alpha))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dropout(dropout_rate)(x)

    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(inputs, outputs)

    return model


def create_CICIOT_Model(input_dim, regularizationEnabled, DP_enabled, l2_alpha):

    # --- Model Definition --- #
    if regularizationEnabled:
        # with regularization
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(input_dim,)),
            Dense(64, activation='relu', kernel_regularizer=l2(l2_alpha)),
            BatchNormalization(),
            Dropout(0.4),  # Dropout layer with 50% dropout rate

            Dense(32, activation='relu', kernel_regularizer=l2(l2_alpha)),
            BatchNormalization(),
            Dropout(0.4),

            Dense(16, activation='relu', kernel_regularizer=l2(l2_alpha)),
            BatchNormalization(),
            Dropout(0.4),

            Dense(8, activation='relu', kernel_regularizer=l2(l2_alpha)),
            BatchNormalization(),
            Dropout(0.4),

            Dense(4, activation='relu', kernel_regularizer=l2(l2_alpha)),
            BatchNormalization(),
            Dropout(0.4),

            Dense(1, activation='sigmoid')
        ])

    elif regularizationEnabled and DP_enabled:
        # with regularization
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(input_dim,)),
            Dense(32, activation='relu', kernel_regularizer=l2(l2_alpha)),
            BatchNormalization(),
            Dropout(0.4),
            Dense(16, activation='relu', kernel_regularizer=l2(l2_alpha)),
            BatchNormalization(),
            Dropout(0.4),
            Dense(8, activation='relu', kernel_regularizer=l2(l2_alpha)),
            BatchNormalization(),
            Dropout(0.4),
            Dense(4, activation='relu', kernel_regularizer=l2(l2_alpha)),
            BatchNormalization(),
            Dropout(0.4),
            Dense(1, activation='sigmoid')
        ])

    elif DP_enabled:
        # with regularization
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(input_dim,)),
            Dense(32, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(16, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(8, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(4, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(1, activation='sigmoid')
        ])

    else:
        # without regularization
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(input_dim,)),
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),  # Dropout layer with 50% dropout rate
            Dense(32, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(16, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(8, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(4, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(1, activation='sigmoid')
        ])

    return model


# ---                   IOTBOTNET Models                  --- #

def create_IOTBOTNET_Model(input_dim, regularizationEnabled, l2_alpha):

    # --- Model Definition --- #
    if regularizationEnabled:
        # with regularization
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(input_dim,)),
            Dense(16, activation='relu', kernel_regularizer=l2(l2_alpha)),
            BatchNormalization(),
            Dropout(0.3),  # Dropout layer with 30% dropout rate
            Dense(8, activation='relu', kernel_regularizer=l2(l2_alpha)),
            BatchNormalization(),
            Dropout(0.3),
            Dense(4, activation='relu', kernel_regularizer=l2(l2_alpha)),
            BatchNormalization(),
            Dropout(0.3),
            Dense(2, activation='relu', kernel_regularizer=l2(l2_alpha)),
            BatchNormalization(),
            Dropout(0.3),
            Dense(1, activation='sigmoid')
        ])

    else:
        # without regularization
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(input_dim,)),
            Dense(16, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),  # Dropout layer with 50% dropout rate
            Dense(8, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(4, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(2, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(1, activation='sigmoid')
        ])

    return model

# -- Real Time models -- #
def create_auto_feature_extraction_realtime_LSTM_Model(input_dim, l2_alpha, dropout_rate, time_steps):
    # Build the enhanced model
    model = tf.keras.Sequential([
        # Input layer for time-series data
        tf.keras.layers.Input(shape=(time_steps, input_dim)),

        # 1D Convolutional Layer for feature extraction
        Conv1D(filters=64, kernel_size=3, activation='relu', kernel_regularizer=l2(l2_alpha)),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),

        # Another Conv1D Layer
        Conv1D(filters=128, kernel_size=3, activation='relu', kernel_regularizer=l2(l2_alpha)),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),

        # Recurrent Layer (LSTM)
        LSTM(100, return_sequences=False, activation='tanh', kernel_regularizer=l2(l2_alpha)),
        Dropout(dropout_rate),

        # Fully connected layers
        Dense(64, activation='relu', kernel_regularizer=l2(l2_alpha)),
        BatchNormalization(),
        Dropout(dropout_rate),

        Dense(32, activation='relu', kernel_regularizer=l2(l2_alpha)),
        BatchNormalization(),
        Dropout(dropout_rate),

        Dense(16, activation='relu', kernel_regularizer=l2(l2_alpha)),
        BatchNormalization(),
        Dropout(dropout_rate),

        # Output layer for binary classification
        Dense(1, activation='sigmoid')
    ])

    return model


def create_auto_feature_extraction_realtime_GRU_Model(input_dim, l2_alpha, dropout_rate, time_steps):
    # Build the real-time sequential model
    model = Sequential([
        # Input layer for real-time time-series data
        tf.keras.layers.Input(shape=(time_steps, input_dim)),

        # 1D Convolutional Layer for feature extraction
        Conv1D(filters=64, kernel_size=3, activation='relu', kernel_regularizer=l2(l2_alpha)),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),

        # Another Conv1D Layer
        Conv1D(filters=128, kernel_size=3, activation='relu', kernel_regularizer=l2(l2_alpha)),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),

        # GRU Layer (lighter than LSTM for real-time application)
        GRU(50, return_sequences=False, activation='tanh', kernel_regularizer=l2(l2_alpha)),
        Dropout(dropout_rate),

        # Fully connected layers for classification
        Dense(32, activation='relu', kernel_regularizer=l2(l2_alpha)),
        BatchNormalization(),
        Dropout(dropout_rate),

        Dense(16, activation='relu', kernel_regularizer=l2(l2_alpha)),
        BatchNormalization(),
        Dropout(dropout_rate),

        # Output layer for binary classification
        Dense(1, activation='sigmoid')
    ])

    return model

def create_realtime_GRU_CICIOT_Model(input_dim, regularizationEnabled, DP_enabled, l2_alpha):

    # with regularization
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(1, input_dim)),  # For real-time step input

        # GRU to capture temporal dependencies
        GRU(50, return_sequences=False, activation='tanh'),

        Dense(64, activation='relu', kernel_regularizer=l2(l2_alpha)),
        BatchNormalization(),
        Dropout(0.4),  # Dropout layer with 50% dropout rate

        Dense(32, activation='relu', kernel_regularizer=l2(l2_alpha)),
        BatchNormalization(),
        Dropout(0.4),

        Dense(16, activation='relu', kernel_regularizer=l2(l2_alpha)),
        BatchNormalization(),
        Dropout(0.4),

        Dense(8, activation='relu', kernel_regularizer=l2(l2_alpha)),
        BatchNormalization(),
        Dropout(0.4),

        Dense(4, activation='relu', kernel_regularizer=l2(l2_alpha)),
        BatchNormalization(),
        Dropout(0.4),

        Dense(1, activation='sigmoid')
    ])

    return model

# -- Bishwas's Advanced NIDS Models

def cnn_lstm_gru_model_binary(input_shape):
    """Define and compile CNN-LSTM-GRU model."""
    model = Sequential([
        Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),

        Conv1D(filters=64, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),

        LSTM(64, return_sequences=True),
        GRU(64, return_sequences=False),

        Flatten(),

        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def cnn_lstm_gru_model_multiclass(input_shape):
    """Define and compile CNN-LSTM-GRU model."""
    model = Sequential([
        Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),

        Conv1D(filters=64, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),

        LSTM(64, return_sequences=True),
        GRU(64, return_sequences=False),

        Flatten(),

        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(15, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def cnn_lstm_gru_model_multiclass_dynamic(input_shape, num_classes):
    model = Sequential([
        Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),

        Conv1D(filters=64, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),

        LSTM(64, return_sequences=True),
        GRU(64, return_sequences=False),

        Flatten(),

        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

