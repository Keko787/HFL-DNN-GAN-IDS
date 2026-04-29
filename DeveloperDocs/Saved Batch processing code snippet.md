    # • Fix shape issues - ensure 2D data
       545 -                          if len(benign_data.shape) > 2:
       546 -                              benign_data = tf.reshape(benign_data, (benign_data.shape[0], -1))
       547 -
       548 -                          # • Ensure one-hot encoding
       549 -                          if len(benign_labels.shape) == 1:
       550 -                              benign_labels_onehot = tf.one_hot(tf.cast(benign_labels, tf.int32),
           - depth=self.num_classes)
       551 -                          else:
       552 -                              benign_labels_onehot = benign_labels
       553 -
       554 -                          # • Ensure correct shape for labels
       555 -                          if len(benign_labels_onehot.shape) > 2:
       556 -                              benign_labels_onehot = tf.reshape(benign_labels_onehot,
       557 -                                                                (benign_labels_onehot.shape[0],
           - self.num_classes))
       558 -
       559 -                          # • Create validity labels
       560 -                          valid_smooth_benign = tf.ones((benign_data.shape[0], 1)) * (1 -
           - valid_smoothing_factor)
       561 -


