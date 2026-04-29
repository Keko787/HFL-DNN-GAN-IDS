 The Problem

  For each discriminator step in the loop, the code does:
  benign_idx = tf.random.shuffle(benign_indices)[:benign_batch_size]

  This shuffles the entire pool of benign indices and takes the first benign_batch_size samples. Because it's random
   each time, you could get:

- Batch 1: samples [5, 12, 3, 8]

- Batch 2: samples [12, 5, 9, 3]  ← duplicates from batch 1!

- Batch 3: samples [3, 8, 15, 5]  ← more duplicates!
  The Solution
  To ensure no repetition within the same training step, the code should shuffle once and then slice consecutive
  batches:
  
  # Shuffle ONCE before the loop
  
  shuffled_benign_indices = tf.random.shuffle(benign_indices)
  shuffled_attack_indices = tf.random.shuffle(attack_indices)
  for d_step in range(d_to_g_ratio):
  
  # Calculate start/end for this batch
  
    benign_start = d_step * self.batch_size
    benign_end = min(benign_start + self.batch_size, len(benign_indices))
    if benign_start < len(benign_indices):
  
        # Take consecutive slice from shuffled indices
        benign_idx = shuffled_benign_indices[benign_start:benign_end]
        benign_batch_data = tf.gather(X_train, benign_idx)
        benign_batch_labels = tf.gather(y_train, benign_idx)
        benign_batches.append((benign_batch_data, benign_batch_labels))
  
    else:
  
        benign_batches.append(None)
  
  # Same pattern for attack batches...
  
  This ensures:

- Each sample is used at most once per training step

- When d_to_g_ratio > 1, different batches get different data

- If you run out of data (e.g., d_to_g_ratio=5 but only 2 batches worth), remaining batches are None
  
  

The Solution

  To ensure no repetition within the same training step, the code should shuffle once and then slice consecutive
  batches:

# Shuffle ONCE before the loop

  shuffled_benign_indices = tf.random.shuffle(benign_indices)
  shuffled_attack_indices = tf.random.shuffle(attack_indices)



  for d_step in range(d_to_g_ratio):
      # Calculate start/end for this batch
      benign_start = d_step * self.batch_size
      benign_end = min(benign_start + self.batch_size, len(benign_indices))


      if benign_start < len(benign_indices):
          # Take consecutive slice from shuffled indices
          benign_idx = shuffled_benign_indices[benign_start:benign_end]
          benign_batch_data = tf.gather(X_train, benign_idx)
          benign_batch_labels = tf.gather(y_train, benign_idx)
          benign_batches.append((benign_batch_data, benign_batch_labels))


      else:
          benign_batches.append(None)

      # Same pattern for attack batches...



# Shuffle once per step

  shuffled_benign = tf.random.shuffle(benign_indices)
  shuffled_attack = tf.random.shuffle(attack_indices)



  for d_step in range(d_to_g_ratio):


      # Calculate batch boundaries
      benign_start = d_step * self.batch_size
      benign_end = min(benign_start + self.batch_size, 

len(benign_indices))
      # Only create batch if we have samples left
      if benign_start < len(benign_indices):
          actual_batch_size = benign_end - benign_start
          benign_idx = shuffled_benign[benign_start:benign_end]
          benign_batches.append((
              tf.gather(X_train, benign_idx),
              tf.gather(y_train, benign_idx)
          ))
      else:
          # No more data - skip this discriminator step for benign
          benign_batches.append(None)


