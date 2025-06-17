import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Actual data: 9 points divided into 3 categories (3 points each)
# Category A (red circles)
category_a_x = [205.82, 196.17, 195.85]
category_a_y = [284.15, 278.35, 279.20]

# Category B (blue squares)
category_b_x = [179.81, 160.78, 165.30]
category_b_y = [172.2, 176.3, 178.7]

# Category C (green triangles)
category_c_x = [119.55, 146.18, 146.12]
category_c_y = [174.3, 176.5, 171.4]

# Combine all data for transformations
all_x = np.array(category_a_x + category_b_x + category_c_x)
all_y = np.array(category_a_y + category_b_y + category_c_y)

# ==================== METHOD 1: SQUARE ROOT TRANSFORMATION ====================
print("Method 1: Square Root Transformation (Less aggressive than log)")
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
# Original data
plt.scatter(category_a_x, category_a_y, color='red', marker='o', s=100, label='Flight Training', alpha=0.7)
plt.scatter(category_b_x, category_b_y, color='blue', marker='s', s=100, label='FLWR Training', alpha=0.7)
plt.scatter(category_c_x, category_c_y, color='green', marker='^', s=100, label='Centralized Training', alpha=0.7)
plt.title('Original Data')
plt.xlabel('Time (Seconds)')
plt.ylabel('CPU Usage (%)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 2)
# Square root transformation
sqrt_a_x = [np.sqrt(x) for x in category_a_x]
sqrt_a_y = [np.sqrt(y) for y in category_a_y]
sqrt_b_x = [np.sqrt(x) for x in category_b_x]
sqrt_b_y = [np.sqrt(y) for y in category_b_y]
sqrt_c_x = [np.sqrt(x) for x in category_c_x]
sqrt_c_y = [np.sqrt(y) for y in category_c_y]

plt.scatter(sqrt_a_x, sqrt_a_y, color='red', marker='o', s=100, label='Flight Training', alpha=0.7)
plt.scatter(sqrt_b_x, sqrt_b_y, color='blue', marker='s', s=100, label='FLWR Training', alpha=0.7)
plt.scatter(sqrt_c_x, sqrt_c_y, color='green', marker='^', s=100, label='Centralized Training', alpha=0.7)
plt.title('Square Root Transformed')
plt.xlabel('√Time')
plt.ylabel('√CPU Usage')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 3)
# Log2 transformation
log2_a_x = [np.log2(x) for x in category_a_x]
log2_a_y = [np.log2(y) for y in category_a_y]
log2_b_x = [np.log2(x) for x in category_b_x]
log2_b_y = [np.log2(y) for y in category_b_y]
log2_c_x = [np.log2(x) for x in category_c_x]
log2_c_y = [np.log2(y) for y in category_c_y]

plt.scatter(log2_a_x, log2_a_y, color='red', marker='o', s=100, label='Flight Training', alpha=0.7)
plt.scatter(log2_b_x, log2_b_y, color='blue', marker='s', s=100, label='FLWR Training', alpha=0.7)
plt.scatter(log2_c_x, log2_c_y, color='green', marker='^', s=100, label='Centralized Training', alpha=0.7)
plt.title('Log₂ Transformed')
plt.xlabel('Log₂(Time)')
plt.ylabel('Log₂(CPU Usage)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ==================== METHOD 2: STANDARDIZATION (Z-SCORES) ====================
print("\nMethod 2: Standardization (Z-scores) - Centers data around mean")

# Standardize the data
scaler = StandardScaler()
combined_data = np.column_stack([all_x, all_y])
standardized_data = scaler.fit_transform(combined_data)

# Split back into categories
std_a_x = standardized_data[0:3, 0]
std_a_y = standardized_data[0:3, 1]
std_b_x = standardized_data[3:6, 0]
std_b_y = standardized_data[3:6, 1]
std_c_x = standardized_data[6:9, 0]
std_c_y = standardized_data[6:9, 1]

plt.figure(figsize=(10, 8))
plt.scatter(std_a_x, std_a_y, color='red', marker='o', s=100, label='Flight Training', alpha=0.7)
plt.scatter(std_b_x, std_b_y, color='blue', marker='s', s=100, label='FLWR Training', alpha=0.7)
plt.scatter(std_c_x, std_c_y, color='green', marker='^', s=100, label='Centralized Training', alpha=0.7)
plt.title('Standardized Data (Z-scores)')
plt.xlabel('Standardized Time (Z-score)')
plt.ylabel('Standardized CPU Usage (Z-score)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
plt.axvline(x=0, color='black', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# ==================== METHOD 3: MIN-MAX NORMALIZATION ====================
print("\nMethod 3: Min-Max Normalization (0-1 scale)")

# Normalize to 0-1 range
minmax_scaler = MinMaxScaler()
normalized_data = minmax_scaler.fit_transform(combined_data)

# Split back into categories
norm_a_x = normalized_data[0:3, 0]
norm_a_y = normalized_data[0:3, 1]
norm_b_x = normalized_data[3:6, 0]
norm_b_y = normalized_data[3:6, 1]
norm_c_x = normalized_data[6:9, 0]
norm_c_y = normalized_data[6:9, 1]

plt.figure(figsize=(10, 8))
plt.scatter(norm_a_x, norm_a_y, color='red', marker='o', s=100, label='Flight Training', alpha=0.7)
plt.scatter(norm_b_x, norm_b_y, color='blue', marker='s', s=100, label='FLWR Training', alpha=0.7)
plt.scatter(norm_c_x, norm_c_y, color='green', marker='^', s=100, label='Centralized Training', alpha=0.7)
plt.title('Min-Max Normalized Data (0-1 scale)')
plt.xlabel('Normalized Time')
plt.ylabel('Normalized CPU Usage')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ==================== METHOD 4: FOCUS ON RELATIVE DIFFERENCES ====================
print("\nMethod 4: Relative Differences (Subtract category means)")

# Calculate means for each category
mean_a_x, mean_a_y = np.mean(category_a_x), np.mean(category_a_y)
mean_b_x, mean_b_y = np.mean(category_b_x), np.mean(category_b_y)
mean_c_x, mean_c_y = np.mean(category_c_x), np.mean(category_c_y)

# Subtract means from each category
rel_a_x = [x - mean_a_x for x in category_a_x]
rel_a_y = [y - mean_a_y for y in category_a_y]
rel_b_x = [x - mean_b_x for x in category_b_x]
rel_b_y = [y - mean_b_y for y in category_b_y]
rel_c_x = [x - mean_c_x for x in category_c_x]
rel_c_y = [y - mean_c_y for y in category_c_y]

plt.figure(figsize=(10, 8))
plt.scatter(rel_a_x, rel_a_y, color='red', marker='o', s=100, label='Flight Training', alpha=0.7)
plt.scatter(rel_b_x, rel_b_y, color='blue', marker='s', s=100, label='FLWR Training', alpha=0.7)
plt.scatter(rel_c_x, rel_c_y, color='green', marker='^', s=100, label='Centralized Training', alpha=0.7)
plt.title('Relative Differences (Centered on Category Means)')
plt.xlabel('Time - Category Mean')
plt.ylabel('CPU Usage - Category Mean')
plt.legend()
plt.grid(True, alpha=0.3)
plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
plt.axvline(x=0, color='black', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# ==================== SUMMARY OF APPROACHES ====================
print("\n" + "="*60)
print("SUMMARY OF APPROACHES TO REDUCE THE GAP:")
print("="*60)
print("1. SQUARE ROOT: Less aggressive than log, still compresses large values")
print("2. STANDARDIZATION: Centers all data around 0, shows relative positions")
print("3. MIN-MAX NORMALIZATION: Scales everything to 0-1 range")
print("4. RELATIVE DIFFERENCES: Shows variation within each category")
print("\nRECOMMENDATION:")
print("- For scientific analysis: Use STANDARDIZATION (Method 2)")
print("- For general visualization: Use SQUARE ROOT (Method 1)")
print("- For comparing patterns: Use RELATIVE DIFFERENCES (Method 4)")