# AC-GAN Fix TODO

## Critical Priority (Fix Immediately)

### 1. Discriminator Freezing Architecture - PARTIALLY DONE

**File:** `Config/ModelTrainingConfig/ClientModelTrainingConfig/CentralTrainingConfig/GAN/FullModel/ACGANCentralTrainingConfig.py`

**Issue:** Discriminator freezing is fundamentally broken - no recompilation after freeze/unfreeze, causing gradient conflicts and mode collapse.

**Location:** Lines 318-331 (freeze/unfreeze methods), Lines 606, 628 (training usage)

**Fix Status:**

- [x] ✅ DONE: Add layer-level freezing with explicit for loops (Lines 318-331)
- [x] ❌ INCOMPLETE: Recompile AC-GAN model AFTER freezing discriminator (freeze methods exist but no recompilation)
- [x] ❌ INCOMPLETE: Recompile discriminator AFTER unfreezing (unfreeze methods exist but no recompilation)
- [x] ✅ DONE: Verify no discriminator gradients computed during generator training (Lines 333-363, 608-634)

**Implementation:**

```python
# Before generator training (line ~517):
def freeze_discriminator_for_generator_training(self):
    self.discriminator.trainable = False
    for layer in self.discriminator.layers:
        layer.trainable = False

    # CRITICAL: Recompile AC-GAN
    self.ACGAN.compile(
        loss={'Discriminator': 'binary_crossentropy', 'Discriminator_1': 'categorical_crossentropy'},
        optimizer=self.gen_optimizer,
        metrics={'Discriminator': ['binary_accuracy'], 'Discriminator_1': ['categorical_accuracy']}
    )

# After generator training (line ~530):
def unfreeze_discriminator_for_discriminator_training(self):
    self.discriminator.trainable = True
    for layer in self.discriminator.layers:
        layer.trainable = True

    # Recompile discriminator
    self.discriminator.compile(
        loss={'validity': 'binary_crossentropy', 'class': 'categorical_crossentropy'},
        optimizer=self.disc_optimizer,
        metrics={'validity': ['binary_accuracy'], 'class': ['categorical_accuracy']}
    )
```

### 2. Repeated Data Sampling - Major Performance Bottleneck

**Status:** ✅ DONE

**Location:** Lines 454-494

**Issue:** `tf.random.shuffle()` + `tf.gather()` called 6-10 times per training step (inside d_to_g_ratio loop).

**Fix:** ✅ Pre-sample batches before discriminator ratio loop is now implemented with pre-sampling section.

### 3. Inefficient Generator Usage

**Status:** ✅ DONE

**Location:** Lines 572, 771, 881, 998, 1164

**Issue:** `generator.predict()` used inside training loop instead of direct call.

**Fix:** ✅ All instances now use direct call: `self.generator([noise, fake_labels], training=False)`

- Line 572: Training discriminator on fake data
- Line 771: Validation discriminator
- Line 881: Validation NIDS
- Line 998: Evaluate discriminator
- Line 1164: Evaluate NIDS

**Expected Impact:** 20-30% performance improvement, reduced memory usage.

---

## High Priority

### 4. Inconsistent Batch Sizing

**Status:** ✅ DONE

**Location:** Lines 463-492

**Issue:** If class has fewer samples than batch_size, that class is skipped entirely, creating imbalanced training.

**Fix:** ✅ Now implemented with dynamic batch sizing

- Line 463: `benign_batch_size = min(len(benign_indices), self.batch_size)`
- Line 474: `attack_batch_size = min(len(attack_indices), self.batch_size)`
- Lines 464-471: Train if ANY samples available (benign)
- Lines 475-482: Train if ANY samples available (attack)
- Lines 484-492: Calculate effective fake batch size with fallback

### 5. Mode Collapse Monitoring & Prevention

**Status:** ❌ NOT DONE

**Location:** Should be added after line 693 (after validation)

**Fix:**

- [ ] Add discriminator health monitoring with warnings
- [ ] Implement early stopping for mode collapse detection
- [ ] Add healthy checkpoint saving

**Implementation:**

```python
# Health monitoring
if d_fake_valid_acc < 0.30:
    self.logger.warning(f"⚠️ Generator dominating! Fake accuracy: {d_fake_valid_acc*100:.2f}%")
if d_benign_valid_acc < 0.70 or d_attack_valid_acc < 0.70:
    self.logger.warning(f"⚠️ Discriminator confused on real data!")

# Early stopping
if not hasattr(self, 'collapse_counter'):
    self.collapse_counter = 0

if d_fake_valid_acc < 0.20:
    self.collapse_counter += 1
    if self.collapse_counter >= 2:
        self.logger.error("🛑 Mode collapse detected! Stopping training.")
        break
else:
    self.collapse_counter = 0
```

### 6. Training Ratio Adjustments

**Status:** ❌ PARTIALLY DONE

**Location:** Line 368 (default parameter), Lines 678-684 (adaptive adjustment)

**Current State:**

- Line 368: Default `d_to_g_ratio=1` (still needs to be changed to 3)
- Lines 678-684: Adaptive ratio exists but needs improvement

**Fixes:**

- [ ] Change initial `d_to_g_ratio` from 1 to 3 (give discriminator more training)
- [ ] Make adaptive ratio more responsive (currently every 5 epochs, change to 2)
- [ ] Use more sensitive thresholds (currently 0.5-2.0, change to 0.7-1.5)
- [ ] Cap maximum ratio at 5:1 (currently uncapped at line 683)

---

## Medium Priority

### 7. Redundant Shape Processing

**Status:** ❌ NOT DONE

**Location:** Lines 511-523 (benign), Lines 537-550 (attack)

**Issue:** Same shape processing code duplicated for benign and attack data.

**Fix:**

- [x] Create `process_batch_data(self, data, labels)` helper function
- [x] Use helper for both benign and attack processing
- [x] Ensure consistent processing across all data types

**Note:** Code duplication still exists despite similar processing for both data types.

### 8. Loss Calculation Inefficiencies

**Status:** ❌ NOT DONE (Still Inefficient)

**Location:** Lines 583-593

**Issue:** Weighted loss calculated up to d_to_g_ratio times per training step (every discriminator step).

**Current Implementation:** `calculate_weighted_loss()` called inside d_step loop at line 583

**Fix:**

- [ ] Collect losses during discriminator steps
- [ ] Calculate weighted loss once per step (on last discriminator step)
- [ ] Average accumulated losses for final metrics

**Note:** Lines 598-599 average losses, but weighted calculation still happens every d_step.

### 9. Label Smoothing Adjustments

**Status:** ❌ NOT DONE

**Location:** Lines 414-423

**Current Values:**

- Line 414: `valid_smoothing_factor = 0.08`
- Line 417: `fake_smoothing_factor = 0.05`
- Line 422: `gen_smoothing_factor = 0.08`

**Fix:**

- [x] Increase `valid_smoothing_factor` from 0.08 to 0.12
- [x] Increase `fake_smoothing_factor` from 0.05 to 0.10
- [x] Reduce `gen_smoothing_factor` from 0.08 to 0.05

### 10. Learning Rate Tuning

**Status:** ✅ DONE (Already Better Than Recommended)

**Location:** Lines 103-108

**Current Values:**

- Line 104: Generator LR = 0.00001 (decay 0.98)
- Line 108: Discriminator LR = 0.00005 (decay 0.98)

**Analysis:**

- Discriminator LR (0.00005) is already 5x higher than generator (0.00001)
- This is BETTER than the recommended 0.00004 discriminator LR
- Ratio is 5:1 instead of recommended 4:1
- ✅ This configuration already addresses the concern of discriminator needing faster learning

---

## Lower Priority (Code Quality & Robustness)

### 11. Inefficient Class Separation

**Status:** ❌ NOT DONE

**Location:** Lines 407-408 (training), Lines 748-749 (validation), Lines 952-953 (evaluation)

**Issue:** `tf.where` + `tf.argmax` recalculated every epoch/call.

**Current Implementation:**

- Line 407-408: Training class separation
- Line 748-749: Validation class separation
- Line 952-953: Evaluation class separation

**Fix:**

- [ ] Create `_get_class_indices()` method with caching
- [ ] Calculate once and reuse for training and validation

### 12. Memory Leaks in Validation

**Status:** ❌ NOT DONE

**Location:** Lines 752-755 (validation), Lines 956-963 (evaluation)

**Issue:** Large tensor gathering operations without proper cleanup.

**Current Implementation:**

- Lines 752-755: Validation data gathering
- Lines 956-963: Evaluation data gathering
- No explicit cleanup or CPU device forcing

**Fix:**

- [ ] Use `tf.device('/CPU:0')` for large tensor operations
- [ ] Add explicit tensor cleanup with `del`
- [ ] Monitor memory usage during long training runs

### 13. Missing Error Handling

**Status:** ❌ NOT DONE

**Location:** Various train_on_batch calls (Lines 529, 556, 578, 624)

**Issue:** No error handling for train_on_batch failures.

**Current Implementation:**

- Line 529: `self.discriminator.train_on_batch(benign_data, ...)` - no error handling
- Line 556: `self.discriminator.train_on_batch(attack_data, ...)` - no error handling
- Line 578: `self.discriminator.train_on_batch(generated_data, ...)` - no error handling
- Line 624: `self.ACGAN.train_on_batch([noise, sampled_labels], ...)` - no error handling

**Fix:**

- [ ] Create `safe_train_on_batch()` wrapper function
- [ ] Check for NaN/Inf in losses
- [ ] Handle `ResourceExhaustedError` gracefully
- [ ] Add recovery mechanisms for common failures

### 14. Logging Improvements

**Status:** ✅ MOSTLY DONE

**Current Implementation:**

- [x] ✅ DONE: Epoch numbering display (Lines 439-440, 660, 689)
- [x] ✅ DONE: Generator fooling rate in logs (Lines 655, 663, 845, 1134, 1143)
- [x] ✅ DONE: Per-class metrics for generator (Lines 651-657 - includes class accuracy)
- [x] ✅ DONE: Discriminator accuracy/loss formatting (Lines 1280-1293 - comprehensive metrics)

**All logging improvements are already implemented!**

---

## Testing Strategy

**After Each Fix:**

1. Run 5-10 epochs with small dataset
2. Monitor discriminator accuracy on fake samples (target: 40-60%)
3. Verify real sample accuracy stays above 80%
4. Check for warnings/errors in logs

**Success Metrics:**

- Discriminator maintains 40-60% accuracy on fake samples
- Real sample accuracy stays above 80%
- No mode collapse warnings for consecutive epochs
- Loss ratio stays between 0.7 and 1.5
- Training completes without crashes

---

## Implementation Order

1. ✅ Fix repeated data sampling (DONE - Lines 454-494)
2. ✅ Fix generator predict() usage (DONE - All instances use direct call)
3. ✅ Fix inconsistent batch sizing (DONE - Dynamic batch sizing implemented)
4. ✅ Fix logging issues (DONE - All logging improvements implemented)
5. ✅ Learning rate tuning (DONE - Already better than recommended)
6. **Fix discriminator freezing recompilation** (CRITICAL - partial, needs recompilation added)
7. **Add mode collapse monitoring** (HIGH PRIORITY - prevent catastrophic failure)
8. **Adjust training ratio defaults** (MEDIUM - change default from 1 to 3)
9. Refactor shape processing (MEDIUM - reduce code duplication)
10. Optimize loss calculations (MEDIUM - reduce redundant weighted loss calls)
11. Add label smoothing adjustments (LOW)
12. Add error handling (LOW - robustness)
13. Optimize class separation with caching (LOW - performance)
14. Fix memory leaks in validation (LOW - long-term stability)

---

## Expected Improvements

**Performance:**

- 20-40% faster training (from generator and sampling fixes)
- 50-70% less memory usage (from redundancy removal)

**Stability:**

- No mode collapse with proper monitoring
- Discriminator maintains ability to classify real/fake
- More balanced training between generator and discriminator

**Code Quality:**

- Cleaner, more maintainable code
- Better error handling and recovery
- Comprehensive logging and monitoring

---

**Last Updated:** 2025-10-02 - Compared against ACGANCentralTrainingConfig.py

**Based on analysis from:**

- ACGAN_Discriminator_Freezing_Analysis.md
- ACGAN_Additional_Training_Flaws_Analysis.md
- GAN_Training_Mode_Collapse_Analysis.md
- ACGAN_Mode_Collapse_Fix_Implementation_Plan.md

**Overall Progress:** 5/14 items fully complete, 2/14 partially complete, 7/14 remaining

**Priority Actions:**

1. Add recompilation to discriminator freeze/unfreeze methods
2. Implement mode collapse monitoring and early stopping
3. Change default d_to_g_ratio from 1 to 3
