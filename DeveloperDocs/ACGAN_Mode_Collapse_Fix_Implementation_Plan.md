# AC-GAN Mode Collapse Fix Implementation Plan

## Date: September 12, 2025
## Model: ACGAN for Network Intrusion Detection

## Overview
This document outlines a phased, incremental approach to fixing the mode collapse and generator dominance issues identified in the AC-GAN training. The plan prioritizes easy-to-implement, testable changes with minimal risk.

## Core Issues to Address
1. Generator dominates discriminator (99.73% of fake samples classified as real)
2. Discriminator loses ability to classify real samples correctly
3. Training imbalance leads to catastrophic model failure
4. Insufficient monitoring to detect problems early

## Implementation Phases

### Phase 1: Training Ratio & Monitoring (Easiest - Start Here)

#### 1.1 Adjust Discriminator-to-Generator Training Ratio
**File:** `ACGANCentralTrainingConfig.py`
**Line:** 311
**Change:** `d_to_g_ratio=1` → `d_to_g_ratio=3`
**Risk:** Low
**Expected Impact:** Gives discriminator 3x more training steps to keep up with generator

#### 1.2 Add Discriminator Health Monitoring
**File:** `ACGANCentralTrainingConfig.py`
**Location:** After line 504 (in training loop)
**Implementation:**
```python
# Monitor discriminator health
if d_fake_valid_acc < 0.30:  # Less than 30% accuracy on fake
    self.logger.warning(f"⚠️ Generator dominating! Fake accuracy: {d_fake_valid_acc*100:.2f}%")
if d_benign_valid_acc < 0.70:  # Less than 70% on real benign
    self.logger.warning(f"⚠️ Discriminator confused on benign! Accuracy: {d_benign_valid_acc*100:.2f}%")
if d_attack_valid_acc < 0.70:  # Less than 70% on real attack
    self.logger.warning(f"⚠️ Discriminator confused on attack! Accuracy: {d_attack_valid_acc*100:.2f}%")
```
**Risk:** None (monitoring only)
**Expected Impact:** Early warning system for mode collapse

### Phase 2: Label Smoothing Adjustments

#### 2.1 Increase Label Smoothing Factors
**File:** `ACGANCentralTrainingConfig.py`
**Lines:** 368-378
**Changes:**
```python
# Current → New
valid_smoothing_factor = 0.08  → 0.12  # Line 368
fake_smoothing_factor = 0.05   → 0.10  # Line 371
gen_smoothing_factor = 0.08    → 0.05  # Line 376 (reduce for generator)
```
**Risk:** Low
**Expected Impact:** Prevents overconfidence, improves training stability

### Phase 3: Learning Rate Tuning

#### 3.1 Adjust Learning Rates
**File:** `ACGANCentralTrainingConfig.py`
**Lines:** 102-106
**Changes:**
```python
# Generator learning rate (slow it down)
lr_schedule_gen = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.000005,  # Changed from 0.00001 (50% reduction)
    decay_steps=10000, 
    decay_rate=0.97, 
    staircase=False)

# Discriminator learning rate (keep same or increase slightly)
lr_schedule_disc = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.00004,  # Changed from 0.00003 (33% increase)
    decay_steps=10000, 
    decay_rate=0.97, 
    staircase=False)
```
**Risk:** Low-Medium
**Expected Impact:** Discriminator learns faster relative to generator

### Phase 4: Adaptive Training Improvements

#### 4.1 Make Ratio Adjustment More Responsive
**File:** `ACGANCentralTrainingConfig.py`
**Line:** 574
**Changes:**
```python
# Current
if epoch > 0 and epoch % 5 == 0:  # Adjust every 5 epochs
    if d_g_loss_ratio < 0.5:  
        ...
    elif d_g_loss_ratio > 2.0:
        ...

# New (more responsive)
if epoch > 0 and epoch % 2 == 0:  # Adjust every 2 epochs
    if d_g_loss_ratio < 0.7:  # More sensitive threshold
        d_to_g_ratio = max(1, d_to_g_ratio - 1)
        self.logger.info(f"Adjusting d_to_g_ratio down to {d_to_g_ratio}:1")
    elif d_g_loss_ratio > 1.5:  # More sensitive threshold
        d_to_g_ratio = min(5, d_to_g_ratio + 1)  # Cap at 5:1
        self.logger.info(f"Adjusting d_to_g_ratio up to {d_to_g_ratio}:1")
```
**Risk:** Low
**Expected Impact:** Faster response to training imbalances

### Phase 5: Early Intervention System

#### 5.1 Add Early Stopping for Mode Collapse
**File:** `ACGANCentralTrainingConfig.py`
**Location:** After monitoring section (around line 505)
**Implementation:**
```python
# Track consecutive bad epochs
if not hasattr(self, 'collapse_counter'):
    self.collapse_counter = 0

# Check for mode collapse
if d_fake_valid_acc < 0.20:  # Less than 20% accuracy on fake
    self.collapse_counter += 1
    if self.collapse_counter >= 2:
        self.logger.error("🛑 Mode collapse detected! Stopping training.")
        self.logger.info("Consider loading last checkpoint or adjusting hyperparameters.")
        break  # Exit training loop
else:
    self.collapse_counter = 0  # Reset counter if healthy
```
**Risk:** Low
**Expected Impact:** Prevents complete model failure

#### 5.2 Checkpoint Before Potential Collapse
**File:** `ACGANCentralTrainingConfig.py`
**Location:** In epoch loop, after validation
**Implementation:**
```python
# Save checkpoint if model is healthy
if d_fake_valid_acc > 0.35 and d_fake_valid_acc < 0.65:  # Healthy range
    checkpoint_name = f"healthy_checkpoint_epoch_{epoch}"
    self.save(checkpoint_name)
    self.logger.info(f"✓ Saved healthy checkpoint: {checkpoint_name}")
```

## Testing Strategy

### Test After Each Phase:
1. Run for 5-10 epochs with small dataset
2. Monitor discriminator accuracy on fake samples
3. Check if warnings are triggered appropriately
4. Verify no sudden accuracy drops

### Success Metrics:
- Discriminator maintains 40-60% accuracy on fake samples
- Real sample accuracy stays above 80%
- No mode collapse warnings for consecutive epochs
- Loss ratio stays between 0.7 and 1.5

## Rollback Plan

If any phase causes issues:
1. Revert the specific change
2. Load checkpoint from before the change
3. Reduce the magnitude of adjustment (e.g., ratio 2:1 instead of 3:1)
4. Document what didn't work in this file

## Implementation Order

1. **Start with Phase 1** - Lowest risk, immediate visibility
2. **Add Phase 2** - Simple parameter changes
3. **Implement Phase 3** - Requires monitoring from Phase 1
4. **Add Phase 4** - Builds on previous changes
5. **Finish with Phase 5** - Safety net for all changes

## Notes

- Each phase should be tested independently before moving to the next
- Keep detailed logs of training metrics after each change
- If mode collapse still occurs, consider architectural changes (Phase 6+)
- Document any unexpected behaviors or new insights

## Future Considerations (Phase 6+)

If the above fixes don't resolve the issue:
- Implement Spectral Normalization in discriminator
- Add gradient penalty (WGAN-GP style)
- Consider different GAN variants (LSGAN, WGAN)
- Implement discriminator replay buffer
- Add diversity loss to generator

---

*Last Updated: September 12, 2025*
*Status: Ready for Implementation*