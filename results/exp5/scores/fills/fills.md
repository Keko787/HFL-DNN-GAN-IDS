# Fill cells, outside the studies' Holm families

Each cell against F on the same seeds; p is raw and no claim is made from it. Means take the trials that reached tau; the paired difference (F - variant) takes the seeds both reached.

| table | cell | variant | metric | trials | n_pairs | ref_mean | variant_mean | reach_ref | reach_variant | paired_diff | ci_low | ci_high | p_raw |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mules | n6k1_knee | FQ-g75 | sim_s_to_tau0.71 | 40 | 40 | 178.923 | 168.728 | 1.000 | 1.000 | 10.195 | 1.540 | 20.106 | 0.023 |
| mules | n6k1_stress | FQ-g75 | sim_s_to_tau0.71 | 40 | 40 | 167.635 | 166.353 | 1.000 | 1.000 | 1.282 | -9.933 | 12.322 | 0.638 |
| mules | n6k1_relaxed | FQ-g75 | sim_s_to_tau0.71 | 40 | 40 | 178.923 | 167.706 | 1.000 | 1.000 | 11.217 | 2.247 | 21.278 | 0.023 |
| mules | n6k3_knee | FQ-g75 | sim_s_to_tau0.71 | 40 | 40 | 52.912 | 61.649 | 1.000 | 1.000 | -8.736 | -15.907 | -2.774 | 0.004 |
| mules | n6k3_stress | FQ-g75 | sim_s_to_tau0.71 | 40 | 40 | 52.912 | 61.214 | 1.000 | 1.000 | -8.302 | -15.577 | -2.337 | 0.013 |
| mules | n6k3_relaxed | FQ-g75 | sim_s_to_tau0.71 | 40 | 40 | 52.912 | 61.835 | 1.000 | 1.000 | -8.923 | -15.793 | -3.184 | 0.001 |
| scale | n24k1_stress | FQ-g75 | sim_s_to_tau0.71 | 20 | 20 | 661.028 | 677.773 | 1.000 | 1.000 | -16.745 | -36.746 | -1.214 | 0.053 |
| scale | n24k1_knee | FQ-g75 | sim_s_to_tau0.71 | 20 | 20 | 749.136 | 896.730 | 1.000 | 1.000 | -147.594 | -200.446 | -102.551 | 0.000 |
| robustness | n6k1_knee_dirichlet0.1 | D4 | sim_s_to_tau0.71 | 20 | 10 | 378.372 | 470.837 | 0.550 | 0.500 | -99.602 | -151.307 | -45.744 | 0.006 |
| robustness | n6k1_knee_dirichlet0.1 | D5 | sim_s_to_tau0.71 | 20 | 11 | 378.372 | 481.707 | 0.550 | 0.550 | -103.334 | -158.383 | -54.076 | 0.002 |
| robustness | n6k1_knee_1mb_none | D3 | sim_s_to_tau0.71 | 20 | 20 | 194.048 | 279.558 | 1.000 | 1.000 | -85.510 | -128.452 | -49.453 | 0.000 |
| robustness | n6k1_knee_1mb_spread | D3 | sim_s_to_tau0.71 | 20 | 15 | 137.422 | 293.159 | 0.750 | 1.000 | -119.069 | -193.537 | -47.042 | 0.005 |
| robustness | n6k1_knee_1mb_stragglers | D3 | sim_s_to_tau0.71 | 20 | 14 | 159.406 | 316.686 | 0.700 | 1.000 | -89.578 | -167.028 | -6.199 | 0.049 |
| robustness | n6k1_knee_base | D3 | sim_s_to_tau0.71 | 20 | 19 | 215.972 | 293.214 | 0.950 | 1.000 | -70.322 | -145.090 | -14.975 | 0.009 |
| robustness | n6k1_knee_base | D4 | sim_s_to_tau0.71 | 20 | 19 | 215.972 | 257.222 | 0.950 | 1.000 | -33.739 | -72.207 | 6.492 | 0.045 |
| robustness | n6k1_knee_base | D5 | sim_s_to_tau0.71 | 20 | 18 | 215.972 | 309.438 | 0.950 | 0.950 | -66.803 | -147.847 | -0.581 | 0.038 |
| robustness | n6k1_knee_amp11 | D3 | sim_s_to_tau0.71 | 20 | 19 | 228.761 | 314.177 | 0.950 | 1.000 | -70.295 | -138.596 | -13.211 | 0.018 |
| robustness | n6k1_knee_amp11 | D4 | sim_s_to_tau0.71 | 20 | 19 | 228.761 | 313.737 | 0.950 | 1.000 | -72.439 | -133.990 | -25.481 | 0.005 |
| robustness | n6k1_knee_amp11 | D5 | sim_s_to_tau0.71 | 20 | 18 | 228.761 | 287.711 | 0.950 | 0.950 | -51.267 | -106.523 | 0.875 | 0.060 |
| robustness | n6k1_knee_ampharsh | D3 | sim_s_to_tau0.71 | 20 | 17 | 255.435 | 263.652 | 0.950 | 0.900 | 10.218 | -57.919 | 87.984 | 0.963 |
| robustness | n6k1_knee_ampharsh | D4 | sim_s_to_tau0.71 | 20 | 19 | 255.435 | 276.346 | 0.950 | 1.000 | -16.289 | -115.261 | 84.125 | 0.922 |
| robustness | n6k1_knee_ampharsh | D5 | sim_s_to_tau0.71 | 20 | 18 | 255.435 | 314.776 | 0.950 | 0.950 | -60.337 | -159.943 | 34.510 | 0.523 |
| robustness | n6k1_knee_npl | D3 | sim_s_to_tau0.71 | 20 | 20 | 257.255 | 303.356 | 1.000 | 1.000 | -46.101 | -127.331 | 29.665 | 0.006 |
| robustness | n6k1_knee_npl | D4 | sim_s_to_tau0.71 | 20 | 18 | 257.255 | 236.616 | 1.000 | 0.900 | 22.788 | -28.845 | 94.896 | 0.347 |
| robustness | n6k1_knee_npl | D5 | sim_s_to_tau0.71 | 20 | 19 | 257.255 | 296.303 | 1.000 | 0.950 | -35.418 | -57.598 | -13.008 | 0.001 |
| robustness | n6k1_knee_sigma8 | D3 | sim_s_to_tau0.71 | 20 | 18 | 218.538 | 264.178 | 0.900 | 1.000 | -36.155 | -63.791 | -4.316 | 0.016 |
| robustness | n6k1_knee_sigma8 | D4 | sim_s_to_tau0.71 | 20 | 18 | 218.538 | 262.012 | 0.900 | 1.000 | -36.948 | -75.736 | 3.343 | 0.038 |
| robustness | n6k1_knee_sigma8 | D5 | sim_s_to_tau0.71 | 20 | 18 | 218.538 | 291.605 | 0.900 | 0.950 | -74.076 | -158.889 | -8.125 | 0.027 |
| pins | n6k1_knee_1mb | FBpwide | network_aou_mean | 40 | 40 | 0.715 | 0.733 | 1.000 | 1.000 | -0.019 | -0.040 | 0.000 | 0.115 |
| pins | n6k1_knee_1mb | FBpwide | sim_s_to_tau0.71 | 40 | 40 | 178.923 | 232.104 | 1.000 | 1.000 | -53.181 | -84.642 | -25.009 | 0.000 |
| pins | n6k1_knee_1mb | FBpmedium | network_aou_mean | 40 | 40 | 0.715 | 0.728 | 1.000 | 1.000 | -0.014 | -0.030 | 0.000 | 0.191 |
| pins | n6k1_knee_1mb | FBpmedium | sim_s_to_tau0.71 | 40 | 40 | 178.923 | 190.008 | 1.000 | 1.000 | -11.085 | -36.136 | 8.714 | 0.330 |
| pins | n6k1_knee_1mb | FBpnarrow | network_aou_mean | 40 | 40 | 0.715 | 0.710 | 1.000 | 1.000 | 0.004 | 0.000 | 0.011 | 0.180 |
| pins | n6k1_knee_1mb | FBpnarrow | sim_s_to_tau0.71 | 40 | 40 | 178.923 | 183.560 | 1.000 | 1.000 | -4.636 | -9.916 | -0.179 | 0.069 |

## FerrySim

| N | budget | policy | served share |
|---|---|---|---|
| 6 | 150 | F | 0.575 |
| 6 | 150 | FX | 0.576 |
| 6 | 150 | greedy_1 | 0.576 |
| 6 | 150 | hyb | 0.576 |
| 12 | 180 | F | 0.592 |
| 12 | 180 | FX | 0.588 |
| 12 | 180 | greedy_1 | 0.590 |
| 12 | 180 | hyb | 0.588 |
