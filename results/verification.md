Monte Carlo: 200000 runs per case, seed 8.

| metric | slit 0.5, mean | slit 0.5, cov | slit 0.2, mean | slit 0.2, cov |
|---|---|---|---|---|
| route | lower | upper | lower | lower |
| optimal cost J | 201.683 | 116.423 | 201.683 | 139.084 |
|   mean part | 201.683 | 115.993 | 201.683 | 138.336 |
|   covariance part | 0.000 | 0.430 | 0.000 | 0.748 |
| MC average cost | 201.683 | 116.413 | 201.683 | 139.098 |
| certified optimality gap | 0e+00 | 5e-10 | 0e+00 | 9e-06 |
| solve time [s] | 0.7 | 48.3 | 0.6 | 40.0 |
| terminal mean error | 4.5e-10 | 2.8e-10 | 4.5e-10 | 2.3e-09 |
| lambda_max(Sigma_xN - Sigma_N)  (<=0 ok) | +6.9e-02 | -1.7e-12 | +6.9e-02 | -1.4e-11 |
| worst chance slack (<=0 ok) | +2.4e-10 | +4.8e-11 | +2.4e-10 | +7.2e-10 |
| MC worst per-face violation (eps=0.001) | 0.00111 | 0.00109 | 0.00111 | 0.00114 |
| MC worst per-step collision rate | 0.00108 | 0.00000 | 0.00108 | 0.00000 |
| MC trajectory collision (sampled k) | 0.00125 | 0.00000 | 0.00125 | 0.00000 |
| MC trajectory collision (continuous) | 0.00136 | 0.00001 | 0.00136 | 0.00011 |
