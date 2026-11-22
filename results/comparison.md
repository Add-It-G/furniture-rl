## Results on the 50 unseen test rooms

| Algorithm | Mean score | Std | Median | Valid layouts | Avg violations | Evals | Sec/room |
|---|---|---|---|---|---|---|---|
| Random search | 1.09 | 3.64 | 2.71 | 21/50 | 4.9 | 5000 | 0.753 |
| Greedy | 1.51 | 3.84 | 3.23 | 28/50 | 4.3 | 635 | 0.086 |
| Simulated annealing | 3.84 | 1.67 | 4.26 | 42/50 | 0.8 | 5000 | 0.694 |
| PPO 300k | -20.59 | 9.21 | -21.80 | 0/50 | 36.2 | 81 | 0.023 |
| PPO 3M | -16.57 | 8.94 | -16.46 | 0/50 | 31.3 | 127 | 0.031 |

## Paired comparison (same rooms, same starting layouts)

Difference = first minus second; positive favours the first. 95% bootstrap CI on the mean paired difference (10,000 resamples of rooms).

| First vs second | First wins | Ties | Second wins | Mean diff [95% CI] |
|---|---|---|---|---|
| PPO 3M vs Random search | 0 | 0 | 50 | -17.67 [-19.79, -15.57] |
| PPO 3M vs Greedy | 0 | 0 | 50 | -18.08 [-20.27, -15.93] |
| PPO 3M vs Simulated annealing | 0 | 0 | 50 | -20.42 [-22.71, -18.10] |
| PPO 3M vs PPO 300k | 30 | 1 | 19 | +4.02 [+1.49, +6.61] |
| PPO 300k vs Random search | 0 | 0 | 50 | -21.69 [-23.95, -19.61] |
| PPO 300k vs Greedy | 0 | 0 | 50 | -22.10 [-24.31, -20.00] |
| PPO 300k vs Simulated annealing | 0 | 0 | 50 | -24.44 [-26.86, -22.10] |
| Simulated annealing vs Random search | 46 | 0 | 4 | +2.75 [+2.09, +3.44] |
| Simulated annealing vs Greedy | 48 | 0 | 2 | +2.33 [+1.58, +3.18] |
| Greedy vs Random search | 30 | 0 | 20 | +0.42 [-0.20, +1.00] |

**Pre-registered criteria applied to PPO 300k:** LOSS (mean -20.59, 0/50 valid)
- PPO 300k one-time training cost: 300,000 steps, 85 s

**Pre-registered criteria applied to PPO 3M:** LOSS (mean -16.57, 0/50 valid)
- PPO 3M one-time training cost: 3,000,000 steps, 941 s
