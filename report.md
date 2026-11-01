# Fire occurrence vs FWI-System indices

Training years [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025], hold-out []. 8,266,338 land cell-days; a fire is at least one unique daytime VIIRS detection in the 0.25 degree cell that day.

## Skill (AUC: 0.5 = no skill; pred/obs: 1.0 = right on average)

| split             |   region |   fine |   fine_dc |   full |
|:------------------|---------:|-------:|----------:|-------:|
| leave-out 2014    |    0.537 |  0.818 |     0.827 |  0.829 |
| leave-out 2015    |    0.556 |  0.83  |     0.843 |  0.844 |
| leave-out 2016    |    0.494 |  0.781 |     0.791 |  0.793 |
| leave-out 2017    |    0.565 |  0.791 |     0.798 |  0.802 |
| leave-out 2018    |    0.551 |  0.825 |     0.834 |  0.836 |
| leave-out 2019    |    0.571 |  0.831 |     0.845 |  0.845 |
| leave-out 2020    |    0.542 |  0.778 |     0.784 |  0.785 |
| leave-out 2021    |    0.531 |  0.776 |     0.78  |  0.782 |
| leave-out 2022    |    0.551 |  0.776 |     0.779 |  0.782 |
| leave-out 2023    |    0.552 |  0.827 |     0.836 |  0.838 |
| leave-out 2024    |    0.57  |  0.814 |     0.821 |  0.823 |
| leave-out 2025    |    0.556 |  0.807 |     0.808 |  0.811 |
| train (in sample) |    0.553 |  0.819 |     0.828 |  0.83  |

| split             |   region |   fine |   fine_dc |   full |
|:------------------|---------:|-------:|----------:|-------:|
| leave-out 2014    |     0.68 |   0.78 |      0.79 |   0.82 |
| leave-out 2015    |     0.62 |   0.81 |      1    |   1.01 |
| leave-out 2016    |     1.6  |   1.25 |      1.2  |   1.18 |
| leave-out 2017    |     2.32 |   1.82 |      1.65 |   1.62 |
| leave-out 2018    |     0.91 |   0.94 |      0.94 |   0.94 |
| leave-out 2019    |     0.61 |   0.83 |      0.97 |   0.97 |
| leave-out 2020    |     1.6  |   1.33 |      1.23 |   1.2  |
| leave-out 2021    |     1.7  |   1.49 |      1.34 |   1.33 |
| leave-out 2022    |     1.92 |   1.41 |      1.23 |   1.21 |
| leave-out 2023    |     0.64 |   0.75 |      0.78 |   0.78 |
| leave-out 2024    |     0.9  |   0.94 |      0.88 |   0.89 |
| leave-out 2025    |     1.09 |   1.07 |      0.96 |   0.96 |
| train (in sample) |     1    |   1    |      1    |   1    |

## Coefficients (model fine_dc, all training years; standard errors clustered by date)

| term                       |   coef |    se |   lo95 |   hi95 |
|:---------------------------|-------:|------:|-------:|-------:|
| const                      |  2.728 | 0.053 |  2.624 |  2.831 |
| ln_fmc                     | -1.595 | 0.013 | -1.621 | -1.569 |
| wind                       |  0.001 | 0.002 | -0.002 |  0.004 |
| dc100                      |  0.261 | 0.006 |  0.249 |  0.274 |
| island_Sumatra             | -0.293 | 0.019 | -0.329 | -0.257 |
| island_Peninsular Malaysia | -1.117 | 0.025 | -1.166 | -1.068 |
| island_Java                | -1.438 | 0.03  | -1.497 | -1.38  |
| island_Sarawak             | -0.403 | 0.021 | -0.444 | -0.363 |
| island_Sabah               | -0.644 | 0.022 | -0.687 | -0.601 |

## Coefficients (model full)

| term                       |   coef |    se |   lo95 |   hi95 |
|:---------------------------|-------:|------:|-------:|-------:|
| const                      |  0.64  | 0.102 |  0.439 |  0.841 |
| ln_fmc                     | -1.205 | 0.021 | -1.245 | -1.164 |
| wind                       |  0.005 | 0.002 |  0.002 |  0.009 |
| dc100                      |  0.131 | 0.008 |  0.115 |  0.147 |
| ln_dmc                     |  0.426 | 0.018 |  0.391 |  0.461 |
| island_Sumatra             | -0.296 | 0.019 | -0.332 | -0.259 |
| island_Peninsular Malaysia | -1.133 | 0.025 | -1.181 | -1.084 |
| island_Java                | -1.435 | 0.029 | -1.493 | -1.378 |
| island_Sarawak             | -0.387 | 0.021 | -0.427 | -0.346 |
| island_Sabah               | -0.607 | 0.022 | -0.651 | -0.564 |

## Calibration of the full model on the hold-out year(s)

| split   | n   | predicted   | observed   |
|---------|-----|-------------|------------|
