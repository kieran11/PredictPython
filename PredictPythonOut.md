Untitled
================

## R Markdown

``` python

import pandas as pd
import numpy as np

AnalyticalDS = pd.read_csv('AnalysisDS.csv')
```

![](PredictPythonOut_files/figure-gfm/showR-1.png)<!-- -->

First, we get the variable names.

``` python

Dimension = AnalyticalDS.shape

print(Dimension)
```

    ## (949, 50)

![](PredictPythonOut_files/figure-gfm/DenPymp-1.png)<!-- -->

![](PredictPythonOut_files/figure-gfm/DenPyNS-1.png)<!-- -->

The next step is to finalize the dataset, the training dataset, and then
the testing dataset. We first need to select the variables of interest,
and then divide the data up.

``` python

AboveMP_zero = AnalyticalDS['mp'] > 0
NotCurrent = AnalyticalDS['LastSeason'] != "2019-20"
AnalyticalDS_2 = AnalyticalDS[AboveMP_zero]
AnalyticalDS_2_a = AnalyticalDS_2[NotCurrent]
```

    ## /Users/rebeccashapiro/Library/r-miniconda/envs/r-reticulate/bin/python:1: UserWarning: Boolean Series key will be reindexed to match DataFrame index.

``` python
AnalyticDS_3 = AnalyticalDS_2_a[['link' ,'mp', 'age', 'FirstSeasonMP', 'fg', 'fga', 'ftpercent', 'FirstSeasonMP', 'PickNumber', 'fga', 'x3p', 'x3pa', 'x3ppercent', 'x2p' , 
'x2pa', 'x2ppercent', 'fgpercent', 'ft', 'fta', 'efgpercent', 'orb', 'drb', 'trb', 'ast', 'stl', 'blk', 'tov', 'pts','wprev', 'w_lpercentprev',  'wcur', 'w_lpercentcur']]
```

    ## /Users/rebeccashapiro/Library/r-miniconda/envs/r-reticulate/bin/python:1: SettingWithCopyWarning: 
    ## A value is trying to be set on a copy of a slice from a DataFrame.
    ## Try using .loc[row_indexer,col_indexer] = value instead
    ## 
    ## See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy

    ## /Users/rebeccashapiro/Library/r-miniconda/envs/r-reticulate/bin/python:1: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().

    ## /Users/rebeccashapiro/Library/r-miniconda/envs/r-reticulate/bin/python:1: SettingWithCopyWarning: 
    ## A value is trying to be set on a copy of a slice from a DataFrame.
    ## Try using .loc[row_indexer,col_indexer] = value instead
    ## 
    ## See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy

    ## GridSearchCV(cv=5, estimator=Lasso(),
    ##              param_grid={'alpha': [1e-15, 1e-10, 1e-08, 0.0001, 0.001, 0.01, 1,
    ##                                    5, 10, 20]},
    ##              scoring='neg_mean_squared_error')
    ## 
    ## /Users/rebeccashapiro/Library/r-miniconda/envs/r-reticulate/lib/python3.6/site-packages/sklearn/linear_model/_coordinate_descent.py:531: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 10985700740.741121, tolerance: 5154478.066487458
    ##   positive)
    ## /Users/rebeccashapiro/Library/r-miniconda/envs/r-reticulate/lib/python3.6/site-packages/sklearn/linear_model/_coordinate_descent.py:531: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 10390588905.999655, tolerance: 4945087.043317857
    ##   positive)
    ## /Users/rebeccashapiro/Library/r-miniconda/envs/r-reticulate/lib/python3.6/site-packages/sklearn/linear_model/_coordinate_descent.py:531: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 9994046410.470337, tolerance: 4854611.256816057
    ##   positive)
    ## /Users/rebeccashapiro/Library/r-miniconda/envs/r-reticulate/lib/python3.6/site-packages/sklearn/linear_model/_coordinate_descent.py:531: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 11359757501.419374, tolerance: 5029514.394179694
    ##   positive)
    ## /Users/rebeccashapiro/Library/r-miniconda/envs/r-reticulate/lib/python3.6/site-packages/sklearn/linear_model/_coordinate_descent.py:531: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 11798032242.429066, tolerance: 5232437.209593743
    ##   positive)
    ## /Users/rebeccashapiro/Library/r-miniconda/envs/r-reticulate/lib/python3.6/site-packages/sklearn/linear_model/_coordinate_descent.py:531: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 10985700740.519518, tolerance: 5154478.066487458
    ##   positive)
    ## /Users/rebeccashapiro/Library/r-miniconda/envs/r-reticulate/lib/python3.6/site-packages/sklearn/linear_model/_coordinate_descent.py:531: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 10390588905.747108, tolerance: 4945087.043317857
    ##   positive)
    ## /Users/rebeccashapiro/Library/r-miniconda/envs/r-reticulate/lib/python3.6/site-packages/sklearn/linear_model/_coordinate_descent.py:531: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 9994046410.245531, tolerance: 4854611.256816057
    ##   positive)
    ## /Users/rebeccashapiro/Library/r-miniconda/envs/r-reticulate/lib/python3.6/site-packages/sklearn/linear_model/_coordinate_descent.py:531: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 11359757501.144567, tolerance: 5029514.394179694
    ##   positive)
    ## /Users/rebeccashapiro/Library/r-miniconda/envs/r-reticulate/lib/python3.6/site-packages/sklearn/linear_model/_coordinate_descent.py:531: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 11798032242.117496, tolerance: 5232437.209593743
    ##   positive)
    ## /Users/rebeccashapiro/Library/r-miniconda/envs/r-reticulate/lib/python3.6/site-packages/sklearn/linear_model/_coordinate_descent.py:531: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 10985700718.579382, tolerance: 5154478.066487458
    ##   positive)
    ## /Users/rebeccashapiro/Library/r-miniconda/envs/r-reticulate/lib/python3.6/site-packages/sklearn/linear_model/_coordinate_descent.py:531: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 10390588880.744528, tolerance: 4945087.043317857
    ##   positive)
    ## /Users/rebeccashapiro/Library/r-miniconda/envs/r-reticulate/lib/python3.6/site-packages/sklearn/linear_model/_coordinate_descent.py:531: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 9994046387.989069, tolerance: 4854611.256816057
    ##   positive)
    ## /Users/rebeccashapiro/Library/r-miniconda/envs/r-reticulate/lib/python3.6/site-packages/sklearn/linear_model/_coordinate_descent.py:531: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 11359757473.93723, tolerance: 5029514.394179694
    ##   positive)
    ## /Users/rebeccashapiro/Library/r-miniconda/envs/r-reticulate/lib/python3.6/site-packages/sklearn/linear_model/_coordinate_descent.py:531: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 11798032211.272367, tolerance: 5232437.209593743
    ##   positive)
    ## /Users/rebeccashapiro/Library/r-miniconda/envs/r-reticulate/lib/python3.6/site-packages/sklearn/linear_model/_coordinate_descent.py:531: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 10985479121.899914, tolerance: 5154478.066487458
    ##   positive)
    ## /Users/rebeccashapiro/Library/r-miniconda/envs/r-reticulate/lib/python3.6/site-packages/sklearn/linear_model/_coordinate_descent.py:531: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 10390336345.832752, tolerance: 4945087.043317857
    ##   positive)
    ## /Users/rebeccashapiro/Library/r-miniconda/envs/r-reticulate/lib/python3.6/site-packages/sklearn/linear_model/_coordinate_descent.py:531: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 9993821594.415537, tolerance: 4854611.256816057
    ##   positive)
    ## /Users/rebeccashapiro/Library/r-miniconda/envs/r-reticulate/lib/python3.6/site-packages/sklearn/linear_model/_coordinate_descent.py:531: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 11359482677.888142, tolerance: 5029514.394179694
    ##   positive)
    ## /Users/rebeccashapiro/Library/r-miniconda/envs/r-reticulate/lib/python3.6/site-packages/sklearn/linear_model/_coordinate_descent.py:531: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 11797720672.06158, tolerance: 5232437.209593743
    ##   positive)
    ## /Users/rebeccashapiro/Library/r-miniconda/envs/r-reticulate/lib/python3.6/site-packages/sklearn/linear_model/_coordinate_descent.py:531: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 10983484421.493689, tolerance: 5154478.066487458
    ##   positive)
    ## /Users/rebeccashapiro/Library/r-miniconda/envs/r-reticulate/lib/python3.6/site-packages/sklearn/linear_model/_coordinate_descent.py:531: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 10388062493.458475, tolerance: 4945087.043317857
    ##   positive)
    ## /Users/rebeccashapiro/Library/r-miniconda/envs/r-reticulate/lib/python3.6/site-packages/sklearn/linear_model/_coordinate_descent.py:531: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 9991797928.146349, tolerance: 4854611.256816057
    ##   positive)
    ## /Users/rebeccashapiro/Library/r-miniconda/envs/r-reticulate/lib/python3.6/site-packages/sklearn/linear_model/_coordinate_descent.py:531: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 11357009078.81084, tolerance: 5029514.394179694
    ##   positive)
    ## /Users/rebeccashapiro/Library/r-miniconda/envs/r-reticulate/lib/python3.6/site-packages/sklearn/linear_model/_coordinate_descent.py:531: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 11794916230.360107, tolerance: 5232437.209593743
    ##   positive)
    ## /Users/rebeccashapiro/Library/r-miniconda/envs/r-reticulate/lib/python3.6/site-packages/sklearn/linear_model/_coordinate_descent.py:531: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 10963524458.328148, tolerance: 5154478.066487458
    ##   positive)
    ## /Users/rebeccashapiro/Library/r-miniconda/envs/r-reticulate/lib/python3.6/site-packages/sklearn/linear_model/_coordinate_descent.py:531: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 10365243428.760094, tolerance: 4945087.043317857
    ##   positive)
    ## /Users/rebeccashapiro/Library/r-miniconda/envs/r-reticulate/lib/python3.6/site-packages/sklearn/linear_model/_coordinate_descent.py:531: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 9971529354.047935, tolerance: 4854611.256816057
    ##   positive)
    ## /Users/rebeccashapiro/Library/r-miniconda/envs/r-reticulate/lib/python3.6/site-packages/sklearn/linear_model/_coordinate_descent.py:531: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 11332254328.80289, tolerance: 5029514.394179694
    ##   positive)
    ## /Users/rebeccashapiro/Library/r-miniconda/envs/r-reticulate/lib/python3.6/site-packages/sklearn/linear_model/_coordinate_descent.py:531: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 11766841367.954082, tolerance: 5232437.209593743
    ##   positive)
    ## /Users/rebeccashapiro/Library/r-miniconda/envs/r-reticulate/lib/python3.6/site-packages/sklearn/linear_model/_coordinate_descent.py:531: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 8776379913.512518, tolerance: 5154478.066487458
    ##   positive)
    ## /Users/rebeccashapiro/Library/r-miniconda/envs/r-reticulate/lib/python3.6/site-packages/sklearn/linear_model/_coordinate_descent.py:531: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 7654631494.867957, tolerance: 4945087.043317857
    ##   positive)
    ## /Users/rebeccashapiro/Library/r-miniconda/envs/r-reticulate/lib/python3.6/site-packages/sklearn/linear_model/_coordinate_descent.py:531: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 7522327865.577566, tolerance: 4854611.256816057
    ##   positive)
    ## /Users/rebeccashapiro/Library/r-miniconda/envs/r-reticulate/lib/python3.6/site-packages/sklearn/linear_model/_coordinate_descent.py:531: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 8391190250.43228, tolerance: 5029514.394179694
    ##   positive)
    ## /Users/rebeccashapiro/Library/r-miniconda/envs/r-reticulate/lib/python3.6/site-packages/sklearn/linear_model/_coordinate_descent.py:531: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 8774474309.356646, tolerance: 5232437.209593743
    ##   positive)
    ## /Users/rebeccashapiro/Library/r-miniconda/envs/r-reticulate/lib/python3.6/site-packages/sklearn/linear_model/_coordinate_descent.py:531: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 1229021893.0730724, tolerance: 5154478.066487458
    ##   positive)
    ## /Users/rebeccashapiro/Library/r-miniconda/envs/r-reticulate/lib/python3.6/site-packages/sklearn/linear_model/_coordinate_descent.py:531: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 1138092501.7515926, tolerance: 4945087.043317857
    ##   positive)
    ## /Users/rebeccashapiro/Library/r-miniconda/envs/r-reticulate/lib/python3.6/site-packages/sklearn/linear_model/_coordinate_descent.py:531: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 1265258270.627636, tolerance: 4854611.256816057
    ##   positive)
    ## /Users/rebeccashapiro/Library/r-miniconda/envs/r-reticulate/lib/python3.6/site-packages/sklearn/linear_model/_coordinate_descent.py:531: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 2671453643.624834, tolerance: 5029514.394179694
    ##   positive)
    ## /Users/rebeccashapiro/Library/r-miniconda/envs/r-reticulate/lib/python3.6/site-packages/sklearn/linear_model/_coordinate_descent.py:531: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 3436086150.5433254, tolerance: 5232437.209593743
    ##   positive)
    ## /Users/rebeccashapiro/Library/r-miniconda/envs/r-reticulate/lib/python3.6/site-packages/sklearn/linear_model/_coordinate_descent.py:531: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 967186187.65135, tolerance: 5154478.066487458
    ##   positive)
    ## /Users/rebeccashapiro/Library/r-miniconda/envs/r-reticulate/lib/python3.6/site-packages/sklearn/linear_model/_coordinate_descent.py:531: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 244497960.46545792, tolerance: 4854611.256816057
    ##   positive)
    ## /Users/rebeccashapiro/Library/r-miniconda/envs/r-reticulate/lib/python3.6/site-packages/sklearn/linear_model/_coordinate_descent.py:531: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 12139316.353530884, tolerance: 5232437.209593743
    ##   positive)
    ## /Users/rebeccashapiro/Library/r-miniconda/envs/r-reticulate/lib/python3.6/site-packages/sklearn/linear_model/_coordinate_descent.py:531: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 11374209.765106201, tolerance: 4854611.256816057
    ##   positive)

    ##               0             1             2             3        4
    ## 0  14072.430122  10571.030977  10482.247896  13379.682474  13447.6
    ## 1  12402.868710   8402.072672   8633.309294  13379.682474   7327.2
    ## 2  28292.808436  29784.463934  29130.263843  13379.682474  25740.1
    ## 3  12485.648139  10164.762904   9813.299402  13379.682474  11510.3
    ## 4  15989.309978   9558.323350   8349.360131  13379.682474    125.9
    ## 5  24933.139991  27086.589587  28204.347692  13379.682474  36658.4
    ## 6  13911.063294  14111.518777  13224.673709  13379.682474  12277.9
    ## 7  10344.440351   9284.569464   9874.725323  13379.682474   5374.2
    ## 8  11204.403964  11024.902073  12041.721731  13379.682474   1756.4
    ## 9   4800.740972   3955.811692   4652.419858  13379.682474    506.6

    ## (197, 5)
