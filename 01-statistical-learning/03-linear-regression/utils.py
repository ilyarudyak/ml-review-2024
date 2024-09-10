import numpy as np
import pandas as pd

# Lets write a function to print the summary of the model
def get_stats(model, 
            summary=False, 
            r_squared=False,
            rse=False, 
            f_statistic=False):
    """
    If `summary` is True return a dataframe with the summary of the model:
        - Index should be the variable names.
        - Column 'coef' should be the coefficients of the model.
        - Column 'std_err' should be the standard errors of the coefficients.
        - Column 't' should be the t-statistics of the coefficients.
        - Column 'p' should be the p-values of the coefficients.
    If `r_squared` is True, return the R-squared value of the model.
    If `F_statistic` is True, return the F-statistic of the model.
    """
    if summary:
        summary = pd.DataFrame({'coef': np.round(model.params, 4),
                                'std_err': np.round(model.bse, 4),
                                't': np.round(model.tvalues, 4),
                                'p': np.round(model.pvalues, 4)}, 
                                index=model.params.index)
        return summary
    elif r_squared:
        return np.round(model.rsquared, 4)
    elif rse:
        return np.round(np.sqrt(model.mse_resid), 4)
    elif f_statistic:
        return np.round(model.fvalue, 4)
    else:
        return None