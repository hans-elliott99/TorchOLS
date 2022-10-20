import torch
import pandas as pd

##blogpost: https://developer.ibm.com/articles/linear-regression-from-scratch/
##sklearn: https://github.com/scikit-learn/scikit-learn/blob/36958fb24/sklearn/linear_model/_base.py#L529
##stasmodels: https://www.statsmodels.org/dev/_modules/statsmodels/regression/linear_model.html#OLS

class LinearRegression():
    def __init__(self, device=torch.device("cpu")):
        self.torch_device = device

        self.coefficients = None
        self.intercept = None
        self.colnames = None
        self.fit_r2 = None
        self.fit_adjr2 = None

    def fit(self, x, y, r2=True, colnames=None):
        """Fit the linear model to estimate coefficients and intercept.

        x = a pandas dataframe, numpy.array, or torch.tensor containing desired independent variables (predictors)
        y = a pandas dataframe, numpy.array, or torch.tensor containing one dependent variable (target)
        r2 = default True. Calculate the R^2 and Adjusted R^2 of the fitted y's.
        colnames = default None. Optionally provide variable names if not feeding in a pandas dataframe.  
        """
        # prepare x and y for coefficient estimates
        if "pandas" in str(type(x)):
            try:
                self.colnames = {i:col for i,col in enumerate(x.columns)}
            except:
                print("Warning, variable names not found. Is this a pandas series?")
        elif colnames is not None:
            self.colnames = colnames

        # MULTPLIE LINEAR REG   
        if x.shape[1] > 1:
            x = self._transform_x_mlr(x)
            y= self._transform_y(y)

            betas = self._estimate_mlr_coefs(x, y)
            self.intercept, self.coefficients = betas[0], betas[1:]
            x = x[:, 1:] #remove column of 1s so we can use x to find fitted y's for R^2
        
        # SIMPLE LINEAR REG
        else:
            x = self._transform_x_slr(x)
            y = self._transform_y(y)
            self.intercept, self.coefficients = self._estimate_slr_coefs(x, y)
            
        # calculate the fit R^2
        if r2:
            y_pred = self.predict(x)
            print(y.is_cuda, y_pred.is_cuda)
            self.fit_r2, self.fit_adjr2 = self.r2_score(y_true=y, y_pred=y_pred)
    
    def predict(self, x):
        """Makes predictions for all samples in x using the fitted model.
        
        x = a pandas dataframe, numpy.array, or torch.tensor containing desired independent variables (predictors)
        """
        # Makes all predictions in one calculation by broadcasting the coefficients vector.
        x = self._convert_to_tensor(x).to(self.torch_device)

        preds = torch.sum(x * self.coefficients, dim=1) + self.intercept
        return preds
    
    def r2_score(self, y_true, y_pred):
            """Calculate R^2 and Adjusted R^2 scores given true and predicted targets.
                r2 = 1 - (rss/tss)
                adjusted r2 = 1 - (rss / (n - K)) / (tss / (n - 1))
                OR, adjusted r2 = 1 - (1 - r2) * (n - 1)/(n - K)

                rss = sum_{i=0}^{n} (y_i - y_hat_i)^2
                tss = sum_{i=0}^{n} (y_i - y_bar)^2
                (n = sample size, K = number of predictors, y_bar = sample mean)
            """        
            y = self._convert_to_tensor(y_true)
            y_hat = self._convert_to_tensor(y_pred)
            y_bar = torch.mean(y)
            assert y.size(0) == y_hat.size(0)

            k_preds = len(self.coefficients)
            n_samples = y.size(0)

            residual_sum_squares = torch.sum((y - y_hat) ** 2)
            total_sum_squares = torch.sum((y - y_bar) ** 2)

            r2 = 1 - (residual_sum_squares / total_sum_squares)
            adj_r2 = 1 - (1 - r2) * ((n_samples - 1) / (n_samples - k_preds))

            return r2.item(), adj_r2.item() 

    def detach_cuda(self):
        """Detach model from GPU.
        """
        self.torch_device = torch.device("cpu")
        self.coefficients = self.coefficients.to(self.torch_device)
        self.intercept = self.intercept.to(self.torch_device)

    def summary(self):
        """Returns a summary of model coefficients and scores."""
        n_coefs = len(self.coefficients)
        if self.colnames is None:
            self.colnames = {i : f'var{i}' for i in range(n_coefs)}

        out = pd.DataFrame({"" : range(n_coefs+1), "coefficient" : range(n_coefs + 1)})
        out.loc[0] = "intercept", self.intercept.item()
        for i in range(n_coefs):
            out.loc[i+1] = [self.colnames[i], self.coefficients[i].item()]
        if self.fit_r2 is not None:
            add = pd.DataFrame({"":["--", 'R^2', 'Adj. R^2'], "coefficient" : ["score", self.fit_r2, self.fit_adjr2]})
            out = pd.concat([out, add])

        return out


    def _estimate_mlr_coefs(self, x, y):
        """Estimate intercept and coefficients for multiple linear regression.

        Beta_hat = ((X'X)^-1)X'y
        """
        xT = x.T
        inversed = torch.linalg.inv( xT @ x) 
        coefficients = ( inversed @ xT) @ y
        
        return coefficients

    def _estimate_slr_coefs(self, x, y):
        """Estimate intercept and coefficients for simple linear regression.
        Special case may be faster to estimate since no matrix multiplication.

        coef = cov(x, y) / var(x) = sum_{i=0}^{n} (x_i - x_bar)(y_i - y_bar) / sum_{i=0}^{n} (x_i - x_bar)^2
        intercept = y_bar - coef * x_bar
        """
        # ensure that x and y are column vectors
        x = x.view(x.size(0), 1) 
        y = y.view(x.size())
        # calculate means
        x_bar = torch.mean(x)
        y_bar = torch.mean(y)
        # calculate covariance and variance to determine coef and intercept
        cov = torch.sum((x - x_bar) * (y - y_bar))
        var = torch.sum((x - x_bar) ** 2)
        coef = cov / var
        intercept = y_bar - coef*x_bar
        ## for consistency with multiple lin reg, return a 1d tensor for coef instead of a 0d tensor
        return intercept, coef.view(1)


    def _transform_x_mlr(self, x):
        """Transform x for MLR fit. Adds column of ones to estimate intercept."""
        x = self._convert_to_tensor(x)
        x = x.clone()
        x = torch.column_stack((
            torch.ones(x.size(0)),
            x
        ))
        return x.to(self.torch_device)

    def _transform_x_slr(self, x):
        """Transform x for SLR fit. Simply convert to tensor"""
        x = self._convert_to_tensor(x)
        x = x.clone()
        return x.to(self.torch_device)

    def _transform_y(self, y):
        """Transform y for fit."""
        y = self._convert_to_tensor(y)
        y = y.clone()
        return y.to(self.torch_device)
    
    def _convert_to_tensor(self, item):
        """Converts input item to tensor or throws error if not possible."""
        if torch.is_tensor(item):
            return item
        elif "pandas" in str(type(item)):
            return torch.tensor(item.values)
        else:
            try:
                return torch.tensor(item)
            except:
                print("provide a tensor or an object that can be converted to one (numpy array, pandas dataframe)")
                raise NotImplementedError

    # def _predict_iter(self, x):
    #     """implements simple prediction by iterating through the rows of x and predicting for each sample
    #     """
    #     x = self._convert_to_tensor(x)
    #     predictions = []
    #     for ix, row in enumerate(x):
    #         pred = torch.sum(row * self.coefficients) + self.intercept
    #         predictions.append(pred)
    #     return predictions

    # def _r2_iter(self, y_true, y_pred):
    #     """R^2 scores calculated by a loop to check validity of the above method.
    #     """        
    #     y = self._convert_to_tensor(y_true)
    #     y_hat = self._convert_to_tensor(y_pred)
    #     y_bar = torch.mean(y)
    #     k_preds = len(self.coefficients)
    #     n_samples = y.size(0)
    #     residual_sum_squares = 0
    #     total_sum_squares = 0
    #     for i in range(y.size(0)):
    #         residual_sum_squares += (y[i] - y_hat[i])**2
    #         total_sum_squares += (y[i] - y_bar)**2
    #     r2 = 1 - (residual_sum_squares / total_sum_squares)
    #     adj_r2 = 1 - (1 - r2) * ((n_samples - 1) / (n_samples - k_preds))
    #     return r2, adj_r2 

