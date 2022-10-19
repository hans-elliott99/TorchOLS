import torch
import pandas as pd

##blogpost: https://developer.ibm.com/articles/linear-regression-from-scratch/
##sklearn: https://github.com/scikit-learn/scikit-learn/blob/36958fb24/sklearn/linear_model/_base.py#L529
##stasmodels: https://www.statsmodels.org/dev/_modules/statsmodels/regression/linear_model.html#OLS

class MultipleLinearRegression():
    def __init__(self):
        self.coefficients = None
        self.intercept = None
        self.colnames = None
        self.fit_r2 = None

    def fit(self, x, y, r2=True):
        # prepare x and y for coefficient estimates
        if "pandas" in str(type(x)):
            self.colnames = {i:col for i,col in enumerate(x.columns)}

        x = self._transform_x(x)
        y = self._transform_y(y)

        betas = self._estimate_coefs(x, y)
        # intercept
        self.intercept = betas[0]
        # coefficients
        self.coefficients = betas[1:]

        # fit R^2
        if r2:
            y_pred = self.predict(x[:, 1:])
            self.fit_r2, self.fit_adjr2 = self.r2_score(y_true=y, y_pred=y_pred)
    
    def predict(self, x):
        """makes predictions for all samples in x in one calculation by broadcasting the coefficients vector
        """
        x = self._convert_to_tensor(x)
        preds = torch.sum(x * self.coefficients, dim=1) + self.intercept ##torch.dot?
        return preds

    def _predict_iter(self, x):
        """implements simple prediction by iterating through the rows of x and predicting for each sample
        """
        x = self._convert_to_tensor(x)
        predictions = []
        for ix, row in enumerate(x):
            pred = torch.sum(row * self.coefficients) + self.intercept
            predictions.append(pred)
        return predictions
    

    def r2_score(self, y_true, y_pred):
        """
            r2 = 1 - (rss/tss)
            rss = sum_{i=0}^{n} (y_i - y_hat)^2
            tss = sum_{i=0}^{n} (y_i - y_bar)^2
        """
        n_coefs = len(self.coefficients)
        n_samples = y_true.size(0)
        
        y_true = self._convert_to_tensor(y_true)
        y_hat = self._convert_to_tensor(y_pred)
        y_bar = torch.mean(y_true)
        
        residual_sum_squares = 0
        total_sum_squares = 0

        for i in range(len(y_true)):
            residual_sum_squares += (y_true[i] - y_pred[i])**2
            total_sum_squares += (y_true[i] - y_bar)**2

        r2 = 1 - (residual_sum_squares / total_sum_squares)
        adj_r2 = 1 - (residual_sum_squares / (n_samples-n_coefs) / (total_sum_squares / n_samples - 1)

        return r2, adj_r2 

    def summary(self):
        n_coefs = len(self.coefficients)
        n_samples = y_true.size(0)
        if self.colnames is None:
            self.colnames = {i : f'var{i}' for i in range(n_coefs)}

        out = pd.DataFrame({"" : range(n_coefs+1), "coefficient" : range(n_coefs + 1)})
        out.loc[0] = "intercept", self.intercept.item()
        for i in range(n_coefs):
            out.loc[i+1] = [self.colnames[i], self.coefficients[i].item()]
        if self.fit_r2 is not None:
            add = pd.DataFrame({"":["--", 'Fit R^2'], "coefficient" : ["value", self.fit_r2.item()]})
            out = pd.concat([out, add])

        return out

    def _estimate_coefs(self, x, y):
        """Estimate intercept and coefficients
        """
        xT = x.T
        inversed = torch.linalg.inv( xT @ x) 
        coefficients = ( inversed @ xT) @ y
        
        return coefficients

    def _transform_x(self, x):
        x = self._convert_to_tensor(x)
        x = x.clone()
        x = torch.column_stack((
            torch.ones(x.size(0)),
            x
        ))
        return x

    def _transform_y(self, y):
        y = self._convert_to_tensor(y)
        y = y.clone()
        return y
    
    def _convert_to_tensor(self, item):
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


