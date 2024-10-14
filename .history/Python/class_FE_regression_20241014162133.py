# Implement fixed effects regression with OLS estimation

# Packages
import numpy as np
import timeit as timeit
import scipy.stats


# Commented out code
# np.set_printoptions(linewidth=250)  # Increase the line width



## Classes


### Simulate specified panel model
class panel_simulation:
    def __init__(self, N, T, beta, reps, min_var = 0, max_var = 1, range_theta = [0,10], shrink_factor = 0):
        '''
        If shrink_factor=0 covar matrix will be diagonal matrix 
        '''
        self.N = N # nr. of cross-sectional units
        self.T = T # nr. of time periods
        self.beta = beta # coefficients
        self.K = len(beta) # nr. of variables
        self.generate_covar_matrix(min_var, max_var, shrink_factor) # Create N*N varcov matrix
        self.reps = reps
        self.range_theta = range_theta

    def generate_theta(self, min = 0, max = 10):
        # FE model: Y = THETA_i + XB + e_i
        # Generate fixed effects for each cross-sectional unit i, for each rep
        ## N*1 Vector of fixed effects
        # 
        self.theta = np.random.uniform(min, max, (self.reps, self.N, 1))
        
 
    def generate_X(self, mean = 0, sd = 1):
        # Generate X values
        ## Reps*N*T*K (they are independent draws, so can draw them all at once)
        # self.X = np.random.uniform(0, 5, [self.reps, self.N, self.T, self.K])
        self.X = np.random.multivariate_normal([mean]*self.K, np.eye(self.K)*sd, [self.reps, self.N, self.T])


    def generate_covar_matrix(self, min_var, max_var, shrink_factor = 0):
        
        # Generate diagonal of the covariance matrix
        variances = np.random.uniform(min_var, max_var, self.N)
        
        if shrink_factor == 0:
            self.covar_matrix = np.diag(variances)
        
        # Cross-sectional correlation: covariance matrix with non-zero values on off-diagonals
        else:
            # Maximum of the covariance is max_variance/shrink_factor
            max_covar = max_var / shrink_factor
            covariances = np.random.uniform(-max_covar, max_covar, (self.N, self.N))
            # Creates symmetrical matrix using the generated variances and covariances
            ## covariance matrix = diagonal matrix + lower triangular matrix + upper triangular matrix
            covar_matrix = np.diag(variances) + np.tril(covariances, -1) + np.triu(covariances.T, 1)
            self.covar_matrix = covar_matrix

    def generate_errors(self):
        # Generate Reps*NT errors with mean 0
        self.errors = np.random.multivariate_normal(np.zeros(self.N), self.covar_matrix, (self.reps, self.T)).transpose((0, 2, 1))

    def simulate_Y(self):
        # Reps*NT matrix of theta
        self.generate_theta(self.range_theta[0], self.range_theta[1])
        # Reps*NT matrix of errors
        self.generate_errors()
        # RepsNT matrix of X
        self.generate_X()
        
        self.Y = self.theta + self.X @ self.beta + self.errors
        
        # theta and X go together, since real data would have theta included in the observations
        self.X_observed = self.X + self.theta.reshape((self.reps, self.N, 1, 1)) # use np broadcasting over X by reshaping last two axes of theta to 1
        
        return self

### Estimate coefficients, calculate t-values & rejection rates for data simulated with panel_simulation class

class panel_estimation():

    def __init__(self, X, Y, N, T, method = "OLS", FE = True):
        self.X = X
        self.Y_hat = None
        self.Y = Y
        self.N = N
        self.T = T
        self.K = self.X.shape[-1] # K = len(beta), which is the last dimension of X
        self.reps = self.X.shape[0] # reps is first dimension of X
        self.method = method
        self.FE = FE
        self.M = self.demean(T) # Demeaning matrix

    def FE_OLS(self):
        '''
        Performs FE OLS estimation on ALL repetitions present in X (first dimension of X, or Y)
        So: includes demeaning transformation
        ## Old implementation: Looping through N to estimate beta is slower by factor of 1.9
        ## This scales with reps: reps*1.9 slower
        
        large_Q and small_q correspond to the Q and q matrix in equation 26.13, 26.21 (Pesaran, p. 640)
        '''
        M_dot_Y = self.Y @ self.M
        self.M_dot_Y = M_dot_Y # Will be used later to calculate the covariance matrix
        small_q = np.einsum('hntr,hnt->hr', self.X, M_dot_Y)

        M_dot_X = self.M @ self.X
        self.M_dot_X = M_dot_X # Will be used later to calculate the covariance matrix
        large_Q = np.einsum('hntk,hntl->hkl', self.X, M_dot_X)
        large_Q_inv = np.linalg.inv(large_Q)
        self.Q_inv = large_Q_inv

        coefficients = np.einsum('ijk, ij ->ik', large_Q_inv, small_q)

        return coefficients
    
    def weighted_FE_OLS(self):
        '''
        Estimates coefficients for X and Y where datapoints are missing
        '''

        # Weighting matrix is 0 where there are missing values, ignoring them in estimation
        weighting_matrix_X = np.where(~np.isnan(self.X), 1, 0)
        self.weighting_matrix_X = weighting_matrix_X

        # We cannot straightforwardly demean the X matrix if values are missing.
        ## So, fill in the NAs with corresponding means, then demean
        ## This is the same as ignoring the datapoints when calculating the mean for each N, since a value at the mean will not affect the demeaning operation, essentially ignoring them
        # Step 1: Compute the mean across the time dimension (ignoring np.nan)
        mean_X = np.nanmean(self.X, axis=2, keepdims=True)  # Keep dims to match original shape

        # Step 2: Fill np.nan values with the mean of each time series
        X_filled = np.where(np.isnan(self.X), mean_X, self.X)
        # Hadamard product sets missing observations to 0
        X_weighted = X_filled * weighting_matrix_X


        M_dot_Y = self.Y @ self.M
        self.M_dot_Y = M_dot_Y # Will be used later to calculate the covariance matrix
        small_q = np.einsum('hntr,hnt->hr', X_weighted, M_dot_Y)

        M_dot_X = self.M @ X_weighted
        self.M_dot_X = M_dot_X
        large_Q = np.einsum('hntk,hntl->hkl', X_weighted, M_dot_X)
        large_Q_inv = np.linalg.inv(large_Q)
        self.Q_inv = large_Q_inv

        coefficients = np.einsum('ijk, ij ->ik', large_Q_inv, small_q)

        return coefficients


    def OLS(self):
        '''
        Performs regular OLS estimation on ALL repetitions present in X (first dimension of X, or Y)
        So: does NOT include demeaning transformation

        large_Q and small_q correspond to the Q and q matrix in equation 26.13, 26.21 (Pesaran, p. 640)
        '''

        small_q = np.einsum('hntr,hnt->hr', self.X, self.Y)
        large_Q = np.einsum('hntk,hntl->hkl', self.X, self.X)
        large_Q_inv = np.linalg.inv(large_Q)
        self.Q_inv = large_Q_inv

        coefficients = np.einsum('ijk, ij ->ik', large_Q_inv, small_q)

        return coefficients


    def estimate_coefs(self):
        if self.method == "OLS":
            self.coefficients = self.OLS()
            
        if self.method == "FE_OLS":
            self.coefficients = self.FE_OLS()

        if self.method == "weighted_FE_OLS":
            self.coefficients = self.weighted_FE_OLS()
        
        return self
    
    def calc_t_values(self):
            # Tensor implementation of Type I errors
            ## First calculates the t-values for all the coefficients
            ## Then calculates whether the coefficient rejects H0 of beta=0
            ### Based on Pesaran p.642,643

            # Compute residuals
            resids_all = self.M_dot_Y - np.einsum('ahij, aj -> ahi', self.M_dot_X, self.coefficients)
            self.residuals = resids_all
            
            # Compute variance for each N
            ## Need to adjust for weighted version since T is not equal for all N
            if self.method == "weighted_FE_OLS":
                # Need to adjust T for proper variance calculation
                ## Since T is no longer the same when observations are randomly missing
                t_per_n = np.sum(self.weighting_matrix_X, axis = 2)
                self.t_per_n = t_per_n # use for degrees of freedom calculation later
                mean_t_per_n = t_per_n.mean(axis = 2)
                # Get variance by dividing residuals by the corresponding number of observations
                variances_all = np.sum(resids_all ** 2, axis = 2) / mean_t_per_n  # Shape: reps x N x T
                
            else:
                variances_all = np.sum(resids_all ** 2, axis = 2) / self.T


            # Construct Gamma matrices, RepsxNxTxT
            ## Each TxT matrix has the corresponding variance on the diagonal
            identity_matrices = np.tile(np.eye(self.T), (self.reps, self.N, 1, 1))
            Gamma = identity_matrices * variances_all[:, :, np.newaxis, np.newaxis]

            V_FENT = np.einsum('anti,antj,antk->aik', self.M_dot_X, Gamma, self.M_dot_X)
            Omega = np.einsum('aij,ajk,akl->ail', self.Q_inv, V_FENT, self.Q_inv)

            # Compute the square root of the diagonal elements of Omega
            diagonal_sqrt_Omega = np.sqrt(np.diagonal(Omega, axis1=1, axis2=2))

            # Divide the coefficients by the square root of the diagonal elements
            t_values = self.coefficients / diagonal_sqrt_Omega

            self.t_values = t_values

    def rejection_rate(self):
            # Calculate degrees of freedom and critical value
            if self.method == "weighted_FE_OLS":
                # We have calculated the nr of valid observations for X 
                df = np.sum(self.t_per_n) - self.K
            else:
                df = self.N * self.T - self.K  # degrees of freedom
                
            critical_value = scipy.stats.t.ppf(1 - 0.05 / 2, df) #two-sided

            # Calculate absolute t-values
            abs_t_vals = np.abs(self.t_values)

            # Count the number of rejections for beta1 and beta2
            nr_of_rejections_beta = []

            for k in range(self.K):
                nr_of_rejections_beta.append(np.sum(abs_t_vals[:, k] > critical_value))

            self.proportion_H0_rejected = np.array(nr_of_rejections_beta) / self.reps

    @staticmethod
    def demean(T):
        M_T = np.eye(T) - (1/T) * np.ones((T, T))
        return M_T
    
    

# Implementation of the simulation study of Inoue (2008), slow (unoptimized)
class GMM_replication():

    def simulate(self, S, T, N_bar, var_ds, var_xst, var_zi, var_vzst):
        '''
        Gamma = individual regressor coefficient
        Beta = group regressor coefficient
        S = groups
        T = time periods
        N_bar = average group size
        var_ds = variance of group FE
        var_xst = variance of group regressor
        var_zi = variance of individual regressor + individual error
        var_vzst = variance of individual regressor
        '''
        
        # Convert to SD
        var_ds = np.sqrt(var_ds)
        var_xst = np.sqrt(var_xst)
        var_zi = np.sqrt(var_zi)
        var_vzst = np.sqrt(var_vzst)
        
        # Determining group sizes
        ## Pi matrix
        # pis = np.random.uniform(0.3, 0.6, size = S*T)
        pis = np.random.uniform(size = S*T)

        Pi = np.diag(pis / pis.sum())

        group_sizes = np.ceil(np.diag(Pi) * N_bar * S * T).astype(int)

        # group_sizes = (np.ones(S*T)*100).astype(int)

        N = int(group_sizes.sum())

        
        var_ai_ei = 1 - var_ds # individual FE + individual error
        var_ezi = var_zi - var_vzst

        # Generate group FE
        d_s = np.random.normal(0, var_ds, S)

        # Generate regressors x, v, iid normal
        x_st = np.random.normal(0, var_xst, S*T)
        v_zst = np.random.normal(0, var_vzst, S*T)
        
        # Simulate y_i = S*T
        ## Since beta, gamma = 0, the only variation in y_i comes from a_i, d_s, e_i
        y_bar_s = []
        z_bar_st = []

        for s in range(S):
            for t in range(T):
                # y_bar
                d = d_s[s] # FE for group s
                n = group_sizes[s+t]
                ai_ei = np.random.normal(0, var_ai_ei, n)
                y_bar_s.append(np.mean(ai_ei)+d)
                
                # z_i
                v = v_zst[s+t]
                z = np.random.normal(0, var_ezi, n)
                # z_i.append(z)
                z_bar_st.append(np.mean(z) + v)
                
        self.X = np.vstack((x_st, np.array(z_bar_st))).T
        self.Y = y_bar_s
        self.group_sizes = group_sizes


    def estimate(self, X, Y, S, T, group_size_vec):
        M = np.kron(np.eye(S), demean(T))
        Ts = np.arange(T-1, S*T, T) # Indices of last time-period, to use in T*N vectors
        
        X1 = M @ X
        X2 = np.delete(X1, Ts, axis = 0)
        
        group_size_mat = np.diag(group_size_vec)
        Omega = M@group_size_mat@M
        Omega = np.delete(Omega, Ts, axis = 0)
        Omega = np.delete(Omega, Ts, axis = 1)
        
        Omega_inv = np.linalg.inv(Omega)
        
        Y1 = M @ Y
        Y2 = np.delete(Y1, Ts)
        
        coefs = np.linalg.inv(X2.T @ Omega_inv @ X2) @ X2.T @ Omega_inv @ Y2 

        return coefs.tolist()
        
            
    def simulate_estimate(self, S, T, N_bar, var_ds, var_xst, var_zi, var_vzst, reps):
        OLS_coefs, GMM_coefs = [], []
        
        for _ in range(reps):
        
            self.simulate(S, T, N_bar, var_ds, var_xst, var_zi, var_vzst)

            # GMM estimator reduces to OLS with identity weighting matrix
            OLS_estimate = self.estimate(self.X, self.Y, S, T, np.ones(S*T))
            OLS_coefs.append(OLS_estimate)
            # Use optimal weighting matrix
            group_sizes_normalized = self.group_sizes / np.sum(self.group_sizes)
            GMM_estimate = self.estimate(self.X, self.Y, S, T, group_sizes_normalized)
            GMM_coefs.append(GMM_estimate)
            
            
        return [np.array(OLS_coefs), np.array(GMM_coefs)]
        


## Functions
    

def demean(T):
    M_T = np.eye(T) - (1/T) * np.ones((T, T))
    return M_T


# Basic implementation of GMM
## Estimates coefs using GMM for ONE sample
def GMM_basic(X, Y, N, T, K, group_size_vec=None):
    X = X[0,:,:,:].reshape((N*T, K))
    Y = Y[0,:,:].reshape((N*T, 1))

    S = N
    M = np.kron(np.eye(S), demean(T))
    Ts = np.arange(0, N*T, T)

    X1 = M @ X
    X2 = np.delete(X1, Ts, axis = 0)
    if not group_size_vec:
        Omega = M@(np.eye(N*T)*10)@M
        Omega = np.delete(Omega, Ts, axis = 0)
        Omega = np.delete(Omega, Ts, axis = 1)
    else:
        Omega = np.diag(group_size_vec/sum(group_size_vec))

    Omega_inv = np.linalg.inv(Omega)

    Y1 = M @ Y
    Y2 = np.delete(Y1, Ts)

    coefficients = np.linalg.inv(X2.T @ Omega_inv @ X2) @ X2.T @ Omega_inv @ Y2 
    return coefficients

