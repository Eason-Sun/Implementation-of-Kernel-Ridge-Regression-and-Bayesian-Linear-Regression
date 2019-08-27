import numpy as np
from numpy.linalg import inv, norm
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
import time
import matplotlib.pyplot as plt

class GBLR:
    def __init__(self, degree, var_noise=1, var_prior=1):
        self.degree = degree
        self.var_noise = var_noise
        self.var_prior = var_prior

    def fit(self, train_data, train_target):
        self.y = train_target.values
        X = train_data.values
        self.phi_X = (PolynomialFeatures(degree=self.degree).fit_transform(X)).T
        sigma = np.eye(self.phi_X.shape[0])
        self.A = (self.var_noise ** (-2)) * (self.phi_X @ np.transpose(self.phi_X)) + inv(sigma)

    def predict(self, test_data):
        phi_X_test = PolynomialFeatures(degree=self.degree).fit_transform(test_data.values)
        return (self.var_noise ** (-2)) * (phi_X_test @ inv(self.A) @ self.phi_X @ self.y)

def MSE(x1, x2):
    return norm(x1 - x2)**2 / x1.shape[0]

def cross_validation(k_fold=10, max_degree=4):
    MSEs = []
    for i in range(1,max_degree+1):
        MSE_per_run = []
        for j in range(k_fold):
            df_validate_data = pd.read_csv('nonlinear-regression-dataset/trainInput' + str(j + 1) + '.csv', header=None)
            df_validate_target = pd.read_csv('nonlinear-regression-dataset/trainTarget' + str(j + 1) + '.csv', header=None)
            df_train_data, df_train_target = merge_train_files(k_fold, skip=j)
            # Create a linear regression classifier
            clf = GBLR(degree=i)
            clf.fit(df_train_data, df_train_target)
            pred = clf.predict(df_validate_data)
            MSE_per_run.append(MSE(df_validate_target, pred))
            # At the end of each k-fold cv, calculate the average MSE
            if j == k_fold - 1:
                avg_MSE = np.mean(np.array(MSE_per_run))
                MSEs.append(avg_MSE)
                print('Degree = {}, MSE = {:8.6f}'.format(i, avg_MSE))
    # Find the index of the minimum MSE so we can get optimal Lambda by multiplying 0.1
    optimal_Lambda = np.argmin(np.array(MSEs)) + 1
    print('The best degree = ', optimal_Lambda)
    return optimal_Lambda, MSEs

# Merge multiple csv file into one data and one label data frames. Optionally, we can exclude certain files
def merge_train_files(num_of_files, skip=None):
    df_train_data = pd.DataFrame()
    df_train_label = pd.DataFrame()
    for k in range(num_of_files):
        if k == skip:
            continue
        data = pd.read_csv('nonlinear-regression-dataset/trainInput' + str(k + 1) + '.csv', header=None)
        df_train_data = df_train_data.append(data, ignore_index=True)
        label = pd.read_csv('nonlinear-regression-dataset/trainTarget' + str(k + 1) + '.csv', header=None)
        df_train_label = df_train_label.append(label, ignore_index=True)
    return df_train_data, df_train_label

optimal_degree, y = cross_validation()

df_train_data, df_train_target = merge_train_files(10)
df_test_data = pd.read_csv('nonlinear-regression-dataset/testInput.csv', header=None)
df_test_target = pd.read_csv('nonlinear-regression-dataset/testTarget.csv', header=None)

clf = GBLR(degree=optimal_degree)
clf.fit(df_train_data, df_train_target)
pred = clf.predict(df_test_data)
print('\nThe MSE for the test set = {:8.6f}'.format(MSE(df_test_target, pred)))

running_times = []
print('\nRun 100 times for each degree:')
for degree in range(1,30,2):
    running_time = []
    for i in range(100):
        start_time = time.time()
        clf = GBLR(degree=degree)
        clf.fit(df_train_data, df_train_target)
        pred = clf.predict(df_test_data)
        running_time.append(time.time() - start_time)
    mean_time = np.array(running_time).mean()
    running_times.append(mean_time)
    print('Average running time of degree {}: {:6.4f}s'.format(degree, mean_time))

# Plot the relationship between degree and MSE
x = [i+1 for i in range(4)]
plt.plot(x, y)
plt.xlabel('Degree', fontsize=14)
plt.ylabel('10-Fold Cross Validation MSE', fontsize=14)
plt.title('Degree vs Mean Squared Error', fontsize=18)
plt.show()

# Plot the relationship between degree and running time
x = [i for i in range(1,30,2)]
plt.plot(x, running_times)
plt.xlabel('Degree', fontsize=14)
plt.ylabel('Running Time (sec)', fontsize=14)
plt.title('Degree vs Running Time', fontsize=18)
plt.show()