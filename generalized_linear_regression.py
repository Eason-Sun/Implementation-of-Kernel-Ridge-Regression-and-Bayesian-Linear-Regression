import numpy as np
from numpy.linalg import inv, norm
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


class GLR:

    def __init__(self, degree, Lambda=0.5):
        self.degree = degree
        self.Lambda = Lambda

    def fit(self, train_data, train_target):
        self.X = train_data.values
        gram_matrix = self.kernel_poly(self.X, self.X)
        n = gram_matrix.shape[0]
        self.a = inv(gram_matrix + self.Lambda * np.eye(n)) @ train_target.values

    def kernel_poly(self, x1, x2):
        return (1 + (x1 @ x2.T)) ** self.degree

    def predict(self, test_data):
        return self.kernel_poly(test_data.values, self.X) @ self.a


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
            clf = GLR(degree=i)
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
    print('\nThe best degree = ', optimal_Lambda)
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

clf = GLR(degree=optimal_degree)
clf.fit(df_train_data, df_train_target)
pred = clf.predict(df_test_data)
print('\nThe MSE for the test set = {:8.6f}'.format(MSE(df_test_target, pred)))

# poly_reg = PolynomialFeatures(degree=optimal_degree)
# X_poly = poly_reg.fit_transform(df_train_data)
# pol_reg = LinearRegression()
# pol_reg.fit(X_poly, df_train_target)
# pred_ = pol_reg.predict(poly_reg.fit_transform(df_test_data))
# print('sklearn: The MSE for the test set = {:8.6f}'.format(MSE(df_test_target, pred_)))

running_times = []
print('\nRun 100 times for each degree:')
for degree in range(1,30,2):
    running_time = []
    for i in range(100):
        start_time = time.time()
        clf = GLR(degree=degree)
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
plt.ylim(0, 0.004)
plt.xlabel('Degree', fontsize=14)
plt.ylabel('Running Time (sec)', fontsize=14)
plt.title('Degree vs Running Time', fontsize=18)
plt.show()