import pandas as pd
from numpy import *


# 选择不等于i的j
def random_select_j(i, m):
    j = i
    while j == i:
        j = int(random.uniform(0, m))
    return j


def clipAlpha(aj, H, L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj


# 计算kernel或者将数据转化到高维空间
def kernel_trans(X, A, kTup):
    m, n = shape(X)
    K = mat(zeros((m, 1)))
    if kTup[0] == 'lin':  # linear kernel
        K = X * A.T
    elif kTup[0] == 'rbf':
        for j in range(m):
            deltaRow = X[j, :] - A
            K[j] = deltaRow * deltaRow.T
        K = exp(K / (-1 * kTup[1] ** 2))
    else:
        print('error')
    return K


class Optimizer:
    def __init__(self, dataMatIn, classLabels, C, tolerance, kTup):
        """
        :param dataMatIn: 数据集
        :param classLabels: 类别标签
        :param C: 常数C
        :param tolerance: 容错率
        :param kTup: 包含核函数信息的元组
        """
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = tolerance
        self.m = shape(dataMatIn)[0]
        self.alphas = mat(zeros((self.m, 1)))
        self.b = 0
        self.eCache = mat(zeros((self.m, 2)))
        self.K = mat(zeros((self.m, self.m)))
        for i in range(self.m):
            self.K[:, i] = kernel_trans(self.X, self.X[i, :], kTup)


def get_Ek(opt, k):
    fXk = float(multiply(opt.alphas, opt.labelMat).T * opt.K[:, k] + opt.b)
    Ek = fXk - float(opt.labelMat[k])
    return Ek


def get_J(i, opt, Ei):
    maxK = -1
    maxDeltaE = 0
    Ej = 0
    opt.eCache[i] = [1, Ei]
    validEcacheList = nonzero(opt.eCache[:, 0].A)[0]
    if (len(validEcacheList)) > 1:
        for k in validEcacheList:
            if k == i:
                continue
            Ek = get_Ek(opt, k)
            deltaE = abs(Ei - Ek)
            if deltaE > maxDeltaE:
                maxK = k
                maxDeltaE = deltaE
                Ej = Ek
        return maxK, Ej
    else:
        j = random_select_j(i, opt.m)
        Ej = get_Ek(opt, j)
    return j, Ej


def update_Ek(opt, k):
    Ek = get_Ek(opt, k)
    opt.eCache[k] = [1, Ek]


def innerL(i, opt):
    Ei = get_Ek(opt, i)
    if ((opt.labelMat[i] * Ei < -opt.tol) and (opt.alphas[i] < opt.C)) or (
            (opt.labelMat[i] * Ei > opt.tol) and (opt.alphas[i] > 0)):
        j, Ej = get_J(i, opt, Ei)
        alpha_i_old = opt.alphas[i].copy()
        alpha_j_old = opt.alphas[j].copy()
        if opt.labelMat[i] != opt.labelMat[j]:
            L = max(0, opt.alphas[j] - opt.alphas[i])
            H = min(opt.C, opt.C + opt.alphas[j] - opt.alphas[i])
        else:
            L = max(0, opt.alphas[j] + opt.alphas[i] - opt.C)
            H = min(opt.C, opt.alphas[j] + opt.alphas[i])
        if L == H:
            # print("L==H")
            return 0
        eta = 2.0 * opt.K[i, j] - opt.K[i, i] - opt.K[j, j]
        if eta >= 0:
            # print("eta>=0")
            return 0
        opt.alphas[j] -= opt.labelMat[j] * (Ei - Ej) / eta
        opt.alphas[j] = clipAlpha(opt.alphas[j], H, L)
        update_Ek(opt, j)
        if abs(opt.alphas[j] - alpha_j_old) < 0.00001:
            # print("j not moving enough")
            return 0
        opt.alphas[i] += opt.labelMat[j] * opt.labelMat[i] * (
                alpha_j_old - opt.alphas[j])
        update_Ek(opt, i)
        b1 = opt.b - Ei - opt.labelMat[i] * (opt.alphas[i] - alpha_i_old) * opt.K[i, i] - opt.labelMat[j] * (
                opt.alphas[j] - alpha_j_old) * opt.K[i, j]
        b2 = opt.b - Ej - opt.labelMat[i] * (opt.alphas[i] - alpha_i_old) * opt.K[i, j] - opt.labelMat[j] * (
                opt.alphas[j] - alpha_j_old) * opt.K[j, j]
        if (0 < opt.alphas[i]) and (opt.C > opt.alphas[i]):
            opt.b = b1
        elif (0 < opt.alphas[j]) and (opt.C > opt.alphas[j]):
            opt.b = b2
        else:
            opt.b = (b1 + b2) / 2.0
        return 1
    else:
        return 0


def smo_Platt(dataMatIn, classLabels, C, tolerance, maxIter, kTup=('lin', 0)):
    opt = Optimizer(mat(dataMatIn), mat(classLabels).transpose(), C, tolerance, kTup)
    epoch = 0
    entireSet = True
    alphaPairsChanged = 0
    while (epoch < maxIter) and ((alphaPairsChanged > 0) or entireSet):
        alphaPairsChanged = 0
        if entireSet:
            for i in range(opt.m):
                alphaPairsChanged += innerL(i, opt)
                # print("fullSet, iter: %d i:%d, pairs changed %d" % (epoch, i, alphaPairsChanged))
            epoch += 1
        else:
            nonBoundIs = nonzero((opt.alphas.A > 0) * (opt.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i, opt)
                # print("non-bound, iter: %d i:%d, pairs changed %d" % (epoch, i, alphaPairsChanged))
            epoch += 1
        if entireSet:
            entireSet = False
        elif alphaPairsChanged == 0:
            entireSet = True
        print("iteration number: %d" % epoch)
    return opt.b, opt.alphas


def load_dataset():
    data = pd.read_csv('./my_data.csv')
    length = data.shape[0]
    data_train = data[:int(length * 0.8)]
    data_test = data[int(length * 0.8):]
    data_train_x = data_train.drop(['Survived'], axis=1).values
    data_test_x = data_test.drop(['Survived'], axis=1).values
    data_train_y = data_train['Survived'].values
    data_test_y = data_test['Survived'].values
    for i in range(len(data_train_y)):
        if data_train_y[i] == 0:
            data_train_y[i] = -1
    for i in range(len(data_test_y)):
        if data_test_y[i] == 0:
            data_test_y[i] = -1
    return data_train_x, data_train_y, data_test_x, data_test_y


def titanic():
    data_train_x, data_train_y, data_test_x, data_test_y = load_dataset()
    k1 = 0.1
    # C = 200 , tolerance = 0.0001 , max_run = 10000
    b, alphas = smo_Platt(data_train_x, data_train_y, 200, 0.0001, 10000, ('rbf', k1))
    dataMat = mat(data_train_x)
    labelMat = mat(data_train_y).transpose()
    # svInd: index of support vector, sVs: support vectors, labelSV: support vector labels
    svInd = nonzero(alphas.A > 0)[0]
    sVs = dataMat[svInd]
    labelSV = labelMat[svInd]
    print("there are %d Support Vectors" % shape(sVs)[0])
    m, n = shape(data_train_x)
    errorCount = 0
    for i in range(m):
        kernelEval = kernel_trans(sVs, dataMat[i, :], ('rbf', k1))
        predict = kernelEval.T * multiply(labelSV, alphas[svInd]) + b
        if sign(predict) != sign(data_train_y[i]):
            errorCount += 1
    print("the number of errors is %d" % errorCount)
    print("the training accuracy rate is: %f" % (1 - float(errorCount) / m))

    # test data running
    testErrorCount = 0
    dataMat = mat(data_test_x)
    m, n = shape(data_test_x)
    for i in range(m):
        kernelEval = kernel_trans(sVs, dataMat[i, :], ('rbf', k1))
        predict = kernelEval.T * multiply(labelSV, alphas[svInd]) + b
        if sign(predict) != sign(data_test_y[i]):
            testErrorCount += 1
    print("the test accuracy rate is %.3f" % (1 - float(testErrorCount) / m))


if __name__ == "__main__":
    titanic()
