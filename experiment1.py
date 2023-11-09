import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as mvn

#k-means clustering, GMM is around line 95

def generateData(sigma):
    aMean, aCov = np.array([-1, -1]), sigma * np.array([2, 0.5, 0.5, 1]).reshape((2,2))
    a = np.random.multivariate_normal(mean=aMean, cov=aCov, size=100)
    bMean, bCov = np.array([1, -1]), sigma * np.array([1, -0.5, -0.5, 2]).reshape((2,2))
    b = np.random.multivariate_normal(mean=bMean, cov=bCov, size=100)
    cMean, cCov = np.array([0, 1]), sigma * np.array([1, 0, 0, 2]).reshape((2,2))
    c = np.random.multivariate_normal(mean=cMean, cov=cCov, size=100)
    return a, b, c

def assignments(data, c1, c2, c3):
    ret = []
    for i in range(len(data)):
        d = []
        p = data[i]
        d.append(np.linalg.norm(p - c1))
        d.append(np.linalg.norm(p - c2))
        d.append(np.linalg.norm(p - c3))
        a.append(d.index(min(d)))
    return ret

def updateCenters(data, assign):
    n, d = np.zeros(6).reshape((3, 2)), np.zeros(3)
    for i in range(len(assign)):
        n[assign[i]] += data[i]
        d[assign[i]] += 1
    c1 = (n[0] / d[0])
    c2 = (n[1] / d[1])
    c3 = (n[2] / d[2])
    return c1, c2, c3

def kLoss(data, assign, c1, c2, c3):
    c = [c1, c2, c3]
    tot = 0
    for i in range(len(assign)):
        tot += np.linalg.norm(data[i] - c[assign[i]])
    tot /= len(assign)
    return tot
        
def kMeans(data, c1, c2, c3):
    assignment = assignments(data, c1, c2, c3)
    loss = kLoss(data, assignment, c1, c2, c3)
    n1, n2, n3 = updateCenters(data, assignment)
    if loss - kLoss(data, assignment, n1, n2, n3) == 0:
        return c1, c2, c3, loss
    return kMeans(data, n1, n2, n3)

def getAccuracyKM(clusters, assign):
    m = np.zeros(9).reshape(3, 3)
    n = len(assign)
    for it in range(n):
        i = assign[it]
        j = clusters[it]
        m[i, j] += 1
    a1,a2,a3 = (m[0, 0] + m[1, 1] + m[2, 2]) / n, (m[0, 1] + m[1, 2] + m[2, 0]) / n, (m[0, 2] + m[1, 0] + m[2, 1]) / n
    return max(a1, a2, a3)

def plotKmeans(data, assign, c1, c2, c3):
    plt.scatter(data[:,0], data[:,1], c=assign, alpha=.5)
    for i in range (1,4):
        plt.scatter(c1[0], c1[i], s=100, c="red", marker="*")
    plt.show()

np.random.seed(5)
s = [0.5, 1, 2, 4, 8]
c1,c2,c3 = [0, 2], [-1, 0], [1, 0]
loss,accuracy = [],[]
clusters = np.array([[0] * 100, [1] * 100, [2] * 100]).reshape(300)
for i in range(len(s)):
    a, b, c = generateData(s[i])
    data = np.array([a, b, c]).reshape(300, 2)
    x1, x2, x3, l = kMeans(data, c1, c2, c3)
    loss.append(l)
    accuracy.append(getAccuracyKM(clusters, assignments(data, x1, x2, x3)))

# plot
plt.plot(s, loss)
plt.ylabel("loss")
plt.xlabel("sigma")
plt.title("k-Means Loss vs Sigma")
plt.show()

plt.plot(s, accuracy)
plt.ylabel("accuracy")
plt.xlabel("sigma")
plt.title("k-Means Accuracy vs Sigma")
plt.show()

# GMM Maximization

def init(sigma):
    MEMean, MECov = np.zeros(2), np.array([1, 0, 0, 1]). reshape((2, 2))
    c1Mean,c1Cov = np.array([-1, -1]) + np.random.multivariate_normal(mean=MEMean, cov=MECov), sigma * np.array([2, 0.5, 0.5, 1]).reshape((2,2))
    c2Mean,c2Cov = np.array([1, -1]) + np.random.multivariate_normal(mean=MEMean, cov=MECov), sigma * np.array([1, -0.5, -0.5, 2]).reshape((2,2))
    c3Mean,c3Cov = np.array([0, 1]) + np.random.multivariate_normal(mean=MEMean, cov=MECov), sigma * np.array([1, 0, 0, 2]).reshape((2,2))
    means = [c1Mean, c2Mean, c3Mean]
    covs = [c1Cov, c2Cov, c3Cov]
    return means, covs

def expectations(data, phi, mu, sig):
    w = np.zeros(len(data) * len(phi)).reshape((len(data), len(phi)))
    for i in range(len(data)):
        d = 0
        for j in range(len(phi)):
            d += mvn.pdf(x=data[i], mean=mu[j], cov=sig[j]) * phi[j]
        for j in range(len(phi)):
            pz = phi[j]
            px = mvn.pdf(x=data[i], mean=mu[j], cov=sig[j])
            w[i, j] = (px * pz) / d
    return w

def maxParams(data, weights):
    phi = sum(weights) / len(data)
    mu = (data.T.dot(weights) / sum(weights)).T
    d = phi * len(data)
    sig = []
    for j in range(len(weights[0])):
        s = np.zeros(4).reshape((2,2))
        for i in range(len(data)):
            w = weights[i, j]
            v = data[i] - mu[j]
            s += w * np.outer(v, v)
        sig.append(s / d[j])
    return phi, mu, sig

def getLoss(weights, labels):
    res = 0
    for i in range(len(labels)):
        res += np.log(weights[i][labels[i]])
    return -res

def getLabels(weights):
    res = []
    for i in range(len(weights)):
        res.append(weights[i].argmax())
    return(res)

def getAccuracyGMM(source, labels):
    m = np.zeros(9).reshape(3, 3)
    n = len(labels)
    for it in range(n):
        i = labels[it]
        j = source[it]
        m[i, j] += 1
    a1 = (m[0, 0] + m[1, 1] + m[2, 2]) / n
    a2 = (m[0, 1] + m[1, 2] + m[2, 0]) / n
    a3 = (m[0, 2] + m[1, 0] + m[2, 1]) / n
    return max(a1, a2, a3)

def plotGmm(data, labels, means):
    plt.scatter(data[:,0], data[:,1], c=labels, alpha=.5)
    for i in range(0,3):
        plt.scatter(means[i][0], means[i][1], s=100, c="red", marker="*")
    plt.show()

def gmm(data, phi, mu, sig, epsilon):
    w = expectations(data, phi, mu, sig)
    loss = getLoss(w, getLabels(w))
    p, m, s = maxParams(data, w)
    w2 = expectations(data, p, m, s)
    loss2 = getLoss(w2, getLabels(w2))
    if abs(loss - loss2) < epsilon:
        return p, m, s, loss2
    else:
        return gmm(data, p, m, s, epsilon=epsilon)

np.random.seed(5)
s = [0.5, 1, 2, 4, 8]
p = np.array([1/3]*3).reshape(3)
loss = []
accuracy = []
source = np.array([[0] * 100, [1] * 100, [2] * 100]).reshape(300)
for i in range(len(s)):
    a, b, c = generateData(s[i])
    data = np.array([a, b, c]).reshape(300, 2)
    mu, sig = init(s[i])
    p2, m, s2, l = gmm(data, p, mu, sig, .1)
    loss.append(l)
    accuracy.append(getAccuracyGMM(source, getLabels(expectations(data, p2, m, s2))))

plt.plot(s, loss)
plt.ylabel("loss")
plt.xlabel("sigma")
plt.title("GMM Loss vs Sigma")
plt.show()

plt.plot(s, accuracy)
plt.ylabel("accuracy")
plt.xlabel("sigma")
plt.title("GMM Accuracy vs Sigma")
plt.show()