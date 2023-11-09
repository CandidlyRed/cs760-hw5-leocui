import numpy as np
import matplotlib.pyplot as plt

data2,data1000 = np.loadtxt("./hw5Data/data2D.csv", delimiter=','), np.loadtxt("./hw5Data/data1000D.csv", delimiter=',')

# PCA, DRO and DRLV begin around line 100

def buggyPCA(X, k):
    u, s, vt = np.linalg.svd(X)
    V = vt[range(k)].reshape((X.shape[1], k))
    Z = np.matmul(X, V)
    tX = np.matmul(Z, V.T)
    return V, Z, tX

def normalize(X):
    Xn = X.copy()
    mu = np.mean(Xn, axis=0)
    sd = np.std(Xn, axis=0)
    ret = ((Xn - mu) / sd)
    return ret, mu, sd

def normalizePCA(X, k):
    Xn, mu, sd = normalize(X)
    V, Z, tX = buggyPCA(Xn, k)
    ret = (tX * sd) + mu
    return V, Z, ret

def demean(X):
    Xd = X.copy()
    mu = np.mean(Xd, axis=0)
    ret = (Xd - mu)
    return ret, mu

def demeandsPCA(X, k):
    Xd, mu = demean(X)
    V, Z, tX = buggyPCA(Xd, k)
    ret = (tX + mu)
    return V, Z, ret

def reconstructionE(X, tX):
    ret = round((np.linalg.norm(X - tX)**2) / len(X), 6)
    return ret

def plotPCA(X, tX):
    plt.scatter(X[:, 0], X[:, 1], c="red", marker='x')
    plt.scatter(tX[:, 0], tX[:, 1], c="blue", marker='+')

bV, bZ, bX = buggyPCA(data2, 1)
print(reconstructionE(data2, bX))
plotPCA(data2, bX)
plt.title("Buggy PCA")
plt.show()

dV, dZ, dX = demeandsPCA(data2, 1)
print(reconstructionE(data2, dX))
plotPCA(data2, dX)
plt.title("Demeaned PCA")
plt.show()

nv, nz, nx = normalizePCA(data2, 1)
print(reconstructionE(data2, nx))
plotPCA(data2, nx)
plt.title("Normalized PCA")
plt.show()

k = [100, 200, 300, 400, 500, 600, 700, 800, 900]
buggyError = []
demeanedError = []
normalizedError = []
for i in range(len(k)):
    bV, bZ, bX = buggyPCA(data1000, k[i])
    buggyError.append(reconstructionE(data1000, bX))
    dV, dZ, dX = demeandsPCA(data1000, k[i])
    demeanedError.append(reconstructionE(data1000, dX))
    nv, nz, nx = normalizePCA(data1000, k[i])
    normalizedError.append(reconstructionE(data1000, nx))
    
plt.plot(k, buggyError)
plt.title("Buggy PCA: Reconstruction Error vs Principal Components ")
plt.show()

plt.plot(k, demeanedError)
plt.title("Demeaned PCA: Reconstruction Error vs. Principal Components")
plt.show()

plt.plot(k, normalizedError)
plt.title("Normalized PCA: Reconstruction Error vs. Principal")
plt.show()

bV, bZ, bX = buggyPCA(data1000, 500)
print(reconstructionE(data1000, bX))

dV, dZ, dX = demeandsPCA(data1000, 500)
print(reconstructionE(data1000, dX))

nv, nz, nx = normalizePCA(data1000, 500)
print(reconstructionE(data1000, nx))

#DRO

def dro(X, k):
    Xdro = X.copy()
    b = np.mean(Xdro, axis=0)
    Q = Xdro - b
    u, s, vt = np.linalg.svd(Q)
    A = vt[range(k)].reshape(X.shape[1], k)
    Z = np.matmul(Q, A)
    return b, A, Z

db2, dA2, dZ2 = dro(data2, 1)
X2d = np.matmul(dZ2, dA2.T) + db2
print(reconstructionE(data2, X2d))
plotPCA(data2, X2d)
plt.title("DRO")
plt.show()

b1000d, A1000d, Z1000d = dro(data1000, 500)
X1000d = np.matmul(Z1000d, A1000d.T) + b1000d
print(reconstructionE(data1000, X1000d))

#DRLV

def drlv(X, k):
    b, A, Z = dro(X, k)
    tX = np.matmul(Z, A.T) + b # reconstructed X
    eta = reconstructionE(X, tX)
    
    d = X.shape[1]
    Xdrlv = X.copy()

    for i in range(10):
        sigma = np.linalg.inv((np.matmul(A, A.T) + eta * np.identity(d)))
        EZ = np.matmul(np.matmul(A.T, sigma), (Xdrlv - b).T).T
        a = np.linalg.inv((eta * np.identity(d)))
        b = np.matmul(EZ.T, (Xdrlv - b))
        c = np.linalg.inv((np.matmul(EZ.T, EZ)))
        MA = np.matmul(a, np.matmul(b.T, c))
        az = np.matmul(EZ, A.T)
        Meta = np.matmul((Xdrlv - az - b).T, (Xdrlv - az - b))
        Z = EZ
        A = MA
        eta = Meta
            
    return b, A, Z, eta


db2, dA2, dZ2, etdA2 = drlv(data2, 1)
X2d = np.matmul(dZ2, dA2.T) + db2
print(reconstructionE(data2, X2d))
plotPCA(data2, X2d)
plt.title("DRLV")
plt.show()

b1000d, A1000d, Z1000d, eta1000d = drlv(data1000, 500)
X1000d = np.matmul(Z1000d, A1000d.T) + b1000d
print(reconstructionE(data1000, X1000d))