import numpy as np
import matplotlib.pyplot as plt
class Node:
    def __init__(self):
        self.rowsWithZero = 0
        self.rank = 0
        self.Ucolumns = None
        self.Vrows = None
        self.eigenvalues = None
        self.children = []
        self.m = 0
        self.n = 0
            

    def draw_matrix(self):
        def build_draw_matrix(node):
            if node.children != []:
                return np.vstack((np.hstack((build_draw_matrix(node.children[0]), build_draw_matrix(node.children[1]))),
                        np.hstack((build_draw_matrix(node.children[2]), build_draw_matrix(node.children[3]))),))
            if node.rank > 0:
                mtrx = np.zeros((node.n, node.m))
                mtrx[:, :node.rank] = 1
                mtrx[:node.rank, :] = 1
                return mtrx
            else:
                return np.zeros((node.n, node.m))
        fig, ax = plt.subplots(figsize=(16, 9))
        ax.matshow(build_draw_matrix(self), cmap=ListedColormap(['w', 'k']))
        plt.show()
    
    def count_nodes(self):
        count = 1  
        for child in self.children:
            count += child.count_nodes()  
        return count
        

def compress_matrix(A, epsilon, r):
    '''Matrix compression using the SVD algorithm.'''

    v = Node()

    numRows, numCols = A.shape
    submatrices = [
        A[:numRows//2, :numCols//2],
        A[:numRows//2, numCols//2:],
        A[numRows//2:, :numCols//2],
        A[numRows//2:, numCols//2:],
    ]
    v.m = numRows
    v.n = numCols

    for B in submatrices:
        if np.count_nonzero(B) == 0:
            w = Node()
            w.rowsWithZero = B.shape[0]
            w.rank = 0
            v.children.append(w)
            w.m = numRows//2
            w.n = numCols//2
        else:
            U, D, Vt = np.linalg.svd(B)

            eigenvalues = np.diag(D)
            k = np.sum(D > epsilon)

            if k == 0:
                w = Node()
                w.rowsWithZero = B.shape[0]
                w.rank = 0
                v.children.append(w)
                w.m = numRows//2
                w.n = numCols//2
            elif (k <= r) and ((k < (numRows // 2)) or (k == 1)):
                w = Node()
                w.rank = k
                w.Ucolumns = U[:, :k]
                w.Vrows = Vt[:k, :].transpose()
                w.eigenvalues = eigenvalues[:k, :k]
                w.Vrows = w.Vrows @ w.eigenvalues
                v.children.append(w)
                w.m = numRows//2
                w.n = numCols//2
            else:
                w = compress_matrix(B, epsilon, r)
                v.children.append(w)
    return v
