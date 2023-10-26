import numpy as np

def cauchy_binet_recursive(A, B):

    if A.shape[1] != B.shape[0]:  
        raise ValueError("Matrix dimensions are not compatible for multiplication")

    n = A.shape[1]

    if n == 1: 
        return A[0, 0] * B[0, 0], 1

    
    # Split matrices A and B into submatrices
    m = n // 2
    A11, A12 = A[:m, :m], A[:m, m:]
    A21, A22 = A[m:, :m], A[m:, m:]
    B11, B12 = B[:m, :m], B[:m, m:]
    B21, B22 = B[m:, :m], B[m:, m:]

    # Recursively compute subdeterminants
    LU1, count1 = cauchy_binet_recursive(A11, B11)
    LU2, count2 = cauchy_binet_recursive(A12, B21)
    RU1, count3 = cauchy_binet_recursive(A11, B12)
    RU2, count4 = cauchy_binet_recursive(A12, B22)
    LD1, count5 = cauchy_binet_recursive(A21, B11)
    LD2, count6 = cauchy_binet_recursive(A22, B21)
    RD1, count7 = cauchy_binet_recursive(A21, B12)
    RD2, count8 = cauchy_binet_recursive(A22, B22)
    
    result = np.zeros((n, n))
    result[:m, :m] = LU1 + LU2
    result[:m, m:] = RU1 + RU2
    result[m:, :m] = LD1 + LD2
    result[m:, m:] = RD1 + RD2
    
    count = count1 + count2 + count3 + count4 + count5 + count6 + count7 + count8 + 4*(n//2)**2

    return result, count

def strassen_matrix_multiply(A, B):
    if A.shape[1] != B.shape[0]:
        raise ValueError("Matrix dimensions are not compatible for multiplication")

    if A.shape[0] == 1 and A.shape[1] == 1:
        return A * B, 1

    # Split the matrices into four equal-sized parts
    m = A.shape[0] // 2

    a11, a12 = A[:m, :m], A[:m, m:]
    a21, a22 = A[m:, :m], A[m:, m:]

    b11, b12 = B[:m, :m], B[:m, m:]
    b21, b22 = B[m:, :m], B[m:, m:]

    # Calculate the 7 products P1 to P7 using Strassen's method
    p1, count1 = strassen_matrix_multiply(a11, (b12 - b22))
    p2, count2 = strassen_matrix_multiply((a11 + a12), b22)
    p3, count3 = strassen_matrix_multiply((a21 + a22), b11)
    p4, count4 = strassen_matrix_multiply(a22, (b21 - b11))
    p5, count5 = strassen_matrix_multiply((a11 + a22), (b11 + b22))
    p6, count6 = strassen_matrix_multiply((a12 - a22), (b21 + b22))
    p7, count7 = strassen_matrix_multiply((a11 - a21), (b11 + b12))

    # Calculate the four parts of the result matrix
    c11 = p5 + p4 - p2 + p6
    c12 = p1 + p2
    c21 = p3 + p4
    c22 = p5 + p1 - p3 - p7

    # Combine the results
    top = np.hstack((c11, c12))
    bottom = np.hstack((c21, c22))
    result = np.vstack((top, bottom))
    
    count = count1 + count2 + count3 + count4 + count5 + count6 + count7 + 18*(A.shape[0]//2)**2
    
    return result, count