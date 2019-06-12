---
layout: post
title: "time-complexity-analysis-of-matrix-multiplication"
date: 2019-06-10
<!-- tags: matrix time-complexity -->
---

## Time compexity Discuss:

Algorithm  | Time Complexity (in eq form) | Big O notation 
--- | --- | --- 
Naive Method | O(n^3) | O(n^3) 
Divide And Conquer | 8 T(n/2) + O(n^2) | O(n^3)
Strassen's Method | 7 T(n/2) + O(n^2) | O(N^(2.8074))

## Naive Method:
A X B = C
[nXm] [mXp] = [nXp]
```c++
void multiply(int A[][N], int B[][N], int C[][N]){ 
    for (int i = 0; i < N; i++){ 
        for (int j = 0; j < N; j++){ 
            C[i][j] = 0; 
            for (int k = 0; k < N; k++){ 
                C[i][j] += A[i][k]*B[k][j]; 
            } 
        } 
    } 
} 

```

## Divide and Conquer Method:
```c++
void divideAndConquer(int[][] matrixA, int[][] matrixB){
    if (matrixA.length == 2){
         //calculate and return base case
    }
    else {
        //make a11, b11, a12, b12 etc. by dividing a and b into quarters      
        int[][] c11 = addMatrix(divideAndConquer(a11,b11),divideAndConquer(a12,b21));
        int[][] c12 = addMatrix(divideAndConquer(a11,b12),divideAndConquer(a12,b22));
        int[][] c21 = addMatrix(divideAndConquer(a21,b11),divideAndConquer(a22,b21));
        int[][] c22 = addMatrix(divideAndConquer(a21,b12),divideAndConquer(a22,b22));
        //combine result quarters into one result matrix and return
    }
}
```

## Strasses's Method:

p1 = a(f-h)
p2 = (a+b)h
p3 = (c+d)e
p4 = d(g-e)
p5 = (a+d)(e+h)
p6 = (b-d)(g+h)
p7 = (a-c)(e+f)

|a b| |e f| = |p5+p4-p2+p6     p1+p2   |
|c d| |f h|   |   p3+p4     p1+p5-p3-p7|

Time comp: O(N^(log7)) = O(N^(2.8074))

Algo:
1. Recursively divide the matrix A and B into 4 submatrix each and run the above method, which reduce the 8 operation(divide and conquer) to 7 operation.


## Conclusion

We showed how Strassen’s algorithm was asymptotically faster than the basic procedure of multiplying matrices. Better asymptotic upper bounds for matrix multiplication have been found since Strassen’s algorithm came out in 1969. The most asymptotically efficient algorithm for multiplying n x n matrices to date is Coppersmith and Winograd’s algorithm, which has a running time of O(n2.376)

However, in practice, Strassen’s algorithm is often not the method of choice for matrix multiplication. Cormen outlines the following four reasons for why:

1. The constant factor hidden in the O(n^(log7)) running time of Strassen’s algorithm is larger than the constant factor in the > O(n3) SQUARE-MATRIX-MULTIPLY procedure.

2. When the matrices are sparse, methods tailored for sparse matrices are faster.

3. Strassen’s algorithm is not quite as numerically stable as the regular approach. In other words, because of the limited precision of computer arithmetic on noninteger values, larger errors accumulate in Strassen’s algorithm than in SQUARE-MATRIX-MULTIPLY.

4. The submatrices formed at the levels of recursion consume space.








## Big Table for different operation on matrices
Matrix algebra

The following complexity figures assume that arithmetic with individual elements has complexity O(1), as is the case with fixed-precision floating-point arithmetic or operations on a finite field.
Operation   Input   Output  Algorithm   Complexity
Matrix multiplication   Two n×n matrices    One n×n matrix  Schoolbook matrix multiplication    O(n3)
Strassen algorithm  O(n2.807)
Coppersmith–Winograd algorithm  O(n2.376)
Optimized CW-like algorithms[25][26][27]    O(n2.373)
Matrix multiplication   One n×m matrix &

one m×p matrix
    One n×p matrix  Schoolbook matrix multiplication    O(nmp)
Matrix inversion*   One n×n matrix  One n×n matrix  Gauss–Jordan elimination    O(n3)
Strassen algorithm  O(n2.807)
Coppersmith–Winograd algorithm  O(n2.376)
Optimized CW-like algorithms    O(n2.373)
Singular value decomposition    One m×n matrix  One m×m matrix,
one m×n matrix, &
one n×n matrix  Bidiagonalization and QR algorithm  O(mn2 + m2 n)
(m ≥ n)
One m×n matrix,
one n×n matrix, &
one n×n matrix  Bidiagonalization and QR algorithm  O(mn2)
(m ≥ n)
Determinant     One n×n matrix  One number  Laplace expansion   O(n!)
Division-free algorithm[28]     O(n4)
LU decomposition    O(n3)
Bareiss algorithm   O(n3)
Fast matrix multiplication[29]  O(n2.373)
Back substitution   Triangular matrix   n solutions     Back substitution[30]   O(n2)

