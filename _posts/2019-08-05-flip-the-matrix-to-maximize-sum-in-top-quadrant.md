---
layout: post
title: "flip-the-matrix-to-maximize-sum-in-top-quadrant"
date: 2019-08-05
tag: matrix c++ hackerrank
---

Problem:
Sean invented a game involving a matrix where each cell of the matrix contains an integer. He can reverse any of its rows or columns any number of times. The goal of the game is to maximize the sum of the elements in the `n X n` submatrix located in the upper-left quadrant of the matrix `2n X 2n`.


For example, given the matrix:
```
1 2
3 4
```
It is `2 X 2` so we want to maximize the top left matrix that is `1 X 1`. 

1. Reverse `2nd` row
2. Reverse `Ist` column
we get,
```
4 2
1 3
```
The maximal sum is `4`


Example 2:
```
112 42 83 119
56 125 56 49
15 78 101 43
62 98 114 108
```
The maximal sum is `414`

Explanation:
1. Reverse `3rd` column
2. then reverse `1st` row

Approach:
If you try to take bottom most corner `(2n, 2n)` element to Ist position `(1,1)`, we need two flip operation
Try flipping matrix so that `(2n-1, 2n-1)` element reach at `(2,2)`,
More precisely, we can take element `(i,j)` to any of other three position `symetrical` to centre position `(c,c)`, which means that we can make swapping of an element with its corresponding `3` element. So total are `4` element at each position. We use this idea to flip matrix to get maximum at the top quadrant in matrix.

Note: We assume `1 based indexing`

```c++
int flippingMatrix(vector<vector<int>> A) {
    int sum = 0;
    int n = A.size(), m = A[0].size();
    int cur, right, down, diag, ans;
    for(int i=0; i<n/2; i++){
        for(int j=0; j<m/2; j++){
            cur = A[i][j];
            right = A[i][m-j-1];
            down = A[n-i-1][j];
            diag = A[n-i-1][m-j-1];
            ans = max({cur, right, down, diag});
            sum += ans;
        }
    }
    return sum;
}
```