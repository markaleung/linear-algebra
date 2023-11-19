# linear-algebra
Linear Algebra Functions Implemented in Numpy
- This repository contains my notebooks for doing the [MIT OpenCourseWare linear algebra course](https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/) using numpy
- There is one notebook for each assignment, and all notebooks import linear.py
- I hope they will be helpful for anyone else studying linear algebra. 

# Table of Contents

## 1 Ax=B and the 4 Subspaces
2. Elimination with Matrices
3. Multiplication and Inverse Matrices
4. Factorization
7. Solving ax=b 0 pivot variables
8. Solving Ax = b Row Reduced Form R
9. Independence, basis and dimension
10. Four Fundamental Subspaces
11. Matrix Spaces, Rank 1, Small World Graphs
12. Graphs, Networks, Incidence Matrices
13. Exam 1 Review
14. Unit 1 Exam

## 2 Least Squares, Determinants and Eigenvalues
1. Orthogonal Vectors and Subspaces
2. Projections onto Subspaces
3. Projection Matrices and Least Squares
4. Orthogonal Matrices and Gram Schmidt
5. Properties of Determinants
6. Determinant Formulas and Cofactors
7. Cramer's Rule, Inverse Matrix and Volume
8. Eigenvalues and Eigenvectors
9. Diagonalisation and Powers of A
10. Differential Equations and exp(At)
11. Markov Series; Fourier Series
12. Exam 2 Review
13. Exam 2

## 3 Positive Definite Matrices and Applications
1. Symmetric Matrices and Positive Definiteness
2. Complex Matrices, Fast Fourier Transform
3. Positive Definite Matrices and Minima
4. Similar Matrices and Jordan Form
5. Singular Value Decomposition
6. Linear Transformations
7. Change of Basis; Image Compression
8. Left and Right Inverses; Pseudoinverse
9. Exam 3 Review
10. Exam 3

## 4 Final Exam
1. Final Course Review
2. Exam

# List of functions

## Helper methods
- multiply(data, func): applies matrix multiplication across an array of numpy matrices (notebook 1_2, 1_Exam)
- multiplyR(data): multiply() but with the matrices in reverse order (notebook 2_11, 3_8)

## Reduced Row Echelon Form
- rref(a, option): applies the Reduced Row Echelon Form to a matrix (many notebooks)
  - option = 'Verbose': print intermediate steps in function
  - option = 'Silent': don't print anything, return pivot columns
  - option = 'Short': don't run second part of function, return whether number of flips is odd (needed for determinent)
- augment(a, b): concatenate a and b
- rrefAugmented(a, b, option): rref(augment(a, b), option) (notebook 1_12, 1_13)
- rrefIdentity(a, option): rrefAugmented(a, np.identity(len(a)), option) (notebook 1_10, 1_12, 2_1, 2_6)
- pivots(a): Get pivot values from rref(), don't run second part of function (notebook 3_3)

## Subspaces (notebook 4_Exam)
- columnSpace(a): return the columnspace of a matrix
- rowSpace(a): return the rowspace of a matrix
- nullSpace(a): return the nullpace of a matrix (notebook 3_8)
- leftNullSpace(a): return the left nullspace of a matrix (notebook 3_9)

## Determinent
- det(a): return the determiant of a matrix (many notebooks)
- det2(a): return the determinant using alternative method (2x2 matrix only, notebook 2_6)
- crossProduct(a): return the cross product of a matrix (notebook 1_9)

## Inverse
- inverse(a): return the inverse of a matrix
- cofactor(a): return the cofactors of a matrix (notebook 2_7, 2_12, 4_Exam)
- inverse2: return the inverse of a matrix using cofactors 
- leftI(a): return the left inverse of a matrix (notebook 3_8)
- rightI(a): return the right inverse of a matrix (notebook 3_8)
- pseudoI(a): return the psuedo inverxe of a matrix (notebook 3_8)

## Projection
- projection(a): return the projection of a matrix (notebook 2_2, 2_3, 3_9)
- normalise(a): return a normalised matrix (notebook 3_7)
- gram(a): use gram-schmidt process to make matrix orthonormal (notebook 2_4)

## Eigenvector
- eigen2x2(a): get eigenvalues of a matrix (2x2 matrix only, many notebooks)
- eigenVectors(a, eigenValues): get eigen vectors of a matrix using given eigen values
- diagonal(eigenValues, power = 1, dimensions = None, func = lambda l, power: l ** power): 
  - Create a diagonal matrix for a given list of eigenvalues
  - Values are 0 except diagonals
  - each ith diagonal value is func(eigenvalue[i], power)
  - Many notebooks
- diagonalExp(e, p, d): return exponential diagonal matrix (notebook 2_10, 3_Exam)
- aku0(a, eigenValues, dFunc, u0, k): 
  - Difference equation u(k+1) = Au(k) 
  - For matrix a, eigenvalues, diagonal function dFunc, initial vector u0, power k
  - notebooks 2_9, 2_10, 2_11, 3_Exam

## Other functions
- svd(a): return singular value decomposition of a matrix (maximum 2x2)
- fft(n): return fft of an integer

## Conversion functions
- toMatrix(a): convert latex matrix to numpy
- toTex(a): convert numpy matrix to latex