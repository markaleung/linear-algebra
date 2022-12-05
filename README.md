# linear-algebra
Linear Algebra Functions Implemented in Numpy
- This repository contains a single file with numpy functions for performing common linear algebra operations. 
- I wrote these functions to help me do the homeworks for the [MIT OpenCourseWare linear algebra course](https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/). 
- I hope they will be helpful for anyone else studying linear algebra. 

# List of functions

## Helper methods
- multiply(data, func): applies matrix multiplication across an array of numpy matrices
- multiplyR(data): multiply() but with the matrices in reverse order

## Reduced Row Echelon Form
- rref(a, option): applies the Reduced Row Echelon Form to a matrix
  - option = 'Verbose': print intermediate steps in function
  - option = 'Silent': don't print anything, return pivot columns
  - option = 'Short': don't run second part of function, return whether number of flips is odd (needed for determinent)
- augment(a, b): concatenate a and b
- rrefAugmented(a, b, option): rref(augment(a, b), option)
- rrefIdentity(a, option): rrefAugmented(a, np.identity(len(a)), option)
- pivots(a): Get pivot values from rref(), don't run second part of function

## Subspaces
- columnSpace(a): return the columnspace of a matrix
- rowSpace(a): return the rowspace of a matrix
- nullSpace(a): return the nullpace of a matrix
- leftNullSpace(a): return the left nullspace of a matrix

## Determinent
- det(a): return the determiant of a matrix
- crossProduct(a): return the cross product of a matrix

## Inverse
- inverse(a): return the inverse of a matrix
