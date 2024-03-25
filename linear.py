import numpy as np, pandas as pd, re
# Helper Methods
def separate(filename, sheet, size):
	df = pd.read_excel(filename, sheet_name = sheet, header=None).astype(float)
	return [np.matrix(df[i:i+size]) for i in range(0, len(df), size)]
def test():
	a = np.matrix([
		[1,2,3],
		[0,1,-2],
		[0,1,4]
	])
	print(a @ np.array([0,0,0]).astype(float))
	print(a @ np.array([[0,0,0]]).T)
	print(a @ np.matrix([0,0,0]).T)
	print(a @ np.array([0,0,0]))
def multiply(data, func = lambda array, output: array @ output):
	output = data[0].copy()
	print(output)
	for array in data[1:]:
		output = func(array, output)
		print(output)
	return output
multiplyR = lambda data: multiply(data, lambda array, output: output @ array)
# RREF Related Methods
def rref(a, option = 'Verbose'):
	def print2(input):
		if option == 'Verbose':
			print(input)
	def add():
		if len(output) == 0 or (len(output) > 0 and not np.array_equal(u, output[-1])):
			print2(u)
			output.append(u.copy())
	def print3(input):
		print2(input)
		add()
	def subtract(start, end):
		for destination in range(start, end):
			if u[source,pivot] != 0:
			# subtract multiple of source row from destination row to cancel out pivot column
				u[destination,:] = u[destination,:] - u[source,:] * u[destination,pivot] / u[source,pivot]
		add()
	output, pivots, pivot, flip = [], [], 0, 1
	u = a.astype(float) if 'int' in str(a.dtype) else a.copy()
	print3('Bottom half')
	# Loop through smaller dimension of matrix
	for source in range(0, min(u.shape)):
		flipped = False
		# Stop if pivot is found or you run out of columns
		while flipped == False and pivot < u.shape[1]:
			# Go from source column to last row of matrix
			for source2 in range(source, len(u)):
				# If source2 row and pivot column is nonzero
				if np.absolute(u[source2,pivot]) > 1e-8:
					# If loop has moved to later row, flip source2 row and source row
					if source != source2:
						# Count whether number of flips is odd
						flip = - flip
						temp = u[source,:].copy()
						u[source,:] = u[source2,:].copy()
						u[source2,:] = temp.copy()
					# Pivot is found. Append pivot, and subtract pivot row from other rows
					flipped = True
					pivots.append(pivot)
					subtract(source+1, len(u))
					break
			# Move pivot to next column
			pivot += 1
	if option in ('Verbose', 'Silent'):
		print('Top Half')
		# Subtract from top row to source row for each pivot. Bottom up
		for source, pivot in reversed(list(enumerate(pivots))):
			subtract(0, source)
		# Divide pivot rows by pivot value to make all pivots 1
		for source, pivot in enumerate(pivots):
			if u[source,pivot] != 0:
				u[source,:] = u[source,:] / u[source,pivot]
		print3('Divide')
	if option == 'Verbose':
		return output
	elif option == 'Silent':
		return output[-1], pivots
	elif option == 'Short':
		return output[-1], flip, pivots
class RREF:
	def __init__(self, matrix_input, option = 'Verbose'):
		self.matrix_input = matrix_input
		self.option = option
		self.matrices, self.pivot_columns, self.column_number, self.flip_sign = [], [], 0, 1
		self.matrix = self.matrix_input.astype(float) if 'int' in str(self.matrix_input.dtype) else self.matrix_input.copy()
	def print(self, input_):
		if self.option == 'Verbose':
			print(input_)
	def append_and_print_matrix(self):
		if len(self.matrices) == 0 or (
			len(self.matrices) > 0 and 
			# Matrix is different from last appended matrix
			not np.array_equal(self.matrix, self.matrices[-1])
		):
			self.print(self.matrix)
			self.matrices.append(self.matrix.copy())
	def flip_rows(self):
		# Count whether number of flips is odd
		self.flip_sign = - self.flip_sign
		self.row_temp = self.matrix[self.row_number,:].copy()
		self.matrix[self.row_number,:] = self.matrix[self.row_number2,:].copy()
		self.matrix[self.row_number2,:] = self.row_temp.copy()
	def subtract_row(self):
		self.pivot_value = self.matrix[self.row_number, self.column_number]
		self.destination_value = self.matrix[self.destination_row, self.column_number]
		self.multiplier = self.destination_value / self.pivot_value
		self.multiplied_source_row = self.matrix[self.row_number, :] * self.multiplier
		if self.pivot_value != 0:
			# Not sure why round is needed in new version but not old version
			self.matrix[self.destination_row, :] = self.matrix[self.destination_row, :] - self.multiplied_source_row.round(10)
	def handle_pivot(self):
		# If loop has moved to later row, flip source2 row and source row
		if self.row_number != self.row_number2:
			self.flip_rows()
		# Pivot is found. Append pivot, and subtract pivot row from other rows
		self.pivot_found = True
		self.pivot_columns.append(self.column_number)
		for self.destination_row in range(self.row_number+1, len(self.matrix)):
			self.subtract_row()
		self.append_and_print_matrix()
	def bottom_half(self):
		self.print('Bottom half')
		self.append_and_print_matrix()
		# Loop through smaller dimension of matrix
		for self.row_number in range(0, min(self.matrix.shape)):
			self.pivot_found = False
			# Stop if pivot is found or you run out of columns
			while self.pivot_found == False and self.column_number < self.matrix.shape[1]:
				# Go from source row to last row of matrix
				for self.row_number2 in range(self.row_number, len(self.matrix)):
					# If source2 row and pivot column is nonzero
					if np.absolute(self.matrix[self.row_number2, self.column_number]) > 1e-8:
						self.handle_pivot()
						break
				# Move next column, even if pivot is not found
				self.column_number += 1
	def top_half(self):
		print('Top half')
		# Subtract from top row to source row for each pivot. Bottom up
		for self.row_number, self.column_number in reversed(list(enumerate(self.pivot_columns))):
			for self.destination_row in range(0, self.row_number):
				self.subtract_row()
			self.append_and_print_matrix()
		# Divide pivot rows by pivot value to make all pivots 1
		for self.row_number, self.column_number in enumerate(self.pivot_columns):
			self.pivot_value = self.matrix[self.row_number, self.column_number]
			if self.pivot_value != 0:
				self.matrix[self.row_number,:] = self.matrix[self.row_number,:] / self.pivot_value
		self.print('Divide')
		self.append_and_print_matrix()
	def make_output(self):
		if self.option == 'Verbose':
			self.output = self.matrices
		elif self.option == 'Silent':
			self.output = self.matrices[-1], self.pivot_columns
		elif self.option == 'Short':
			self.output = self.matrices[-1], self.flip_sign, self.pivot_columns
	def main(self):
		self.bottom_half()
		if self.option in ('Verbose', 'Silent'):
			self.top_half()
		self.make_output()
		return self.output
def rref(a, option = 'Verbose'):
	rref_ = RREF(matrix_input = a, option = option)
	return rref_.main()
augment = lambda a, b: np.concatenate((a, b), axis=1)
rrefAugmented = lambda a, b, option = 'Verbose' : rref(augment(a, b), option)
rrefIdentity = lambda a, option = 'Verbose' : rrefAugmented(a, np.identity(len(a)), option)
def pivots(a):
	u, _, pivots = rref(a, 'Short')
	return u[list(range(len(pivots))), pivots]
# Subspaces
def columnSpace(a):
	# Pivot columns of original matrix
	u, pivots = rref(a, 'Silent')
	return a[:,pivots]
def rowSpace(a):
	# Top p rows of reduced matrix, where p = # of pivots
	u, pivots = rref(a, 'Silent')
	return u[:len(pivots),:]
def nullSpace(a):
	# Free columns of reduced matrix, put into pivot rows
	u, pivots = rref(a, 'Silent')
	# Get free nonpivot columns
	freeColumns = [c for c in range(a.shape[1]) if c not in set(pivots)]
	# Make blank matrix - nrows = a's number of columns, ncolumns = free columns
	output = np.asmatrix(np.zeros([a.shape[1], len(freeColumns)]))
	output = output.astype(complex) if 'complex' in str(a.dtype) else output
	# negative nonblank rows, free columns
	free = - u[:len(pivots),freeColumns]
	# Put nonblank row free columns in pivot rows
	output[pivots,:] = free
	# Put identity everywhere else
	output[freeColumns,:] = np.identity(len(freeColumns))
	return output
def leftNullSpace(a):
	# Reduce a attached to identity
	u, pivots = rrefIdentity(a, 'Silent')
	# Get Pivot columns for a only
	aPivots = [p for p in pivots if p < a.shape[1]]
	# Get rows past pivot rows, for transformed identity
	return u[len(aPivots):,a.shape[1]:]
def xParticular(a, b):
	u, pivots = rrefAugmented(a, b, 'Silent')
	output = np.asmatrix(np.zeros([a.shape[1], 1]))
	output = output.astype(complex) if 'complex' in str(a.dtype) else output 
	output[pivots,0] = u[:,-1]
	return output
# Determinant Related Methods
def square(a):
	if a.shape[0] != a.shape[1]:
		raise Exception	
def det(a, option='Verbose'):
	square(a)
	u, output, _ = rref(a, 'Short')
	if option == 'Verbose':
		print(u)
	for row in range(u.shape[0]):
		output *= u[row, row]
	return output
def det2(a):
	square(a)
	if len(a) == 1:
		return a[0,0]
	output = 0
	for column in range(len(a)):
		temp = list(range(len(a)))
		temp.remove(column)
		output += a[0,column] * (-1) ** column * det2(a[1:,temp])
	return output
def crossProduct(a):
	if a.shape != (2,3):
		raise Exception
	list_ = [det(a[:,i]) for i in ([1,2], [2,0], [0,1])]
	return np.matrix([list_])
# Inverse Related Methods
def inverse(a):
	square(a)
	return rrefIdentity(a, 'Silent')[0][:,a.shape[1]:]
def cofactor(a):
	square(a)
	output = np.asmatrix(np.zeros([len(a), len(a)]))
	output = output.astype(complex) if 'complex' in str(a.dtype) else output 
	for row in range(len(a)):
		rowTemp = list(range(len(a)))
		rowTemp.remove(row)
		for column in range(len(a)):
			columnTemp = list(range(len(a)))
			columnTemp.remove(column)
			output[row, column] = (-1) ** (column + row) * det(a[rowTemp, :][:, columnTemp], 'Silent')
	return output
inverse2 = lambda a: cofactor(a).T / det(a, 'Silent')
leftI = lambda a: (a.T @ a).I @ a.T
rightI = lambda a: a.T @ (a @ a.T).I
def pseudoI(a):
	u, s, vt = np.linalg.svd(a)
	s = [ss for ss in s if np.absolute(ss) > 1e-3]
	return vt.T @ diagonal(s, -1, a.T.shape) @ u.T
# Projection related methods
def projection(a, option='Verbose'):
	if option == 'Verbose':
		aList = [a, (a.T @ a).I, a.T]
		print('Arrays')
		for a in aList:
			print(a)
		print('Multiply')
		return multiplyR(aList)
	else:
		return a @ (a.T @ a).I @ a.T
def normalise(a):
	q = a.astype(float) if 'int' in str(a.dtype) else a.copy()
	for column in range(q.shape[1]):
		e = q[:,column]
		length = np.sqrt(e.T @ e)
		q[:,column] = e / length if length != 0 else 0
	return q
def gram(a):
	projection2 = lambda b, q: b - q @ (q.T @ q).I @ q.T @ b
	e1 = lambda: projection2(a[:,column], q)
	def e2():
		e = a[:,column]
		for column2 in range(column):
			e = projection2(e, q[:,column2])
		return e
	# Projection
	q = a[:,0]
	for column in range(1, a.shape[1]):
		q = augment(q, e1())
	print(q)
	q = normalise(q)
	print(q)
	return q
# Eigenvector related methods
def eigen2x2(a):
	if a.shape != (2,2):
		raise Exception
	b = - a[0,0] - a[1,1]
	c = det(a, 'Silent')
	return ((-b) + np.array([1,-1]) * np.lib.scimath.sqrt(b ** 2 - 4 * c))/2
def eigenVectors(a, eigenValues, option='Verbose'):
	square(a)
	column = 0
	output = np.asmatrix(np.zeros([len(a), len(a)]))
	output = output.astype(complex) if 'complex' in str(a.dtype)+str(np.asmatrix(eigenValues).dtype) else output 
	for l in eigenValues:
		# Get Eigenvectors
		eigen = nullSpace(a-l*np.identity(len(a)))
		size = eigen.shape[1]
		if size == 0:
			if option == 'Verbose':
				print(l)
		else:
			for e in eigen.T:
				if option == 'Verbose':
					print(l, e, e * l, (a @ e.T).T)
				output[:,column] = e.T
				column += 1
	return output
def diagonal(eigenValues, power=1, dimensions = None, func = lambda l, power: l ** power):
	dimensions = [len(eigenValues), len(eigenValues)] if dimensions is None else dimensions
	output = np.asmatrix(np.zeros(dimensions))
	output = output.astype(complex) if isinstance(power, complex) else output
	for i, l in enumerate(eigenValues):
		if i < min(dimensions):
			output[i,i] = func(l, power)
	return output
diagonalExp = lambda e, p=1, d = None: diagonal(e, p, d, lambda l, p: np.exp(l * p))
def aku0(a, eigenValues, dFunc, u0, k):
	s = eigenVectors(a, eigenValues, 'Silent')
	return s @ dFunc(eigenValues, k) @ s.I @ u0
# Other Methods
def svd(a, option='Verbose'):
	ata, aat = a.T @ a, a @ a.T
	if a.shape == (1,2):
		s2 = [aat[0,0],0]
	elif a.shape in ((2,1), (1,1)):
		s2 = [ata[0,0],0]
	elif a.shape == (2,2):
		s2 = eigen2x2(ata)
	else:
		raise Exception
	s = diagonal(s2, 0.5, a.shape)
	si = diagonal(s2, -0.5, a.T.shape)
	if a.shape == (2,1):
		u = normalise(eigenVectors(aat, s2, 'Silent'))
		v = a.T @ u @ si.T
	else:
		v = normalise(eigenVectors(ata, s2, 'Silent'))
		u = a @ v @ si
		# u = normalise(eigenVectors(aat, s2, 'Silent'))
	if option == 'Verbose':
		print((u @ s @ v.T).round(2))
	return u, s, v.T
def fft(n):
	if n == 1:
		return np.matrix(1)
	if np.log2(n) % 1 != 0:
		raise Exception
	zero = lambda : np.asmatrix(np.zeros([n, n])).astype(float).astype(complex)
	half = int(n/2)
	idid, fhfh, p = zero(), zero(), zero()
	# IDID
	idid[:half,:half] = np.identity(half)
	idid[half:,:half] = np.identity(half)
	d = diagonal(list(range(half)), 1j, func= lambda l, power: power ** l)
	idid[:half,half:] = d
	idid[half:,half:] = -d
	# FHFH
	fhalf = fft(half)
	fhfh[:half,:half] = fhalf
	fhfh[half:,half:] = fhalf
	# P
	for column in range(n):
		row = column/2 if column % 2 == 0 else (column+n)/2
		p[int(row),column] = 1
	output = idid @ fhfh @ p
	return output
## Convert Latex to Matrix and Vice Versa
def toMatrix(a):
	a = re.sub(r'\$.*egin{bmatrix}(.*)\\end{bmatrix}\s*\$', r'\1', a)
	output = []
	for row in a.split('\\'):
		output.append([])
		for item in row.split('&'):
			output[-1].append(float(item.strip()))
	return np.matrix(output)
def toTex(a):
	output = r' \\ '.join([' & '.join([str(item) for item in row]) for row in np.asarray(a)])
	return '$ \\begin{bmatrix} %s \\end{bmatrix} $' % output
if __name__=='__main__':
	u2 = np.matrix([
		[1,2,2,2],
		[2,4,6,8],
		[3,6,8,10]
	])
	rref(u2)
	print(nullSpace(u2))
