% =========================================================================
%                           Parametric Dictionary Learning for Graph Signals
% =========================================================================

The code is the MATLAB implementation of the dictionary learning algorithm discussed in Section IV.B of the reference paper:

D. Thanou, D. I Shuman, and P. Frossard, “Learning Parametric Dictionaries for Signals on Graphs”, Submitted to IEEE Transactions on Signal Processing, Available at: 
http://arxiv.org/pdf/1401.0887.pdf

In the toolbox, you can find the following files: 

1) testdata.mat: The necessary data that are needed to reproduce the synthetic results of Section V.A.1. It contains the following variables:
	XCoords, YCoords: x, y coordinates of a random graph
	A: Adjacency Matrix of the graph
	W: Weight Matrix
	D: ‘oracle’ polynomial dictionary
	C: polynomial coefficients of the oracle dictionary
	TrainSignal: set of training signals
	TestSignal: set of testing signals

2) Polynomial_Dictionary_Learning.m: The main file that implements the polynomial dictionary learning algorithm. Input and output parameters are described inside. 

3) OMP.m: Compute the sparse coding coefficients for a set of signals Y, given a dictionary D and a specified sparsity level T0. The function is a slight modification of the OMP.m function included in the KSVD toolbox that implements the following paper: 

	"M. Aharon, M. Elad, and A.M. Bruckstein, "The K-SVD: An Algorithm for Designing of  	Overcomplete Dictionaries for Sparse Representation", IEEE Trans. On Signal 	Processing, Vol. 54, no. 11, pp. 4311-4322, November 2006."

4) coefficient_update_interior_point.m: It implements the function for learning the polynomial coefficients using interior points methods. We use the sdpt3 solver in the YALMIP optimization toolbox to solve the quadratic program. Both are publicly available in the following links:

	sdpt3: http://www.math.nus.edu.sg/~mattohkc/sdpt3.html
	YALMIP: http://users.isy.liu.se/johanl/yalmip/

4) coefficient_update_ADMM.m: It implements the alternating direction method of multipliers in order to learn the polynomial coefficients. The implementation is based on the following paper:

	"S. Boyd, N. Parikh, E. Chu, B. Peleato, and J. Eckstein, “Distributed optimization and  			statistical learning via the alternating direction method of multipliers,” Foundations and Trends 		in Machine Learning, vol. 3, no. 1, pp. 1–122, 2011."


5) demo_Polynomial_Dictionary_Learning.m:  Run file that applies the polynomial dictionary learning algorithm in the data contained in testdata.mat. 

For comments or questions please contact: Dorina Thanou (dorina.thanou@epfl.ch). 
Copyright (c) 2014, Dorina Thanou

