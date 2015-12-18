rsvm_simulation_code
====

**rsvm_simulation_code** contains the file's necessary for reproducing the results in
> [M. Sundin][masundi], [C.R. Rojas][rojas], [M. Jansson][mjansson] and [S. Chatterjee][saikat]. "Relevance Singular Vector Machine for Low-rank Matrix Reconstruction". *IEEE Transaction
 on Signal Processing*, 2015. Submitted. A preprint of the manuscript is avaliable [here][arxiv].

It provides the simulation code and scripts required to reproduce the  figures from the paper.

Please report any bugs or errors to <sundin83martin@gmail.com>.

## Abstract
We develop Bayesian learning methods for lowrank matrix reconstruction and completion from linear measurements. For under-determined systems, the developed methods reconstruct low-rank matrices when neither the rank nor the noise power is known a-priori. We derive relations between 
the proposed Bayesian models and low-rank promoting penalty functions. The relations justify the use of Kronecker structured covariance matrices in a Gaussian based prior. In the methods, 
we use expectation-maximization to learn the model parameters. The performance of the methods is evaluated through extensive numerical simulations on synthetic and real data.

## Description of simulations
The simulation files are of two types, **algorithm scripts** and **simulation files**. **Algorithm scripts** contain code for the implemented algorithms and the **simulation files** contain code for estimating the error of the algorithms for certain parameter values. The convex optimization based methods require the [cvx toolbox][cvx] tot run.


## Description of the problem

In the paper we study the Low Rank Matrix Reconstruction (LRMR) problem. The LRMR problem is to estimate a (p times q) low rank matrix **X** from measurements

**y = A**vec(**X**) + **n**

where **A** is the (m times pq) measurement matrix, **n** is additive random noise and **y** is the observed measurements. In the case of matrix completion we sometimes use the (`p` times `q`) matrix **Y** with entries

Y(i,j) = X(i,j) + n(i,j), if entry (i,j) is observed,

Y(i,j) = 0              , if entry (i,j) is not observed.

The Goal is to estimate **X** from **y** (or **Y**). We use the Normalized Mean Square Error (NMSE) as performance measure.

## Running the simulations
To run the simulations perform the following steps:
1. If not already installed, install and and setup [cvx][cvx].
2. Open the simulation file you want to run and chose appropriate parameter values.
3. Change the variable `filename` to the name of your choice. 
4. Run the chosen file.
5. The simulation results will be saved in `filename.mat`.

## Description of algorithm scripts

The algorithm scripts implements the different estimation algorithms.

For references, see the paper.

### `rsvm_ld.m`
- Relevance Vector Machine for log-determinant penalty (introduced in paper).

- Call as `Xhat = rsvm_ld(y,A,p,q)`.

- Runs RSVM-LD for a `p` times `q` matrix.

### `rsvm_schatten.m`
- Relevance Vector Machine for Schatten norm penalty (introduced in paper).

- Call as `Xhat = rsvm_schatten(y,A,p,q,s)`.

- Runs RSVM-SN for a `p` times `q` matrix with Schatten `s`-norm.

### `Nuclear_norm.m`
- Nuclear norm minimization for low-rank matrix reconstruction and completion.

- Call as `Xhat = nuclear_norm(y,A,p,q,lambda)`.

- Solves the convex optimization problem

Xhat = arg min norm_nuc(X), subject to norm(**y - A**vec(**X**),2) <= lambda.

- Uses the [cvx toolbox][cvx].

### `weighted_trace_norm.m`

- Weighted Trace norm for matrix completion.

- Call as `Xhat = weighted_trace_norm(Y,lambda)`.

- Solves the convex optimization problem

Xhat = arg min norm_nuc(**PXQ**), subject to norm(**Y - X**,'fro') <= lambda,

for diagonal weighting matrices **P** and **Q**.

- Uses the [cvx toolbox][cvx].

### `prob_matrix_fact.m`

- Probabilistic Matrix factorization for matrix completion

- Call as `Xhat = prob_matrix_fact(Y)`.

### `bayesian_pca.m`

- Bayesian PCA for missing values.

- Call as `Xhat = bayesian_pca(Y)`.

### `schatten_norm_type1.m`

- Cost function J(**X**) = sum(svd(**X**).^s) + lambda*norm(**y - A**vec(**X**),2)^2, for Schatten `s`-norm. Can be minimized for using standard Matlab functions to be used as a Type-I estimate.

- Call function as `[J,grad] = schatten_norm_type1(X,A,y,p,q,s,lambda)`.

- Minimize function as e.g.
```
options = optimset('GradObj', 'on', 'MaxIter', 100);
Xhat = fminunc(@(t)(schatten_norm_type1(t,A,y,p,q,s,lambda)),pinv(A)*y,options);
Xhat = reshape(Xhat,p,q);
```

### `variational_movierating.m`

- Variational approach to movie rating prediction. 

- Call as `Xhat = variational_movierating(Y)`.

### `vb_completion.m`

- Sparse Bayesian methods for low rank matrix estimation.

- Call as `Xhat = vb_completion(y,A,p,q,r_prior)`.

- A standard choice is to set `r_priot = min(p,q)`.

### `vb_reconstruction.m`

- Variational Bayesian method for matrix reconstruction. No partition of the factor matrices.

- Call as `[Xhat,Ahat,Bhat] = vb_reconstruction(y,Phi,p,q,rmax)`

### `vb_reconstruction_column.m`

- Variational Bayesian method for matric reconstruction. Column-wise partition of the factor matrices.

- Call as `[Xhat,Ahat,Bhat] = vb_reconstruction_column(y,Phi,p,q,rmax)`.

- A standard choice is `rmax = min(p,q)`.

### `vb_reconstruction_row.m`

- Variational Bayesian method for matric reconstruction. Row-wise partition of the factor matrices.

- Call as `[Xhat,Ahat,Bhat] = vb_reconstruction_row(y,Phi,p,q,rmax)`.

- A standard choice is `rmax = min(p,q)`.

## Description of simulation files

### `alpha_schatten_reconstruction.m`

- Measure how the NMSE varies for the RSVM-SN methods for different `s` and number of measurements `m`. Figure 1 in paper.

### `error_alpha_rec.m`

- Measure how the NMSE varies with measurement fraction `alpha = m/pq` for matrix reconstruction. Figure 2.a. and Figure 3.a. in paper.

### `error_snr_rec.m`

- Measure how the NMSE varies with SNR for matrix reconstruction. Figure 2.b. in paper.

### `rank_reconstruction_plot.m`

- Measure how the NMSE varies with rank for matrix reconstruction. Figure 2.c. and Figure 3.b. in paper.

### `alpha_schatten_completion.m`

- Measure how the NMSE varies with measurement fraction `alpha = m/pq` for matrix completion. Figure 5 in paper.

### `error_alpha_comp.m`

- Measure how the NMSE varies with measurement fraction `alpha = m/pq` for matrix completion. Figure 6.a. in paper.

### `error_snr_comp.m`

- Measure how the NMSE varies with SNR for matrix completion. Figure 6.b. in paper.

### `rank_completion_plot.m`

- Measure how the NMSE varies with rank for matrix completion. Figure 6.c. in paper.

### `error_comp_q.m`

- Measure how the NMSE varies with `q` for matrix completion. Figure 7 in paper.

### `vb_comparission.m`

- Measures the NMSE of different variational Bayesian methods for matrix reconstruction and varying `alpha = m/pq`. Excluded from paper.

## Auxiliary files

### `transpose_operator.m`

- Generates the transpose operator `T` for `p` times `q` matrices such that `T vec(X) = vec(X')`.

- Call as `T = transpose_operator(p,q)`.


## License and referencing
This source code is licensed under the [MIT][mit] license. If you in any way use this code for research that results in publications, please cite our original article. The following [Bibtex][bibtex] entry can be used.

```
@Article{Sundin2015Submitted,
  Title                    = {Relevance Singular Vector Machine for Low-rank Matrix Reconstruction},
  Author                   = {M. Sundin, C.R. Rojas, M. Jansson and S. Chatterjee},
  Journal                  = {{IEEE} Trans. Sig. Proc.},
  Year                     = {2015},
  Note                     = {Submitted}
}
```

[masundi]: https://www.kth.se/profile/masundi/
[rojas]: https://www.kth.se/profile/crro/
[saikat]: https://www.kth.se/profile/sach/
[mjansson]: https://www.kth.se/profile/janssonm/
[arxiv]: http://arxiv.org/abs/1501.05740
[mit]: http://choosealicense.com/licenses/mit
[bibtex]: http://www.bibtex.org/
[cvx]: http://cvxr.com/cvx/