# energy 1.7-7

*  User level changes:
     - dcorT.test replaces dcor.ttest, now deprecated.
     - dcorT replaces dcor.t, now deprecated.
     - edist method "discoF" removed
     
*  Internal changes     
     - BCDCOR function (used in the high dim. dcorT test) has been revised. 
     - edist method "discoB" correction
     - changes for compatibility with R 4.0.0

# energy 1.7-6

*  User level changes:
     - normality.test (new) implements the energy test of univariate 
       normality based on the null limit distribution for the composite 
       hypothesis (estimated parameters). 
     - dataset EVnormal (new) of eigenvalues for energy test of normality.
     - mvnorm.test replaces mvnorm.etest, and mvnorm.etest now is a
       wrapper for mvnorm.test. 

# energy 1.7-5

*  User level changes:
     - kgroups: (new) implements energy clustering for a specified 
       number k classes by energy distance criterion, analogous to the k 
       classes of the k-means algorithm.
     - dcov2d and dcor2d: (new) O(n log n) methods to compute the
       U or V statistics for real x and y 
     - sortrank() function added (a utility)
     
*  Internal changes:
     - B-tree.cpp: Btree_sum and other internal functions
       implement binary tree search for faster O(n log n)
       calculation of paired distances in dcov2d
     - kgroups.cpp: Rcpp implementation of k-groups algorithm
     - energy.hclust implementation: replaced C++ code with call 
       to stats::hclust; since R > 3.0.3 it is now equivalent for 
       alpha = 1 with method = "ward.D". Input and return value 
       unchanged except heights from hclust are half. 

# energy 1.7-4

*  User level changes
     - disco: handle the case when the user argument x is dist with
       conflicting argument distance=FALSE
     - dcor.t and dcor.ttest: handle the cases when class of argument x or y
       conflicts with the distance argument
     - Split manual page of dcovU into two files.
     - indep.etest and indep.e removed now Defunct (were Deprecated since
       Version 1.1-0, 2008-04-07; replaced by indep.test).

*  Internal changes
     - BCDCOR: handle the cases when class of argument x or y conflicts with
       the distance argument

# energy 1.7-2
 

*  User level changes
     -  Provided new dcor.test function, similar to dcov.test but using the
        distance correlation as the test statistic.
     -  Number of replicates R for Monte Carlo and permutation tests now matches
        the argument of the boot::boot function (no default value, user must specify).
     -  If user runs a test with 0 replicates, p-value printed is NA
*  Internal changes
     -  energy_init.c added for registering routines

# energy 1.7-0
 

*  Partial Distance Correlation statistics and tests added
     - pdcov, pdcor, pdcov.test, pdcor.test
     - dcovU: unbiased estimator of distance covariance
     - bcdcor: bias corrected distance correlation
     - Ucenter, Dcenter, U_center, D_center: double-centering and U-centering utilities
     - U_product: inner product in U-centered Hilbert space

*  updated NAMESPACE and DESCRIPTION imports, etc.
*  revised package Title and Description in DESCRIPTION
*  package now links to Rcpp
*  mvnorm c code ported to c++ (mvnorm.cpp); corresponding changes in Emvnorm.R
*  syntax for bcdcor: "distance" argument removed, now argument can optionally
     be a dist object
*  syntax for energy.hclust: first argument must now be a dist object
*  default number of replicates R in tests: for all tests, R now defaults to 0
     or R has no default value.

# energy 1.6.2
 

*  inserted GetRNGstate() .. PutRNGState around repl.
     loop in dcov.c.

# energy 1.6.1
 

*  replace Depends with Imports in DESCRIPTION file

# energy 1.6.0
 

*  implementation of high-dim distance correlation t-test
     introduced in JMVA Volume 117, pp. 193-213 (2013).
*  new functions dcor.t, dcor.ttest in dcorT.R
*  minor changes to tidy other code in dcov.R
*  removed unused internal function .dcov.test

# energy 1.5.0
 

*  NAMESPACE: insert UseDynLib; remove zzz.R, .First.Lib()

# energy 1.4-0
 

*  NAMESPACE added.
*  (dcov.c, Eindep.c) Unused N was removed.
*  (dcov.c) In case dcov=0, bypass the unnecessary loop
	   that generates replicates (in dCOVtest and dCovTest).
	   In this case dcor=0 and test is not significant.
	   (dcov=0 if one of the samples is constant.)
*  (Eqdist.R) in eqdist.e and eqdist.etest, method="disco"
	   is replaced by two options: "discoB" (between sample
	   components) and "discoF" (disco F ratio).
*  (disco.R) Added disco.between and internal functions
	   that compute the disco between-sample component and
	   corresponding test.
*  (utilities.c) In permute function replaced rand_unif
	   with runif.
*  (energy.c) In ksampleEtest the pval computation
	   changed from ek/B to (ek+1)/(B+1) as it should be for
	   a permutation test, and unneeded int* n removed.

# energy 1.3-0
 

*  In distance correlation, distance covariance functions
	   (dcov, dcor, DCOR) and dcov.test, arguments x and y can now
	   optionally be distance objects (result of dist function or
	   as.dist). Matrices x and y will always be treated as data.

*  Functions in dcov.c and utilities.c were modified to support
	   arguments that are distances rather than data. In utilities.c
	   the index_distance function changed. In dcov.c there are many
	   changes. Most importantly for the exported objects, there is
	   now an extra required parameter in the dims argument passed
	   from R. In dCOVtest dims must be a vector c(n, p, q, dst, R)
	   where n is sample size, p and q are dimensions of x and y,
	   dst is logical (TRUE if distances) and R is number of replicates.
	   For dCOV dims must be c(n, p, q, dst).

# energy 1.2-0
 

*  disco (distance components) added for one-way layout.
*  A method argument was added to ksample.e, eqdist.e, and
	   eqdist.etest, method = c("original", "disco").
*  A method argument was added to edist, which summarizes cluster
     distances in a table:
         method = c("cluster","discoB","discoF"))

