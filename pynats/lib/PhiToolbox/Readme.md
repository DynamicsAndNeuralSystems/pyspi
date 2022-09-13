# Practical &Phi; Toolbox for Integrated Information Analysis
Jun Kitazono and Masafumi Oizumi (The University of Tokyo) </br>
Email: c-kitazono@g.ecc.u-tokyo.ac.jp, c-oizumi@g.ecc.u-tokyo.ac.jp

This toolbox provides MATLAB codes for end-to-end computation in "practical versions" of integrated information theory (IIT): 
1. Computing practical statistics of integrated information (&Phi;).
2. Searching for minimum information partitions (MIPs).
3. Searching for complexes.

These are key concepts in IIT and can be generally utilized for analyzing stochastic systems, i.e., evaluating how much information is integrated in a system, finding the optimal partition and the cores of a system. </br>
In general, these computations take a large amount of time, which has hindered the application of IIT to real data. This toolbox provides fast algorithms, enabling us to analyze large systems in a practical amount of time. This toolbox is an update of our previous version of the toolbox available at figshare (doi:10.6084/m9.figshare.3203326). In this new version, the algorithms for searching for complexes are newly implemented.  

 
## General description
 
### Computation of practical statistics of &Phi;
The codes for computing practical statistics of integrated information (&Phi;), namely, Mutual Information (Tononi, 2004), Stochastic Interaction (Ay, 2001, 2015; Barrett & Seth, 2011), integrated information based on mismatched decoding &Phi;<sub>\*</sub> [1] and geometric integrated information &Phi;<sub>G</sub> [2]. Integrated information quantifies the amount of information that is integrated within a system. For quantifying integrated information, we need to assume what distribution data obeys. Two options are available for a probability distribution with which integrated information is computed: a Gaussian distribution and a discrete distribution. The relations among the computation times of these statistics are summarised as </br>
MI (Gauss) < SI (Gauss) << &Phi;<sub>\*</sub> (Gauss) << &Phi;<sub>G</sub> (Gauss) << MI (dis) < SI (dis) << &Phi;<sub>\*</sub> (dis). </br>
The computation of &Phi;<sub>G</sub> for discrete distributions is not implemented. 
In general, the computation for Gaussian distributions is much faster than that for discrete distributions. The computation of &Phi;<sub>\*</sub> and &Phi;<sub>G</sub> takes more time than that for MI and SI because the computation of &Phi;<sub>\*</sub> and &Phi;<sub>G</sub> requires solving continuous optimization problems. Please look at “demo_phi_Gauss.m” and “demo_phi_dis.m” to see how to use the core functions for &Phi; computation.

 
### Search for minimum information partitions (MIPs)
The codes for searching for the minimum information partition (MIP) (see Tononi, 2008, Biol Bull for example). Two algorithms for the MIP search are provided, namely, an exhaustive search and Queyranne’s algorithm (Queyranne, 1998). An exhaustive search is applicable for all the statistics of integrated information but it takes a large amount of computation time. If the number of elements in a system is larger than several dozen, the computation is practically impossible. Queyranne's algorithm can exactly and effectively find the MIP only when mutual information is used as a statistic of integrated information [3]. It enables us to find the MIP in a relatively large system (consisting of several hundred elements) in a practical amount of time when the probability distribution is Gaussian. Queyranne's algorithm can be also used as an approximate algorithm for finding the MIP even when other statistics are used [4]. Please look at “demo_MIP_Gauss.m” and “demo_MIP_dis.m” to see how to use the core functions for MIP search.
 
### Search for complexes
The codes for searching for complexes (See Balduzzi & Tononi, 2008, PLoS Comp. Biol. for example). Two algorithms are provided, namely, an exhaustive search and Hierarchical Partitioning for Complex search (HPC), which we proposed in [5]. An exhaustive search is applicable for all the statistics of integrated information but it takes an extremely large amount of computation time. As a remedy for the computational intractability, HPC can exactly and effectively find complexes when mutual information is used as a statistic of integrated information [5]. It enables us to find complexes in a large system (consisting of several hundred elements) in a practical amount of time. HPC does not necessarily find all of the complexes or well-approximated complexes when the statistics other than MI are used. Please look at “demo_Complex_Gauss.m” and “demo_Complex_dis.m” to see how to use the core functions for complex search. Please also look at “demos_HPC” folder containing codes for reproducing the simulations in [5].
 
You can freely use this toolbox at your own risk. Please cite this toolbox and the papers listed below when the toolbox is used for your publication. Comments, bug reports, and proposed improvements are always welcome. 
 
 
### Misc.
This toolbox uses 
- “colorcet.m” written by Peter Kovesi, which provides perceptually uniform color maps. See the link below for more details. 
https://peterkovesi.com/projects/colourmaps/ 
 
- “minFunc” written by Mark Schmidt, which is needed for solving unconstrained optimization. See the link below for more details. 
http://www.cs.ubc.ca/~schmidtm/Software/minFunc.html
 
 
## Troubleshooting
- *Invalid MEX File Errors*. In the case you get "Invalid MEX-file" error messages related to minFunc, please compile minFunc files by executing mexAll.m in PhiToolbox/tools/minFunc_2012 or please set Options.useMex = 0 in phi_G_Gauss-LBFGS.m, phi_star_dis.m and phi_star_Gauss.m.
 
 
## Acknowledgment
We thank Shohei Hidaka (JAIST) for providing the codes for Queyranne’s algorithm. We thank Yuma Aoki, Ayaka Kato, Genji Kawakita, Takeru Kimura, Daiki Kiyooka and Kaio Misawa for testing this toolbox for usability.
 
## References
[1] Oizumi, M., Amari, S, Yanagawa, T., Fujii, N., & Tsuchiya, N. (2016). Measuring integrated information from the decoding perspective. PLoS Comput Biol, 12(1), e1004654. http://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1004654
 
[2] Oizumi, M., Tsuchiya, N., & Amari, S. (2016). Unified framework for information integration based on information geometry. Proceedings of the National Academy of Sciences, 113(51), 14817-14822. http://www.pnas.org/content/113/51/14817.short
 
[3] Hidaka, S., Oizumi, M. (2017). Fast and exact search for the partition with minimal information loss. arXiv, 1708.01444. https://arxiv.org/abs/1708.01444
 
[4] Kitazono, J., Kanai, R., Oizumi, M. (2018). Efficient algorithms for searching the minimum information partition in integrated information theory. Entropy, 20, 173.
http://www.mdpi.com/1099-4300/20/3/173
 
[5] Kitazono, J., Kanai, R., Oizumi, M. (2020). Efficient search for informational cores in complex systems: Application to brain networks. bioRxiv.
 
These papers are in the "papers" folder.





