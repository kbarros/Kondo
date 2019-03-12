Kondo
=====

What it does
------------

This code builds upon the [FastKPM](https://github.com/kbarros/FastKPM) library to enable fast simulations of the Kondo lattice model (KLM).

The KLM is a model of itinerant electrons interacting with classical magnetic moments localized on each site. After "integrating out" the electrons, the effective interactions between magnetic moments can become long-range and many-body. For example, at weak coupling, the effective interactions are of the [RKKY type](https://en.wikipedia.org/wiki/RKKY_interaction). However, this code makes no assumptions about the coupling strength or the interaction range.

With GPU acceleration enabled, this code readily enables simulating the dynamics of 10k interacting magnetic moments, or even more.

Building
--------

Building is handled with CMake.

The [FastKPM](https://github.com/kbarros/FastKPM) library should already be compiled and installed. Kondo will then automatically link to it.

This package includes tests. Please start with `bin/test_kpm` and `bin/test_kondo` to make sure everything is installed correctly.

Usage
-----

Still needs to be documented...

Applications
------------

An early version of this code was presented in:
* [_Efficient Langevin simulation of coupled classical fields and fermions_](https://doi.org/10.1103/PhysRevB.88.235101), Barros et al., PRB (2013).

However, the method has evolved significantly since then! Two important improvements are [gradient-based probing](https://arxiv.org/abs/1711.10570) and [more accurate](https://arxiv.org/abs/1002.1801) integration of the [magnetic dynamics](https://en.wikipedia.org/wiki/Landau%E2%80%93Lifshitz%E2%80%93Gilbert_equation).

Subsequent versions of the code have been used to produce the following research papers:
* [_Exotic magnetic orderings in the kagome Kondo-lattice model_](https://doi.org/10.1103/PhysRevB.90.245119), Barros et al., PRB (2014)
* [_Vortex Crystals with Chiral Stripes in Itinerant Magnets_](https://doi.org/10.7566/JPSJ.85.103703), Ozawa et al., JPSJ (2016)
* [_Resistivity Minimum in Highly Frustrated Itinerant Magnets_](https://doi.org/10.1103/PhysRevLett.117.206601), Wang et al., PRL (2016)
* [_Zero-Field Skyrmions with a High Topological Number in Itinerant Magnets_](https://doi.org/10.1103/PhysRevLett.118.147205), Ozawa et al., PRL (2017)
* [_Shape of magnetic domain walls formed by coupling to mobile charges_](https://doi.org/10.1103/PhysRevB.96.094417), Ozawa et al., PRB (2017)
* [_Simulated floating zone method_](https://doi.org/10.1088/1742-6596/807/10/102005), Ozawa et al., JPCS (2017)
* [_Semiclassical dynamics of spin density waves_](https://doi.org/10.1103/PhysRevB.97.035120), Chern et al., PRB (2018)
* [_Multiple-Q magnetic orders in Rashba-Dresselhaus metals_](https://doi.org/10.1103/PhysRevB.98.224406), Okada et al., PRB (2018)
* [_Nonequilibrium dynamics of superconductivity in the attractive Hubbard model_](https://doi.org/10.1103/PhysRevB.99.035162), Chern et al., PRB (2019)

Please let us know if you use this code in your work!

Citing
------

If you find this code useful, please cite our gradient-based probing paper:

```
@article{doi:10.1063/1.5017741,
author = {Wang, Zhentao and Chern, Gia-Wei and Batista, Cristian D. and Barros, Kipton},
title = {Gradient-based stochastic estimation of the density matrix},
journal = {J. Chem. Phys.},
volume = {148},
pages = {094107},
year = {2018},
}
```
