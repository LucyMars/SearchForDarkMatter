This file outlines the installation process required to run this project. 
The information is taken from the github repositories of XENON1T and Jelle Aalbers and is included here for convinence.

PaX works best in either Linux or mac. I had to use a virtual machine to run Linux; the following website gives good instructions on how to install one:  
https://brb.nci.nih.gov/seqtools/installUbuntu.html 

## Setting Up PaX
This section follows the Linux installation process suggested by the PAX github page - https://github.com/XENON1T/pax (the page also includes information for installing on a Mac).

1) Install Python 3 and Anaconda libraries
  ```
  wget http://repo.continuum.io/archive/Anaconda3-2.4.0-Linux-x86\_64.sh
  bash Anaconda3-2.4.0-Linux-x86_64.sh 
  ```

2) Set up Anaconda libraries
  ```
  export PATH=~/anaconda3/bin:\$PATH 
  conda config --add channels http://conda.anaconda.org/NLeSC
  ```

3) Add additional python packages
  ```
  conda install conda 
  conda create -n pax python=3.4 root=6 numpy scipy=0.18.1 pyqt=4.11 matplotlib pandas cython h5py numba pip python-snappy pytables   scikit-learn rootpy pymongo psutil jupyter dask root_pandas jpeg=8d isl=0.12.2 gmp=5.1.2 glibc=2.12.2 graphviz=2.38.0=4 gsl=1.16 linux-headers=2.6.32 mpc=1.0.1 mpfr=3.1.2 pcre=8.37 python-snappy=0.5 pyopenssl=0.15.1
  ```

4) Activate the environment - THIS MUST BE DONE EVERY TIME YOU USE PAX 
  ```
  source activate pax 
  ```
5) Installing Pax
  ```
  git clone https://github.com/XENON1T/pax.git 
	source activate pax
	cd pax
	python setup.py develop
  ```
  
## Simulate Data - Package Installation
**Wimprates** - https://github.com/JelleAalbers/wimprates

Gives a dark matter energy spectrum - "Differential rates of WIMP-nucleus scattering in the standard halo model, for liquid xenon detectors."
  ```
  pip install wimprates
  ```

**Multihist** - https://github.com/JelleAalbers/multihist

Package that groups numpy's histogram functions into one class which can add new data to existing histograms.
  ```
  pip install multihist 
  ```

**Laidbax** - https://github.com/XENON1T/laidbax

Converts energy spectrum into the a model (i.e. gives the number of photons, number of electrons)
  ```
  git clone https://github.com/XENON1T/laidbax
  cd laidbax
  python setup.py develop
  cd ..
  ```
**Blueice** - https://github.com/JelleAalbers/blueice

"Build Likelihoods Using Efficient Interpolations and monte-Carlo generated Events".
  ```
  git clone https://github.com/JelleAalbers/blueice
  cd blueice
  python setup.py develop
  cd ..
  ```
