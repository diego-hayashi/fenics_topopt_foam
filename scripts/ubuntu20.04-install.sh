# FEniCS + OpenFoam 7 in Ubuntu 20.04

sudo apt update

# FEniCS -> https://fenicsproject.org/download/archive/
sudo apt install software-properties-common
sudo add-apt-repository ppa:fenics-packages/fenics
sudo apt update
sudo apt install fenics

pip install --upgrade pip

pip install "mpi4py<4.0.0"

# dolfin-adjoint -> http://www.dolfin-adjoint.org/
pip install dolfin-adjoint

sudo mv /usr/lib/python3/dist-packages/ufl \
        /usr/lib/python3/dist-packages/orig.ufl
sudo mv /usr/lib/python3/dist-packages/fenics_ufl-2022.1.0.egg-info \
        /usr/lib/python3/dist-packages/orig.fenics_ufl-2022.1.0.egg-info

# IPOPT -> https://github.com/mechmotum/cyipopt/blob/master/docs/source/install.rst
sudo apt install build-essential pkg-config python3-pip python3-dev cython3 python3-numpy coinor-libipopt1v5 coinor-libipopt-dev
pip install "cyipopt<1.3.0"

pip install "numpy<2.0.0"

# OpenFOAM 7
sudo sh -c "wget -O - https://dl.openfoam.org/gpg.key | apt-key add -"
sudo add-apt-repository http://dl.openfoam.org/ubuntu
sudo apt update
sudo apt install openfoam7
source /opt/openfoam7/etc/bashrc

# FEniCS TopOpt Foam
pip install --user git+https://github.com/diego-hayashi/fenics_topopt_foam.git@main

ln -s /usr/bin/python3 ~/bin/python

