<!--![](fenics_topopt_foam_logo.png)-->
<p align="center">
	<img src="./img/fenics_topopt_foam_logo.png" alt="FEniCS TopOpt Foam" width="600"/>
</p>

<p align="center">
  <p align="center"><b>FEniCS TopOpt Foam: Topology optimization combining OpenFOAM<sup>&reg;</sup> and FEniCS/dolfin-adjoint.</b></p>
</p>

Interface for implementing topology optimization with the simulation from OpenFOAM<sup>&reg;</sup> and the automatically derived adjoint model from FEniCS/dolfin-adjoint. The FEniCS TopOpt Foam library is able to automatically generate the meshes and degree-of-freedom mapping between OpenFOAM<sup>&reg;</sup> and FEniCS/dolfin-adjoint.

## 🐾️ How to use

**1.** Enter the OpenFOAM<sup>&reg;</sup> environment, which depends on where OpenFOAM<sup>&reg;</sup> is installed. For example,

```console
[]$ source /opt/OpenFOAM-7/etc/bashrc
```

**2.** Import FEniCS TopOpt Foam in your Python code:

```python
import fenics_topopt_foam
```

**3.** There is an "`examples`" folder alongside the source code, and an auto-generated API documentation in HTML format (with [pdoc3](https://pdoc3.github.io/pdoc/)).

### │ Observations

* Parallelism is implemented in two completely independent levels. This independency is necessary when considering that the mesh partitioning may be different in each platform.

	* **OpenFOAM<sup>&reg;</sup> level**: Parallelism in the OpenFOAM<sup>&reg;</sup> level can be achieved by setting  "`fenics_foam_solver.foam_solver.setToRunInParallel([...])`" inside the code.

	* **FEniCS level**: Parallelism in the FEniCS level can be achieved by changing the way the code is called, by, for example,

```console
[]$ mpiexec -n 2 python my_code.py
```

* FEniCS TopOpt Foam is developed for the "openfoam.org" version of OpenFOAM<sup>&reg;</sup>, which means that it may possibly not work in other OpenFOAM<sup>&reg;</sup> versions (although there is some preliminary work in "`fenics_topopt_foam/utils/foam_information.py`").
* Setting "`surfaceVectorField`", "`surfaceScalarField`", and "`TensorField`" from FEniCS to OpenFOAM<sup>&reg;</sup> is currently not supported (i.e., only "`volVectorField`" and "`volScalarField`" are supported).
* FEniCS TopOpt Foam currently supports only triangular/tetrahedral meshes.

## 📥️ Download

**1.** FEniCS TopOpt Foam is based on [Python 3](https://www.python.org/), relies on the installation in a Linux environment, and requires the installation of the following packages:

 - [OpenFOAM<sup>&reg;</sup> 7.0 (openfoam.org)](https://openfoam.org/download/archive/)
 - [FEniCS 2019.1.0](https://fenicsproject.org/download/)
 - [dolfin-adjoint 2019.1.0](http://www.dolfin-adjoint.org/en/latest/download/index.html)
 - [meshio](https://github.com/nschloe/meshio)
 - [Matplotlib](https://matplotlib.org/)
 - [mpi4py](https://github.com/mpi4py/mpi4py)

**2.** After installing the dependencies, FEniCS TopOpt Foam may be installed through [pip](https://pypi.org/project/pip/):

```console
[]$ pip install --user git+https://github.com/diego-hayashi/fenics_topopt_foam.git@main
```

## 📑 Citation

Please cite the following paper in any publication in which you find FEniCS TopOpt Foam to be useful.

<!--![]
Alonso, D. H., Garcia Rodriguez, L. F., Silva, E. C. N. (2021) **Flexible framework for fluid topology optimization with OpenFOAM<sup>&reg;</sup> and finite element-based high-level discrete adjoint method (FEniCS/dolfin-adjoint)**. Structural and Multidisciplinary Optimization TBD:TBD-TBD
-->
Alonso, D. H., Garcia Rodriguez, L. F., Silva, E. C. N. (2021) **Flexible framework for fluid topology optimization with OpenFOAM<sup>&reg;</sup> and finite element-based high-level discrete adjoint method (FEniCS/dolfin-adjoint)**. Structural and Multidisciplinary Optimization 64, 4409-4440. [https://doi.org/10.1007/s00158-021-03061-4](https://doi.org/10.1007/s00158-021-03061-4)

## 📕️ License

FEniCS TopOpt Foam is licensed under the GNU General Public License (GPL), version 3.

> FEniCS TopOpt Foam is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
> 
> FEniCS TopOpt Foam is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
> 
> You should have received a copy of the GNU General Public License along with FEniCS TopOpt Foam. If not, see <https://www.gnu.org/licenses/>.

<sub>**Disclaimer:** The development of FEniCS TopOpt Foam is neither related nor tied in any way to the development of OpenFOAM<sup>&reg;</sup> and FEniCS/dolfin-adjoint.

<!-- However, there may be updates in FEniCS TopOpt Foam for more recent versions of them. </sub>-->

