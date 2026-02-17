# Real-space topology in quasicrystal Chern mosaics

Code and data accompanying the manuscript:  
https://arxiv.org/abs/2512.18397

This repository provides:
- Core algorithms used in the paper
- Example workflows demonstrating typical usage
- Precomputed Hamiltonian Matrix Product Operators (MPOs) used in the main text

---

## Repository structure

### `TopoTN.jl/src`

Core implementation.

- **`2D_lattices.jl`**  
  Tools for representing 2D tight-binding lattices in the quantics basis.

- **`Hamiltonian.jl`**  
  Utilities for constructing tight-binding Hamiltonians as Matrix Product Operators (MPOs).

- **`KPM_tk.jl`**  
  Kernel Polynomial Method (KPM) toolkit for computing observables via MPO Chebyshev expansions:
  - Density matrix
  - Local density of states (LDoS)

- **`Topology_tk.jl`**  
  Tensor-network toolkit for constructing local topological markers.

---

### `TopoTN.jl/examples`

Example workflows illustrating the tensor-network formalism on simple systems.  
Intended as entry points for new users.

- **`Real_Space_Topology_TNs.ipynb`**  
  Demonstrates how to:
  - Build tensor-network representations of:
    - The π-flux model (main text)
    - The Haldane model (appendix)
  - Compute the density matrix (projector onto occupied states)
  - Construct the Chern operator
  - Visualize:
    - Real-space Chern marker
    - Local density of states (LDoS)

---

## Precomputed MPO Hamiltonians

Ready-to-use Hamiltonians with quasicrystalline modulations used in the manuscript:

- `H_mpo_214_mod10_per320.h5`
- `H_mpo_214_mod8_per320.h5`
- `H_mpo_215_mod8_per640.h5`
- `H_piflux_modpi_mpo.h5`

These can be loaded directly to reproduce results without rebuilding the MPOs.

---

## Installation

The code is written in Julia and depends on the following packages.

### Required packages

Run the following in the Julia REPL before using the repository:

```julia
using Pkg

Pkg.add("LinearAlgebra")   # stdlib (usually already available)
Pkg.add("ITensors")
Pkg.add("NDTensors")
Pkg.add("ITensorMPS")
Pkg.add("Quantics")
Pkg.add("QuanticsTCI")
Pkg.add("QuanticsGrids")
Pkg.add("TCIITensorConversion")
Pkg.add("TensorCrossInterpolation")
Pkg.add("Plots")



