'''
compute_dist.py

Functions for pipeline of blackbox that computes ground truth distance field for 3D molecular 
  conformers.
'''

from rdkit import Chem
from rdkit.Chem.AllChem import EmbedMolecule
from rdkit.Chem import rdDistGeom
import torch
from collections import defaultdict
import numpy as np
import json

# global constant of atomic symbols
with open("atomic_symbols.json", "r") as f:
  _ATOMIC_SYMBOLS_ = json.load(f)

def gen_conformer(mol_smiles, seed):
  '''
  Generate a 3D conformer of a molecule from SMILES string.

  INPUTS:
    mol_smiles: str, canonical smiles string for molecule
    seed: int: for seeding RNG of conformer generation
  OUTPUTS:
    mol: rdkit.Chem.rdchem.Mol object, with 3D coordinates computed.
  '''
  mol = Chem.MolFromSmiles(mol_smiles)
  mol = Chem.AddHs(mol)
  EmbedMolecule(mol, randomSeed = seed)
  Chem.rdMolTransforms.CanonicalizeMol(mol) # canonicalize rigid orientation
  return mol


def gen_pos_dict(mol):
  '''
  Generate a dictionary whose keys are either atomic symbols or atomic numbers, and whose values
    are 2D torch tensors of shape [n_i, 3]. The [j, :] entry of the ith numpy array are the 3D
    Euclidean coordinates of the jth occurence of molecule i.

  INPUTS:
    mol: rdkit.Chem.rdchem.Mol object, with 3D coordinates computed; the output of gen_conformer()
  OUTPUT:
    pos_dict: dict, whose keys are integers representing atomic numbers, and whose values are 
      [n_i, 3]-shaped tensors of atom positions.
  '''
  keys = list(atom.GetAtomicNum() for atom in mol.GetAtoms())

  positions = mol.GetConformer().GetPositions()
  pos_dict = defaultdict(list)
  for key, pos in zip(keys, positions):
    pos_dict[key].append(pos)

  for key in pos_dict:
    pos_dict[key] = torch.tensor(np.array(pos_dict[key]))
  
  return pos_dict


def compute_grid(x_range, y_range, z_range, N):
  '''
  Compute grid of points in a 3D box.

  INPUTS:
    x_range: (low, high) for x range
    y_range: (low, high) for y range
    z_range: (low, high) for z range
    N: int, density of grid points along each axis

  OUTPUTS:
    grid: (N, N, N, 3)-shaped tensor of points in R3.
  '''
  xs = torch.linspace(x_range[0], x_range[1], N)
  ys = torch.linspace(y_range[0], y_range[1], N)
  zs = torch.linspace(z_range[0], z_range[1], N)
  
  xx, yy, zz = torch.meshgrid(xs, ys, zs, indexing="ij")
  grid = torch.stack((xx, yy, zz))

  grid = grid.permute(1, 2, 3, 0)

  return grid


def compute_min_dist(x, atom_positions):
  '''
  Compute the distance from a point x to the closest copy of an atom, given its position.

  INPUTS:
    x: (3,)-shaped tensor from which we want to compute distances to the atoms in the molecule
    atom_positions: (n, 3)-shaped tensor of positions of a particular element; value of pos_dict

  OUTPUTS:
    min_dist: float, equal to min_{atom locations}\|x - atom location\|
  '''
  dists = torch.linalg.norm(atom_positions - x, axis = 1)
  min_dist = torch.min(dists)
  return min_dist


compute_min_dist_grid = torch.vmap(
  torch.vmap(
    torch.vmap(
      compute_min_dist, in_dims = (0, None),
    ), in_dims = (0, None)
  ), in_dims = (0, None)
)
'''
Compute the distances from each pointin a grid to the cloest copy of an atom, given its position.

INPUTS:
  x_grid: (N, N, N, 3)-shaped tensor of grid points along box where we wish to compute distance
    field
  atom_positions: (n, 3)-shaped tensor of positions of a particular element; value of pos_dict

OUTPUTS:
  min_dists: (N, N, N)-shaped tensor, equal to min_{atom locations}\|x - atom location\| for all x
    in x_grid
'''

def compute_dist_field(grid, pos_dict):
  '''
  Given a point x in Euclidean R3, compute a vector of length num_atoms where the ith entry is the
    euclidean L2 distance to the nearest instance of the ith element in the molecule. Elements are
    ordered by ascending atomic number.

  INPUTS:
    grid: (N, N, N, 3)-shaped tensor of grid points along box where we wish to compute distance
      field
    pos_dict: dict, whose keys are either integers representing atomic numbers or strings 
      representing chemical symbols, and whose values are [n_i, 3]-shaped tensors of atom positions;
      output of gen_pos_dict()

  OUTPUTS:
    dists: tensor of size (N, N, N, len(pos_dict)); the [i, j, k, l] element is the euclidean 
      distance of grid[copy of the ith element 
    elems: list of size len(pos_dict); the elements in the atom, listed in ascending atomic number
  '''
  N = grid.shape[0]

  elems = np.sort(np.array(list(pos_dict.keys())))
  dist_field = torch.zeros((N, N, N, elems.size,))
  for idx, elem in enumerate(elems):
    elem_dist_field = compute_min_dist_grid(grid, pos_dict[elem])
    dist_field[:, :, :, idx] = elem_dist_field

  return dist_field, elems








  


    
