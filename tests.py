import os
os.environ['CUDA_VISIBLE_DEVICES'] = "-1"

import unittest
import torch
torch.manual_seed(0)
torch.set_default_dtype(torch.float64)

import sys
sys.path.insert(0, "./data_pipeline")
from compute_dist import * 

import unittest

_TEST_SMILES_ = [
"CCCC", # butane
"OCC", # ethanol
"CN1CCC[C@H]1C2=CN=CC=C2", #nicotine
"CC(=O)NC1=CC=C(C=C1)O", # acetominophen
"CN1C=NC2=C1C(=O)N(C(=O)N2C)C", # caffeiene
"CC(=O)OC1=CC=CC=C1C(=O)O", # aspirin
"CC(C)CC1=CC=C(C=C1)C(C)C(=O)O", # ibuprofen
"C1=CC2=C(C=C1O)C(=CN2)CCN ", # serotinin
"CN1CC[C@]23C4=C5C=CC(O)=C4O[C@H]2[C@@H](O)C=C[C@H]3[C@H]1C5", # morphine
"C1=CC(=C(C=C1CCN)O)O" # dopamine
] # ten valid SMILES strings

_NUM_TRIALS_ = 10

class test_misc(unittest.TestCase):
  '''
  Test miscellaneous stuff
  '''
  def test_valid_smiles(self):
    '''
    Check that our test SMILES strings are all valid
    '''
    for smi in _TEST_SMILES_:
      m = Chem.MolFromSmiles(smi, sanitize = False)
      self.assertTrue(m is not None)

class test_distance_computations(unittest.TestCase):
  '''
  Test functions in data_pipeline/compute_dist.py
  '''
  def test_gen_conformer(self):
    '''
    Check that generated conformers has 3D coordinates.
    '''
    for seed, smi in enumerate(_TEST_SMILES_):
      m = gen_conformer(smi, seed)
      self.assertTrue(m.GetConformer().Is3D())

  def test_gen_pos_dict(self):
    '''
    Check that position dictionary generated by gen_pos_dict() matches with manual, loopy naive 
      construction.
    '''
    for seed, smi in enumerate(_TEST_SMILES_):
      m = gen_conformer(smi, seed)
      pos_dict = gen_pos_dict(m)

      positions = m.GetConformer().GetPositions()
      for idx, atom in enumerate(m.GetAtoms()):
        key = atom.GetAtomicNum()
        value = positions[idx]

        # compute distance of atom position from each row of pos_dict values
        norms = torch.linalg.norm(pos_dict[key] - value, axis = 1)
        self.assertTrue(np.any(np.isclose(0.0, norms, rtol = 0, atol = 1e-7)))

  def test_compute_min_dist_grid(self):
    '''
    Check compute_min_dist_grid() against manual computation.
    '''
    for seed, smi in enumerate(_TEST_SMILES_):
      x_range = torch.sort((torch.rand((2,)) - 0.5) * 2).values
      y_range = torch.sort((torch.rand((2,)) - 0.5) * 2).values
      z_range = torch.sort((torch.rand((2,)) - 0.5) * 2).values

      N = torch.randint(5, 10, ())

      grid = compute_grid(x_range, y_range, z_range, N)
      m = gen_conformer(smi, seed)
      pos_dict = gen_pos_dict(m)

      elem = np.random.choice(list(pos_dict.keys()))

      atom_positions = pos_dict[elem]

      min_dist_grid = compute_min_dist_grid(grid, atom_positions)

      min_dist_grid_naive = np.zeros(shape = (N,N,N))

      for i in range(N):
        for j in range(N):
          for k in range(N):
            dists = []
            for atom_position in atom_positions:
              dists.append(np.linalg.norm(atom_position - grid[i, j, k]))
            min_dist_grid_naive[i, j, k] = np.min(np.array(dists))
    
      self.assertTrue(np.allclose(min_dist_grid, min_dist_grid_naive))

  def test_compute_grid(self):
    '''
    Check compute_grid() against naive computation.
    '''
    for _ in range(_NUM_TRIALS_):
      x_range = torch.sort((torch.rand((2,)) - 0.5) * 2).values
      y_range = torch.sort((torch.rand((2,)) - 0.5) * 2).values
      z_range = torch.sort((torch.rand((2,)) - 0.5) * 2).values

      N = torch.randint(5, 10, ())

      grid = compute_grid(x_range, y_range, z_range, N)

      xs = np.linspace(x_range[0], x_range[1], N)
      ys = np.linspace(y_range[0], y_range[1], N)
      zs = np.linspace(z_range[0], z_range[1], N)

      for i in range(N):
        for j in range(N):
          for k in range(N):
            self.assertTrue(np.allclose(np.array(grid[i, j, k]), np.array([xs[i], ys[j], zs[k]])))

if __name__ == "__main__":
  unittest.main(verbosity = 1)  