import os
os.environ['CUDA_VISIBLE_DEVICES'] = "-1"

import unittest
import torch
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

_NUM_TRIALS_ = 5

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
  Tet functions in data_pipeline/compute_dist.py
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
    Check that position dictionary generated by gen_pos_dict() matches with manual,
      loopy naive construction.
    '''
    for seed, smi in enumerate(_TEST_SMILES_):
      if seed % 2 == 1:
        use_symb = True
      else:
        use_symb = False
      m = gen_conformer(smi, seed)
      pos_dict = gen_pos_dict(m, use_symb = use_symb)

      positions = m.GetConformer().GetPositions()
      for idx, atom in enumerate(m.GetAtoms()):
        if use_symb:
          key = atom.GetSymbol()
        else:
          key = atom.GetAtomicNum()
        value = positions[idx]

        # compute distance of atom position from each row of pos_dict values
        norms = torch.linalg.norm(pos_dict[key] - value, axis = 1)
        self.assertTrue(np.any(np.isclose(0.0, norms, rtol = 0, atol = 1e-7)))

  
        


      
        

        




if __name__ == "__main__":
  unittest.main(verbosity = 1)  