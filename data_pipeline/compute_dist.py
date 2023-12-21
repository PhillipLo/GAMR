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

def gen_pos_dict(mol, use_symb = False):
  '''
  Generate a dictionary whose keys are either atomic symbols or atomic numbers, and whose values
    are 2D torch tensors of shape [n_i, 3]. The [j, :] entry of the ith numpy array are the 3D
    Euclidean coordinates of the jth occurence of molecule i.

  INPUTS:
    mol: rdkit.Chem.rdchem.Mol object, with 3D coordinates computed; the output of gen_conformer()
    use_symb: bool, if True then dict keys are atomc symbols (C, F, N, etc.). If False, then 
      dict keys are atomic numbers (6, 9, 7, etc.)

  OUTPUT:
    pos_dict: dict, whose keys are either integers representing atomic numbers or strings 
      representing chemical symbols, and whose values are [n_i, 3]-shaped tensors of atom positions.
  '''
  if use_symb:
    keys = list(atom.GetSymbol() for atom in mol.GetAtoms())
  else:
    keys = list(atom.GetAtomicNum() for atom in mol.GetAtoms())

  positions = mol.GetConformer().GetPositions()
  pos_dict = defaultdict(list)
  for key, pos in zip(keys, positions):
    pos_dict[key].append(pos)

  for key in pos_dict:
    pos_dict[key] = torch.tensor(np.array(pos_dict[key]))
  
  return pos_dict
    
