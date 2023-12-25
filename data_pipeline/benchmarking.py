from tqdm import tqdm
import time

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from compute_dist import *

torch.set_default_device('cuda')

def main():
  print(f"cuda is available: {torch.cuda.is_available()}")
  print(f"number of devices: {torch.cuda.device_count()}")
  print(f"device name: {torch.cuda.get_device_name(0)}")

  print("reading nci smiles 250k...", end = "")
  t0 = time.time()
  with open("nci_smiles_250k", "r", encoding='utf-8-sig') as f:
    lines = [line for line in f]
  t1 = time.time()

  num_samples = 10000
  subset = np.random.choice(np.arange(len(lines)), size = (num_samples,), replace = False)
  lines = [lines[i] for i in subset]
  print(f"took {(t1 - t0):.2f} seconds")  

  print("computing minimum bounding box...", end = "")
  t0 = time.time()
  seed = 0
  x_min, x_max, y_min, y_max, z_min, z_max = torch.tensor([0., 0., 0., 0., 0., 0.]) 
  for smi in tqdm(lines):
    try:
      m = gen_conformer(smi, seed)
    except:
      print(f"failed conformer generation on {smi}")
      pass
    x_range, y_range, z_range = get_box(m)
    x_min = torch.minimum(x_range[0], x_min)
    x_max = torch.maximum(x_range[1], x_max)
    y_min = torch.minimum(y_range[0], y_min)
    y_max = torch.maximum(y_range[1], y_max)
    z_min = torch.minimum(z_range[0], z_min)
    z_max = torch.maximum(z_range[1], z_max)
  x_range = torch.tensor([x_min, x_max])
  y_range = torch.tensor([y_min, y_max])
  z_range = torch.tensor([z_min, z_max])
  t1 = time.time()
  print(f"took {(t1 - t0):.2f} seconds")
  print(f"x range: [{x_min:.2f}, {x_max:.2f}]")  
  print(f"y range: [{y_min:.2f}, {y_max:.2f}]")  
  print(f"z range: [{z_min:.2f}, {z_max:.2f}]")

  N = 100
  grid = compute_grid(x_range, y_range, z_range, N)
  
  # do the actual benchmarking
  gen_conformer_times = []
  pos_dict_times = []
  dist_field_times = []
  for smi in tqdm(lines):
    try:
      t0 = time.time()
      m = gen_conformer(smi, seed)
      t1 = time.time()
    except:
      pass
    gen_conformer_times.append(t1 - t0)
    
    t0 = time.time()
    pos_dict = gen_pos_dict(m)
    t1 = time.time()

    pos_dict_times.append(t1 - t0)

    t0 = time.time()
    compute_dist_field(grid, pos_dict)
    t1 = time.time()
    dist_field_times.append(t1 - t0)

  mean_gc_time = np.mean(gen_conformer_times)
  var_gc_time = np.var(gen_conformer_times)
  mean_pd_time = np.mean(pos_dict_times)
  var_pd_time = np.var(pos_dict_times)
  mean_df_time = np.mean(dist_field_times)
  var_df_time = np.var(dist_field_times)

  f = open("benchmarking_results.txt", "w")
  f.write(f"run on a single NVIDIA A40 GPU with {num_samples} molecules from NCI dataset with grid of size {N}x{N}x{N}\n")
  f.write(f"mean time for conformer generation: {mean_gc_time:.4f} +- {var_gc_time:.4f} seconds\n")
  f.write(f"mean time for position dictionary computation: {mean_pd_time:.4f} +- {var_pd_time:.4f} seconds\n")
  f.write(f"mean time for distance field computation: {mean_df_time:.4f} +- {var_df_time:.4f} seconds")
  f.close()
  
    








if __name__ == "__main__":
  main()