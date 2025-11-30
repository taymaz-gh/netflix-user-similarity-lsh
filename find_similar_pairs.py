# Imports
import time
import numpy as np
from scipy.sparse import csr_matrix
from tqdm import tqdm
from itertools import combinations
import os

# Configurations
DATA_PATH='./user_movie_rating.npy' 
NUM_PERMUTATIONS=80
MAX_BUCKET_SIZE=100


def build_csr(data):
  """
  Converts the data into a sparse matrix.
  CSR format was chosen for efficiency, since it facilitates row-wise operations (handy for permutations)
  """
  print("Building a sparse data matrix for efficiency...")
  users=data[:,0].astype(np.int32) # Convert to int32 for memory efficiency
  movies=data[:,1].astype(np.int32)
  reviews_placehold = np.ones_like(movies, dtype=bool) # Here, bool is most efficient
  # We shape the CSR to have movies in the rows and users in the columns, since the aim is to compare users.
  # The dtype of the csr is set to bool to make reviews binary (since we only care about a movie being reviewed/not)
  csr = csr_matrix(
      (reviews_placehold, (movies-1, users-1)), # -1 to go from ID's to indexes
      shape=(movies.max(),users.max()),
      dtype=bool)
  print("Built sparse data matrix")
  return csr

def compute_signature_m(csr):
  """
  Computes the signature matrix M.
  The rows of the csr matrix are randomly permuted. For each permutation,
  the index of the first 1 ("True" in our binary case) is found for each user.
  """
  print("Computing the MinHash signature matrix...")
  print(f"Using {NUM_PERMUTATIONS} permutations")

  # Shape of the signature matrix M will be num permutations x num columns (users), as in the lecture. Preallocate:
  M=np.zeros((NUM_PERMUTATIONS, csr.shape[1]),dtype=np.int32)
  # Set the seed. Running the script with given seed will produce the same results (same pairs will be found)
  rng = np.random.default_rng(SEED)
  # For each permutation
  for perm in tqdm(range(NUM_PERMUTATIONS)):
    # Get the permuted indexes
    permuted_indexes = rng.permutation(csr.shape[0]).astype(np.int32)
    # Permute the matrix
    csr_permuted=csr[permuted_indexes,:]
    # For each permutation, get the index at which the first 1 occurs
    M[perm,:] = csr_permuted.argmax(axis=0) 

  print("MinHash signature matrix computed")
  return M

def find_b_r():
  """
  Finds the optimal values of bands and rows.
  The optimization is done by maximizing the function: 1 - (1-0.5^r)^b (from the lecture). It represents the probability of
  detecting a pair that has a similarity of 50%. This is the same as minimizing the number of false negatives (1-0.5^r)^b.
  r and b are also subject to constraints for feasibility (b*r can't be bigger that the dataset),
  coverage (b*r have to cover at least 90% of the dataset), and efficiency (r>3, otherwise r=1 will always give best result).
  """
  print(f"Finding best b and r values given {NUM_PERMUTATIONS} permutations...")
  best_prob_of_detect=0
  opt_r, opt_b=0,0
  for b in np.arange(1,NUM_PERMUTATIONS):
    for r in np.arange(1,NUM_PERMUTATIONS):
      # Constraints
      if (((b*r)<=NUM_PERMUTATIONS) and ((b*r)>0.9*NUM_PERMUTATIONS) and (r>3)):
        prob_of_detect=1-(1-0.5**r)**b
        if prob_of_detect>best_prob_of_detect:
          best_prob_of_detect=prob_of_detect
          opt_r=r
          opt_b=b
  print(f"Highest probability of detection {best_prob_of_detect:.2}, with r {opt_r} and b {opt_b}")
  print(f"Corresponding false negative rate {(1-0.5**opt_r)**opt_b:.2}, with r {opt_r} and b {opt_b}")
  return opt_r, opt_b

def LSH(M, opt_r, opt_b):
  """
  Performs LSH to find candidate pairs.
  Done by dividing the signature matrix in the given number of bands (containing r rows),
  hashing the chunks corresponding to each user in each band, and assigning the results in buckets per band.
  The number of buckets is initialized to the number of users; changing the number of buckets did not lead to big changes in collisions.
  """

  print(f"Performing LSH, with a cap on {MAX_BUCKET_SIZE} collisions per bucket...")

  # Hashing function to be applied in vectorized way
  def hashing_func(chunk):
    return hash(tuple(chunk)) % num_buckets

  all_candidates=set() # A set to store all candidate pairs from all buckets; helps to prevent duplicates
  num_buckets=M.shape[1] # The number of buckets defined as the number of users
  for b in tqdm(range(opt_b)): # For each band
    # Retrieve a single band
    band=M[opt_r*b:opt_r*(b+1)]  # Logic: we want rows [0:4] ; [4:8]; which is same as [r*0:r*1]; [r*1:r*2]; or [r*b:r*(b+1)].

    # Create a new bucket dictionary
    bucket_dict={key:[] for key in range(num_buckets)}
    bucket_ids=np.apply_along_axis(hashing_func, 0, band) # Apply hash function to all columns in the band
    # Fill the buckets
    for user, bucket_id in enumerate(bucket_ids):
      bucket_dict[bucket_id].append(user)

    # Go through the buckets
    for vals in bucket_dict.values():
      # Skip if we have less than 2 users or more than 100 (to prevent expensive computation)
      if ((len(vals)<2) or (len(vals)>MAX_BUCKET_SIZE)):
        continue
      # If we have 2 values (2 users only), add them to candidates straight away
      elif len(vals)==2:
        all_candidates.add(tuple(sorted(vals))) # Sorted, to make sure duplicates will not be included in the set
      # If more than 2 users in a bucket: add all possible combinations
      elif (len(vals)<MAX_BUCKET_SIZE):
        all_candidates.update(tuple(sorted(c)) for c in combinations(vals, 2))

  print(f"Found {len(all_candidates)} candidate pairs")

  return all_candidates

def signature_sim(u1_idx, u2_idx):
  """
  Compute the similarity between the columns of signature matrix M
  """
  return np.mean(M[:,u1_idx]==M[:,u2_idx])

def jaccard_sim(u1_idx, u2_idx):
  """
  Compute Jaccard similarity between the users.
  Works on the transposed initial CSR matrix
  """
  row_u1 = csr.getrow(u1_idx).indices
  row_u2 = csr.getrow(u2_idx).indices

  inter = np.intersect1d(row_u1, row_u2, assume_unique=True)
  union = row_u1.size + row_u2.size - inter.size

  return inter.size / union if union > 0 else 0

def check_similarities(all_candidates):
  """
  Checks whether the found candidate pairs are actually similar.
  First, simple similarity is computed based on the signature matrix; if it exceeds the threshold of 0.5,
  the Jaccard similarity score is computed on the csr matrix. The pairs with the scores above 0.5 are written to the results file.
  """
  print(f"Checking signature and Jaccard similarity...")

  written_pairs=0 # To record the final number of written pairs
  passed_sig_sim=0 # To keep track of the pairs that are filtered out along the way
  with open('result.txt', 'w') as f:
    for pair in tqdm(all_candidates):
      # Get signature similarity
      sig_sim=signature_sim(pair[0], pair[1])
      # Only compute Jaccard similarity if the signature similarity is high enough
      if sig_sim>0.5:
        passed_sig_sim+=1
        # Compute Jaccard simlarity on the original data
        data_sim=jaccard_sim(pair[0], pair[1])
        if data_sim>0.5:
          f.write(str(pair[0]+1)+','+str(pair[1]+1) + '\n') # +1 to convert indexes to user ID's
          written_pairs+=1

  print('\n')
  # We get 150-200 pairs across different seeds
  print(f"Filtered out {len(all_candidates)-passed_sig_sim} pairs: signature similarity < 0.5")
  print(f"Filtered out {passed_sig_sim-written_pairs} pairs: Jaccard similarity < 0.5")
  print(f"Written {written_pairs} pairs to result.txt")

if __name__ == "__main__":

  # Prompt for the random seed, and set the seed variable
  SEED = int(input("Please enter the random seed"))
  print (f"Random seed is set to {SEED}")
  # Timing
  start=time.time()

  # Load the data
  if not os.path.exists(DATA_PATH): # Check path validity
      raise FileNotFoundError(f"No data file found at: {DATA_PATH}")
  print("Loading data...")
  data = np.load(DATA_PATH)
  print("Data is loaded")

  # Store the data in a sparse matrix
  csr=build_csr(data)
  # The original data can now be deleted to free up memory
  del data
  # Compute the signature matrix
  M=compute_signature_m(csr)
  # Find the best b and r values, given the chosen number of permutations
  opt_r, opt_b=find_b_r()
  # Perform LSH
  all_candidates=LSH(M, opt_r, opt_b)

  # Transpose the csr for efficiency and overwrite the original 
  csr = csr.transpose().tocsr()
  # Check the candidate pairs
  check_similarities(all_candidates)
  # Check the time
  end=time.time()
  elapsed=(end-start)/60
  print(f"Total execution time: {elapsed:.2f} minutes")