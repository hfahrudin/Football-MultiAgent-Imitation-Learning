
import numpy as np
from immitation_learning.Utilities import *
from immitation_learning.Sequencing import *
from immitation_learning.Loader import *
from immitation_learning.train import SinglePolicy


ds_path = 'Your path'
off_means_path  = "assets/Role_means/off_means.npy"
def_means_path  = "assets/Role_means/def_means.npy"
Loader = DSSportsFormat(ds_path)

ds = Loader.load_data()

# please change the loader function since it was for my specific case
# The output 'ds' is a tuple:
# ds[0]: all_off (List[np.ndarray]) - Group A data (e.g., offensive players).
#        Shape for each array: (T, num_group_A_entities * features).
# ds[1]: all_def (List[np.ndarray]) - Group B data (e.g., defensive players).
#        Shape for each array: (T, num_group_B_entities * features).
# ds[2]: all_ball (List[np.ndarray]) - Central entity data (e.g., the ball).
#        Shape for each array: (T, features).
# ds[3]: all_length (List[int]) - Number of timesteps (T) for each sample.

all_off, all_def, all_ball, all_length = ds
off_means = np.load(off_means_path)
def_means  = np.load(def_means_path)

seq = RoleAssignment()
_, def_seq = seq.assign_roles(all_def, def_means, all_length)
_, off_seq = seq.assign_roles(all_off, off_means, all_length)

single_game = [np.concatenate([def_seq[i], off_seq[i], all_ball[i]], axis=1) for i in range(len(def_seq))]
train, target = get_sequences(single_game, 50, 15)

hyperparam = {
    'horizon' : [10],
    'num_policies' : 11,
    'batch_size' : 32,
    'time_steps' : 1,
    'dup_ft' : 4,
    'learning_rate' : 0.0005,
    'n_epoch' : 5,
    'total_timesteps': 49
}

s_p = SinglePolicy(hyperparam)


role = 9
s_p.train_single_policy(role , train, target)