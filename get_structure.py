
import numpy as np
import immitation_learning.Utilities as util
import pandas as pd
import os
import immitation_learning.StructureLearning as RA


ds_path = 'your path'
all_ds_path =  os.listdir(ds_path)
player_num = 11
n_defend = 11
n_offend = 11
n_ind = 4
n_comp = 11

#upload ds
all_def = []
all_off = []
first = True
all_ball = []
all_length = []
for ds in all_ds_path:
    join_path = ds_path+ds
    df = pd.read_csv(join_path)
    df = util.get_velocities(df)
    
    ball_df = df[["ball_x", "ball_y"]]
    off_ids = np.unique( [ c for c in df.columns if c[:3] in ['off'] ])
    deff_ids = np.unique( [ c for c in df.columns if c[:3] in ['def'] ])

    off_df = df[off_ids]
    deff_df = df[deff_ids]

    off = off_df.to_numpy()
    deff = deff_df.to_numpy()
    ball = ball_df.to_numpy()

    all_def.append(deff)
    all_off.append(off)
    all_ball.append(ball)

    # all_defend_moments_ = np.concatenate([deff_flatten[:, i:i+n_ind] for i in range(0, n_ind*n_defend, n_ind)], axis=0)
    # all_offend_moments_ = np.concatenate([off_flatten[:, i:i+n_ind] for i in range(0, n_ind*n_offend, n_ind)], axis=0)
    
    lgth = deff_df.shape[0]

    # if first:
    #     first = False
    #     all_def = all_defend_moments_
    #     all_off = all_offend_moments_
    # else:
    #     all_def = np.vstack((all_def, all_defend_moments_))
    #     all_off = np.vstack((all_off, all_offend_moments_))

    

    all_length.append(lgth)

#Please change the data processing according to use case


    #preproc for hmm
all_moments_def = np.concatenate(all_def, axis=0)
all_moments_off = np.concatenate(all_off, axis=0)
all_moments_def_ = np.concatenate([all_moments_def[:, i:i+n_ind] for i in range(0, n_ind*n_defend, n_ind)], axis=0)
all_moments_off_ = np.concatenate([all_moments_off[:, i:i+n_ind] for i in range(0, n_ind*n_offend, n_ind)], axis=0)

lengths_repeat = np.concatenate([all_length for _ in range(11)], axis=0)
    
    
hmm_model  =  RA.RoleAssignment(500, True)

defend_state_sequence_, defend_means, defend_covs, _ = hmm_model.train_hmm(all_moments_def_, lengths_repeat, n_comp)

offend_state_sequence_, offend_means, offend_covs, _ = hmm_model.train_hmm(all_moments_def_, lengths_repeat, n_comp)


# get role orders
_, defend_roles = hmm_model.assign_roles(all_moments_def_, all_moments_def, defend_means, all_length)
_, offend_roles = hmm_model.assign_roles(all_moments_off_, all_moments_off, offend_means, all_length)
