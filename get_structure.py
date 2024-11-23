import argparse
import numpy as np
import immitation_learning.Utilities as util
import pandas as pd
import os
import immitation_learning.StructureLearning as RA


def main(args):
    ds_path = args.ds_path
    all_ds_path = os.listdir(ds_path)
    player_num = args.player_num
    n_defend = args.n_defend
    n_offend = args.n_offend
    n_ind = args.n_ind
    n_comp = args.n_comp
    n_epoch = args.n_epoch

    # Initialize lists for data storage
    all_def = []
    all_off = []
    all_ball = []
    all_length = []

    # Process dataset
    for ds in all_ds_path:
        join_path = os.path.join(ds_path, ds)
        df = pd.read_csv(join_path)
        df = util.get_velocities(df)
        
        # Extract ball and player data
        ball_df = df[["ball_x", "ball_y"]]
        off_ids = np.unique([c for c in df.columns if c[:3] in ['off']])
        deff_ids = np.unique([c for c in df.columns if c[:3] in ['def']])

        off_df = df[off_ids]
        deff_df = df[deff_ids]

        off = off_df.to_numpy()
        deff = deff_df.to_numpy()
        ball = ball_df.to_numpy()

        all_def.append(deff)
        all_off.append(off)
        all_ball.append(ball)

        lgth = deff_df.shape[0]
        all_length.append(lgth)

    # Data processing for HMM
    all_moments_def = np.concatenate(all_def, axis=0)
    all_moments_off = np.concatenate(all_off, axis=0)
    all_moments_def_ = np.concatenate([all_moments_def[:, i:i + n_ind] for i in range(0, n_ind * n_defend, n_ind)], axis=0)
    all_moments_off_ = np.concatenate([all_moments_off[:, i:i + n_ind] for i in range(0, n_ind * n_offend, n_ind)], axis=0)

    lengths_repeat = np.concatenate([all_length for _ in range(player_num)], axis=0)

    # Train HMM model for role assignment
    hmm_model = RA.RoleAssignment(n_epoch, True)

    defend_state_sequence_, defend_means, defend_covs, _ = hmm_model.train_hmm(all_moments_def_, lengths_repeat, n_comp)
    offend_state_sequence_, offend_means, offend_covs, _ = hmm_model.train_hmm(all_moments_def_, lengths_repeat, n_comp)

    # Get role assignments
    _, defend_roles = hmm_model.assign_roles(all_moments_def_, all_moments_def, defend_means, all_length)
    _, offend_roles = hmm_model.assign_roles(all_moments_off_, all_moments_off, offend_means, all_length)

    #implement save npy


if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser(description="Process football dataset for role assignment using HMM.")
    parser.add_argument("--ds_path", type=str, required=True, help="Path to the dataset folder.")
    parser.add_argument("--player_num", type=int, default=11, help="Number of players (default: 11).")
    parser.add_argument("--n_defend", type=int, default=11, help="Number of defensive players (default: 11).")
    parser.add_argument("--n_offend", type=int, default=11, help="Number of offensive players (default: 11).")
    parser.add_argument("--n_ind", type=int, default=4, help="Number of individual features per player (default: 4).")
    parser.add_argument("--n_comp", type=int, default=11, help="Number of components for HMM (default: 11).")
    parser.add_argument("--n_epoch", type=int, default=500, help="")
    args = parser.parse_args()

    main(args)
