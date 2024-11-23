import argparse
import numpy as np
from immitation_learning.Utilities import *
from immitation_learning.Sequencing import *
from immitation_learning.Loader import *
from immitation_learning.train import SinglePolicy


def main(args):
    # Load dataset
    loader = DSSportsFormat(args.ds_path)
    ds = loader.load_data()
    all_off, all_def, all_ball, all_length = ds
    # The output 'ds' is a tuple:
    # ds[0]: all_off (List[np.ndarray]) - Group A data (e.g., offensive players).
    #        Shape for each array: (T, num_group_A_entities * features).
    # ds[1]: all_def (List[np.ndarray]) - Group B data (e.g., defensive players).
    #        Shape for each array: (T, num_group_B_entities * features).
    # ds[2]: all_ball (List[np.ndarray]) - Central entity data (e.g., the ball).
    #        Shape for each array: (T, features).
    # ds[3]: all_length (List[int]) - Number of timesteps (T) for each sample.
    
    # Load role mean data
    off_means = np.load(args.off_means_path)
    def_means = np.load(args.def_means_path)

    # Role assignment
    seq = RoleAssignment()
    _, def_seq = seq.assign_roles(all_def, def_means, all_length)
    _, off_seq = seq.assign_roles(all_off, off_means, all_length)

    # Combine sequences for single game
    single_game = [np.concatenate([def_seq[i], off_seq[i], all_ball[i]], axis=1) for i in range(len(def_seq))]
    train, target = get_sequences(single_game, window_size=50, step_size=15)

    # Define hyperparameters
    hyperparam = {
        'horizon': [10],
        'num_policies': 11,
        'batch_size': args.batch_size,
        'time_steps': 1,
        'dup_ft': 4,
        'learning_rate': args.learning_rate,
        'n_epoch': args.n_epoch,
        'total_timesteps': 49,
    }

    # Train single policy
    s_p = SinglePolicy(hyperparam)
    s_p.train_single_policy(args.role, train, target)


if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser(description="Train a single policy for football data.")
    parser.add_argument("--ds_path", type=str, required=True, help="Path to the dataset folder.")
    parser.add_argument("--off_means_path", type=str, required=True, help="Path to the offensive role means file.")
    parser.add_argument("--def_means_path", type=str, required=True, help="Path to the defensive role means file.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training (default: 32).")
    parser.add_argument("--learning_rate", type=float, default=0.0005, help="Learning rate for the model (default: 0.0005).")
    parser.add_argument("--n_epoch", type=int, default=5, help="Number of training epochs (default: 5).")
    parser.add_argument("--role", type=int, required=True, help="Role index to train the single policy.")
    args = parser.parse_args()

    main(args)
