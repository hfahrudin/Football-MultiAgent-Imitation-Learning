
# Football Defensive Ghosting Agent using Imitation Learning

This project applies the principles from the paper *[Coordinated Multi-Agent Imitation Learning](https://arxiv.org/pdf/1703.03121)* and adapts them for creating defensive player agents to imitate real-world tactics in football. It focuses on training football agents using both single and joint policy frameworks, leveraging imitation learning to produce intelligent agents that can perform tactical movements similar to human players.

## Key Features

- **Imitation Learning**: Trains agents to imitate actions from expert demonstrations rather than through traditional reinforcement learning.
- **Single Policy Training**: Uses a single policy model to train individual players in specific roles (e.g., defensive players).
- **Joint Policy**: Trains multiple agents simultaneously to coordinate their actions as a team.
- **Structured Training**: Structure learning is used to find the mean positions of each role (offensive and defensive).

**WARNING:** There are a lot of parameters that you could changes and tune. Although, you need to modify the script a little bit.

## Installation

To get started, you need to install the required dependencies. This can be done via `pip`:

```bash
pip install -r requirements.txt
```

Ensure you have the necessary environment and libraries for running the imitation learning algorithms.

## Usage

### 1. Prepare the dataset

**Notes:** You need to edit the data loader function according to your specific needs.  

```bash
    # The output 'ds' is a tuple:
    # ds[0]: all_off (List[np.ndarray]) - Group A data (e.g., offensive players).
    #        Shape for each array: (T, num_group_A_entities * features).
    # ds[1]: all_def (List[np.ndarray]) - Group B data (e.g., defensive players).
    #        Shape for each array: (T, num_group_B_entities * features).
    # ds[2]: all_ball (List[np.ndarray]) - Central entity data (e.g., the ball).
    #        Shape for each array: (T, features).
    # ds[3]: all_length (List[int]) - Number of timesteps (T) for each sample.

```
### 2. Get Role Position Means

Processes football datasets and applies Hidden Markov Models (HMM) for role assignment of players (offensive and defensive roles) .
 
```bash
python get_structure.py --ds_path /path/to/your/dataset --player_num 11 --n_defend 11 --n_offend 11 --n_ind 4 --n_comp 11 --n_epoch 500
```

  
### 3. Training the Model

There are two types of models you can train: **SinglePolicy** and **JointPolicy**.

- **SinglePolicy**: Train a policy for one player (e.g., defensive role).
- **JointPolicy**: Train policies for multiple players (e.g., both offensive and defensive roles).

For **SinglePolicy**:

```bash
python single_policy_train.py --ds_path /path/to/dataset --off_means_path /path/to/off_means.npy --def_means_path /path/to/def_means.npy
```

For **JointPolicy**:

```bash
python joint_policy_train.py --ds_path /path/to/dataset --off_means_path /path/to/off_means.npy --def_means_path /path/to/def_means.npy
```

### 4. Hyperparameters

The following hyperparameters are used in both the SinglePolicy and JointPolicy training scripts:

- `horizon`: Number of timesteps the agent should consider for decision-making.
- `num_policies`: The number of policies (agents) to be trained.
- `batch_size`: The batch size for training.
- `learning_rate`: Learning rate for optimization.
- `n_epoch`: Number of epochs for training.
- `total_timesteps`: Total number of timesteps for training.

Modify these parameters in the script or pass them via the command line.

## Example Results

## How It Works


## References

1. **Coordinated Multi-Agent Imitation Learning**  
   Hoang M. Le, Yisong Yue, Peter Carr, Patrick Lucey (2017). *Coordinated Multi-Agent Imitation Learning*. Retrieved from [link](https://arxiv.org/abs/1703.03121)

2. **Data-Driven Ghosting using Deep Imitation Learning**  
   Hoang M. Le, Peter Carr, Yisong Yue, and Patrick Lucey (2017). *Data-Driven Ghosting using Deep Imitation Learning*. Retrieved from [link](https://la.disneyresearch.com/wp-content/uploads/Data-Driven-Ghosting-using-Deep-Imitation-Learning-Paper1.pdf)
