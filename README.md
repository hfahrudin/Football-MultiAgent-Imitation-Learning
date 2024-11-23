
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

- **SinglePolicy**: Train a policy for each agent, independently.
- **JointPolicy**: Train policies for all agents simultanously with shared environment.

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

## How It Works

### Imitation Learning

Imitation Learning (IL) is a method where an AI model learns to perform tasks by mimicking expert demonstrations rather than being explicitly programmed or optimized via trial-and-error (e.g., reinforcement learning)[4]. The process involves:

1. **Expert Demonstration**: Collecting data from skilled individuals or pre-recorded activities. In this case, football players’ movements and strategies are captured.
2. **Learning Policy**: The model is trained to map observations (e.g., player positions) to actions (e.g., movement, passing) by minimizing the difference between its predictions and the expert’s actions.
3. **Goal**: To replicate real-world tactical decisions and strategies, allowing the agents to behave like expert players.

Imitation Learning is effective when sufficient high-quality data is available and when the goal is to replicate human-like behaviors rather than optimize for predefined rewards.

### HMM in Finding Roles

The Hidden Markov Model (HMM) is a probabilistic model used to infer hidden states based on observed data. In the context of role assignment for football players:

1. **Input**:
   - Positional data for players (e.g., offensive and defensive positions).
   - Observations for each timestep (e.g., player coordinates, velocities).

2. **HMM Training**:
   - The HMM learns statistical patterns from the dataset, treating observed player movements as evidence for underlying "hidden" roles (e.g., striker, defender, midfielder).
   - It models transitions between roles over time, capturing the fluid nature of football strategies.

3. **Role Assignment**:
   - Once trained, the HMM assigns players to specific roles at each timestep by identifying the most probable sequence of states (roles) given the observed data.
   - The output includes role sequences, role-specific mean positions, and variances, which define the spatial and tactical behavior of each role.

Definition of each agent role as a training features could improve imitation loss substantially[1]

### Input Feature Overview

1. **Temporal Context**: The input features are organized in a time series, where each moment represents the state of the game for all players.

2. **Role Assignment**: Player roles (e.g., forward, midfielder) are derived from their spatial behavior using Euclidean distance to pre-defined cluster centers.

3. **Player Features**: For each player, at each timestamp, the following features are calculated:
   - **Distance to Goal**
   - **Distance to Ball**
   - **Distance to Teammates/Opponents**
   - **Velocity**
   - **Relative Movement** (change in position relative to others)

4. **Input Shape**: The input is a 3D array with shape `(num_moments, num_players, 13)`:
   - `num_moments`: Temporal dimension (time steps)
   - `num_players`: Number of players
   - `13`: Features per player at each time step


## Example Results

**Dataset:** I used this [dataset](https://arxiv.org/abs/1703.03121) combined with my private dataset around comparable size.

<div align="center">
  <img src="https://github.com/user-attachments/assets/c4899c0e-6431-48e3-9326-26ffe0df1db4" alt="Demo GIF" width="500">
    <p><b>Joint Policy Agent Inference Demo</b></p>
</div>

<div align="center">
  <img src="https://github.com/user-attachments/assets/6e4615d2-0408-4c6f-8400-56d52be05f12" alt="Demo GIF" width="500">
       <p><b>Single Policy Agent Inference Demo</b></p> 
</div>

Both policies achieve reasonable accuracy given the limited dataset. However, the **Joint Policy** shows smoother movements, fewer abrupt role changes, and less frequent instances of agents unexpectedly swapping positions compared to the **Single Policy**.

## References

1. **Coordinated Multi-Agent Imitation Learning**  
   Hoang M. Le, Yisong Yue, Peter Carr, Patrick Lucey (2017). *Coordinated Multi-Agent Imitation Learning*. Retrieved from [link](https://arxiv.org/abs/1703.03121)

2. **Data-Driven Ghosting using Deep Imitation Learning**  
   Hoang M. Le, Peter Carr, Yisong Yue, and Patrick Lucey (2017). *Data-Driven Ghosting using Deep Imitation Learning*. Retrieved from [link](https://la.disneyresearch.com/wp-content/uploads/Data-Driven-Ghosting-using-Deep-Imitation-Learning-Paper1.pdf)

3. **Coordinated-Multi-Agent-Imitation-Learning Implementation (Basketball)**  
   samshipengs. Retrieved from [Github Repo](https://github.com/samshipengs/Coordinated-Multi-Agent-Imitation-Learning)

4. **A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning**  
   Stephane Ross, Geoffrey J. Gordon, J. Andrew Bagnell. Retrieved from [link](https://arxiv.org/abs/1011.0686)
