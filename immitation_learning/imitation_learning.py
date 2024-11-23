import numpy as np, tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Activation, Dropout, TimeDistributed, BatchNormalization
from tensorflow.keras.layers import LSTM, GRU
from tensorflow.keras.optimizers import RMSprop, Adagrad, Adam, SGD
import os
from math import sqrt
from tensorflow.python.keras.engine import training

class PolicyTraining:

    def __init__(self, hyper_params):
        self.horizon = hyper_params["horizon"]
        self.num_policies = hyper_params["num_policies"]
        self.batch_size = hyper_params["batch_size"]
        self.time_steps = hyper_params["time_steps"]
        self.dup_ft = hyper_params["dup_ft"]
        self.num_player_ft = 13
        self.total_feature = self.num_player_ft * (22 + self.dup_ft * 2) + 4
        self.learning_rate = hyper_params["learning_rate"]
        self.n_epoch = hyper_params["n_epoch"]
        self.loss_fn = hyper_params["loss_fn"]
        self.optimizer = hyper_params["optimizer"]
        self.total_timesteps = hyper_params["total_timesteps"]
        self.hyperparam = hyper_params
        if hyper_params["agents_path"] is not None:
            agents_path = os.listdir(hyper_params["agents_path"])
            agents_path.sort()
            self.agents = [load_model(hyper_params["agents_path"] + p) for p in agents_path]
        else:
            self.agents = [self.create_agent() for _ in range(self.num_policies)]

    def create_agent(self):
        agent = Sequential()
        agent.add(LSTM(512, return_sequences=True, batch_input_shape=(self.batch_size, self.time_steps, self.total_feature), stateful=True))
        agent.add(LSTM(512, return_sequences=False, stateful=True))
        agent.add(Dense(2))
        agent.add(Activation("linear"))
        agent.reset_states()
        return agent

    def get_agents(self):
        return self.agents

    def get_param(self):
        return self.hyperparam

    def save_agents(self, list_idx=None):
        if list_idx is not None:
            for idx in list_idx:
                title = "assets/agents/role_" + str(idx)
                self.agents[idx].save(title)

        else:
            for role, agent in enumerate(self.agents):
                title = "assets/agents/role_" + str(role)
                agent.save(title)

    def feature_roll(self, role_idx, x_curr, x_prev=None):
        k = self.dup_ft
        total_dup = k * 2
        curr_player_ft = x_curr[:, :, :44]
        curr_ball_ft = x_curr[:, :, -2:]
        if x_prev is not None:
            prev_player_ft = x_prev[:, :, :44]
            prev_ball_ft = x_prev[:, :, -2:]
        shape = list(x_curr.shape)
        shape[2] = 286
        new_feature = np.zeros(shape)
        goal_pos = [1.0, 0]
        active_player = curr_player_ft[:, :, role_idx * 2:role_idx * 2 + 2]
        ball_vel = np.zeros(curr_ball_ft.shape)
        if x_prev is not None:
            ball_vel = curr_ball_ft - prev_ball_ft
        new_ball_ft = np.concatenate((curr_ball_ft, ball_vel), axis=2)
        dist_def = np.zeros((shape[0], shape[1], 11))
        dist_off = np.zeros((shape[0], shape[1], 11))
        for i in range(22):
            new_feature[:, :, i * 13:i * 13 + 2] = curr_player_ft[:, :, i * 2:i * 2 + 2]
            goal_pos = [1.0, 0]
            if x_prev is None:
                new_feature[:, :, i * 13 + 2:i * 13 + 4] = 0
            else:
                new_feature[:, :, i * 13 + 2:i * 13 + 4] = curr_player_ft[:, :, i * 2:i * 2 + 2] - prev_player_ft[:, :, i * 2:i * 2 + 2]
            pos = new_feature[:, :, i * 13:i * 13 + 2]
            new_feature[:, :, i * 13 + 4] = ((pos[:, :, 0] - curr_ball_ft[:, :, 0]) ** 2 + (pos[:, :, 1] - curr_ball_ft[:, :, 1]) ** 2) ** 0.5
            a = pos[:, :, 0] - curr_ball_ft[:, :, 0]
            b = pos[:, :, 1] - curr_ball_ft[:, :, 1]
            c = new_feature[:, :, i * 13 + 4]
            new_feature[:, :, i * 13 + 5] = np.divide(a, c, out=(np.zeros_like(a)), where=(c != 0))
            new_feature[:, :, i * 13 + 6] = np.divide(b, c, out=(np.zeros_like(b)), where=(c != 0))
            new_feature[:, :, i * 13 + 7] = ((pos[:, :, 0] - goal_pos[0]) ** 2 + (pos[:, :, 1] - goal_pos[1]) ** 2) ** 0.5
            a = pos[:, :, 0] - goal_pos[0]
            b = pos[:, :, 1] - goal_pos[1]
            c = new_feature[:, :, i * 13 + 7]
            new_feature[:, :, i * 13 + 8] = np.divide(a, c, out=(np.zeros_like(a)), where=(c != 0))
            new_feature[:, :, i * 13 + 9] = np.divide(b, c, out=(np.zeros_like(b)), where=(c != 0))
            new_feature[:, :, i * 13 + 10] = ((pos[:, :, 0] - goal_pos[0]) ** 2 + (pos[:, :, 1] - goal_pos[1]) ** 2) ** 0.5
            a = pos[:, :, 0] - active_player[:, :, 0]
            b = pos[:, :, 0] - active_player[:, :, 1]
            c = new_feature[:, :, i * 13 + 10]
            new_feature[:, :, i * 13 + 11] = np.divide(a, c, out=(np.zeros_like(a)), where=(c != 0))
            new_feature[:, :, i * 13 + 12] = np.divide(b, c, out=(np.zeros_like(b)), where=(c != 0))
            if i < 11:
                dist_def[:, :, i] = new_feature[:, :, i * 13 + 10]
            else:
                dist_off[:, :, i - 11] = new_feature[:, :, i * 13 + 10]

        k_nearest_teammate = dist_def.argsort()[:, :, 1:k + 1]
        k_nearest_opponent = 11 + dist_off.argsort()[:, :, :k]
        k_combine = np.concatenate((k_nearest_teammate, k_nearest_opponent), axis=2)
        nearest_player_ft = np.zeros((shape[0], shape[1], k * 2 * 13))
        nearest_player_ft[:, :, :total_dup] = k_combine[:, :, :] * 13
        for z in range(1, 13):
            nearest_player_ft[:, :, z * total_dup:z * total_dup + total_dup] = nearest_player_ft[:, :, (z - 1) * total_dup:(z - 1) * total_dup + total_dup] + 1

        nearest_player_ft.sort()
        nearest_player_ft = nearest_player_ft.astype(int)
        final_feature = np.concatenate((new_feature, np.take(new_feature, nearest_player_ft), new_ball_ft), axis=2)
        return final_feature


class SinglePolicy(PolicyTraining):

    def __init__(self, hyper_params):
        super().__init__(hyper_params)

    def train_single_policy(self, role_idx, train, target, seq):
        title = "Epoch {} - Loss : {}"
        for k in self.horizon:
            print("Horizon: ", k)
            for epoch in range(self.n_epoch):
                total_loss_perbatch = []
                for batch in seq.iterate_minibatches(train, target, (self.batch_size), shuffle=False):
                    x_batch, y_batch = batch
                    for t in range(0, self.total_timesteps + 1 - k, k):
                        with tf.GradientTape() as tape:
                            all_loss = []
                            x_curr = x_batch[:, t:t + 1, :]
                            x_up = self.feature_roll(role_idx, x_curr)
                            y_curr = y_batch[:, t, role_idx * 2:role_idx * 2 + 2]
                            for i in range(1, k):
                                pred = self.agents[role_idx](x_up, training=True)
                                loss = self.loss_fn(y_curr, pred)
                                all_loss.append(loss)
                                y_curr = y_batch[:, t + i, role_idx * 2:role_idx * 2 + 2]
                                x_prev = x_curr
                                x_curr = x_batch[:, t + i:t + i + 1, :]
                                x_curr[:, :, role_idx * 2:role_idx * 2 + 2] = np.expand_dims(pred, 1)
                                x_up = self.feature_roll(role_idx, x_curr, x_prev=x_prev)

                            grads = tape.gradient(all_loss, self.agents[role_idx].trainable_variables)
                            self.optimizer.apply_gradients(zip(grads, self.agents[role_idx].trainable_variables))

                    total_loss_perbatch.append(tf.reduce_mean(all_loss))
                    self.agents[role_idx].reset_states()

                print(title.format(epoch, tf.reduce_mean(total_loss_perbatch)))


class JointPolicy(PolicyTraining):

    def __init__(self, hyper_params):
        super().__init__(hyper_params)

    @tf.function(experimental_relax_shapes=True)
    def grad_update(self, role_idx, x, y):
        pred = self.agents[role_idx](x, training=True)
        loss = self.loss_fn(y, pred)
        grads = tf.gradients(loss, self.agents[role_idx].trainable_variables)
        return (loss, grads)

    def train_joint_policy(self, train, target, seq):
        title = "Role {} - Loss : {}"
        for k in self.horizon:
            print("Horizon: ", k)
            for epoch in range(self.n_epoch):
                total_loss_perbatch = [[] for _ in range(len(self.agents))]
                for batch in seq.iterate_minibatches(train, target, (self.batch_size), shuffle=False):
                    x_batch, y_batch = batch
                    for t in range(0, self.total_timesteps + 1 - k, k):
                        x_updated = np.copy(x_batch)
                        x_prev = None
                        for i in range(k):
                            x_curr = x_updated[:, t + i:t + i + 1, :]
                            for role_idx, agent in enumerate(self.agents):
                                x_up = self.feature_roll(role_idx, x_curr, x_prev=x_prev)
                                pred = agent(x_up, training=True)
                                x_updated[:, t + i + 1, role_idx * 2:role_idx * 2 + 2] = pred

                            x_prev = x_curr

                        for role_idx, agent in enumerate(self.agents):
                            total_loss = []
                            x_prev = None
                            for i in range(k):
                                x_curr = x_updated[:, t + i:t + i + 1, :]
                                x = self.feature_roll(role_idx, x_curr, x_prev=x_prev)
                                y = y_batch[:, t + i, role_idx * 2:role_idx * 2 + 2]
                                loss, grads = self.grad_update(role_idx, x, y)
                                self.optimizer.apply_gradients(zip(grads, self.agents[role_idx].trainable_variables))
                                x_prev = x_curr
                                total_loss.append(loss)

                            total_loss_perbatch[role_idx].append(tf.reduce_mean(total_loss))

                    for role_idx, agent in enumerate(self.agents):
                        agent.reset_states()

                print("Epoch ", epoch)
                for idx in range(len(self.agents)):
                    print(title.format(idx, tf.reduce_mean(total_loss_perbatch[idx])))
