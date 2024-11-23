
import numpy as np, tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.layers import LSTM
from Sequencing import *
import os
from math import sqrt
from datetime import datetime

class PolicyTraining:

    def __init__(self, hyper_params):
        time_stamp = datetime.now().strftime("%Y%m%d_%H%M")
        self.horizon = hyper_params["horizon"]
        self.num_policies = hyper_params["num_policies"]
        self.batch_size = hyper_params["batch_size"]
        self.time_steps = hyper_params["time_steps"] if "time_steps" in hyper_params else 1
        self.dup_ft = hyper_params["dup_ft"] if "dup_ft" in hyper_params else 4
        self.num_player_ft = 13
        self.total_feature = self.num_player_ft * (22 + self.dup_ft * 2) + 4
        self.learning_rate = hyper_params["learning_rate"] if "learning_rate" in hyper_params else 0.001
        self.n_epoch = hyper_params["n_epoch"]
        self.loss_fn = hyper_params["loss_fn"] if "loss_fn" in hyper_params else tf.keras.losses.MeanSquaredError()
        self.total_timesteps = hyper_params["total_timesteps"]
        self.hyperparam = hyper_params
        self.upload_path = hyper_params["upload_path"] if "upload_path" in hyper_params else "assets/agents/" + time_stamp + "/"
        self.opt = hyper_params["opt"] if "opt" in hyper_params else tf.keras.optimizers.Adam
        if "agents_path" in hyper_params:
            agents_path = os.listdir(hyper_params["agents_path"])
            agents_path.sort(key=float)
            self.agents = [load_model(hyper_params["agents_path"] + p) for p in agents_path]
            for agent in self.agents:
                agent.compile(loss="mse", optimizer=self.opt(learning_rate=(self.learning_rate)))

        else:
            self.agents = [self.create_agent() for _ in range(self.num_policies)]

    def create_agent(self):
        agent = Sequential()
        agent.add(LSTM(512, return_sequences=True, batch_input_shape=(self.batch_size, self.time_steps, self.total_feature), stateful=True))
        agent.add(LSTM(512, return_sequences=False, stateful=True))
        agent.add(Dense(2))
        agent.add(Activation("linear"))
        agent.compile(loss="mse", optimizer=self.opt(learning_rate=(self.learning_rate)))
        agent.reset_states()
        return agent

    @tf.function(experimental_relax_shapes=True)
    def grad_update(self, role_idx, x, y):
        pred = self.agents[role_idx](x, training=True)
        loss = self.loss_fn(y, pred)
        grads = tf.gradients(loss, self.agents[role_idx].trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.agents[role_idx].trainable_variables))
        return loss

    def save_agents(self, list_idx=None):
        if list_idx is not None:
            for idx in list_idx:
                title = self.upload_path + str(idx)
                self.agents[idx].save(title)

        else:
            for role, agent in enumerate(self.agents):
                title = self.upload_path + str(role)
                agent.save(title)


class SoccerAgent(PolicyTraining):

    def __init__(self, model_path):
        self.model_path = model_path
        self.num_player_ft = 13
        self.dup_ft = 4
        self.total_feature = self.num_player_ft * (22 + self.dup_ft * 2) + 4
        self.batch_size = 1
        self.learning_rate = 0.005
        self.time_steps = None
        agents_path = os.listdir(self.model_path)
        agents_path.sort(key=float)
        self.old_agents = [load_model(self.model_path + p) for p in agents_path]
        self.opt = tf.keras.optimizers.Adam
        self.agents = [self.create_agent() for _ in range(len(self.old_agents))]
        for _, (old_agent, agent) in enumerate(zip(self.old_agents, self.agents)):
            agent.set_weights(old_agent.get_weights())


class SinglePolicy(PolicyTraining):

    def __init__(self, hyper_params):
        super().__init__(hyper_params)

    def train_single_policy(self, role_idx, train, target):
        title = "Epoch {} - Loss : {}"
        agent = self.agents[role_idx]
        header = "==================Agent " + str(role_idx) + " ================="
        print(header)
        for k in self.horizon:
            print("Horizon: ", k)
            for epoch in range(self.n_epoch):
                total_loss_perbatch = []
                for batch in iterate_minibatches(train, target, (self.batch_size), shuffle=False):
                    x_batch, y_batch = batch
                    for t in range(0, self.total_timesteps - k, k):
                        all_loss = []
                        x_updated = np.copy(x_batch)
                        x_prev = None
                        for i in range(k):
                            x_curr = x_updated[:, t + i:t + i + 1, :]
                            x_up = feature_roll(role_idx, x_curr, x_prev=x_prev)
                            pred = agent(x_up, training=True)
                            x_updated[:, t + i + 1, role_idx * 2:role_idx * 2 + 2] = pred
                            x_prev = x_curr

                        x_prev = None
                        for i in range(k):
                            x_curr = x_updated[:, t + i:t + i + 1, :]
                            x = feature_roll(role_idx, x_curr, x_prev=x_prev)
                            y = y_batch[:, t + i, role_idx * 2:role_idx * 2 + 2]
                            loss = agent.train_on_batch(x, y)
                            all_loss.append(loss)

                    total_loss_perbatch.append(tf.reduce_mean(all_loss))
                    agent.reset_states()

                print(title.format(epoch, tf.reduce_mean(total_loss_perbatch)))


class JointPolicy(PolicyTraining):

    def __init__(self, hyper_params):
        super().__init__(hyper_params)

    def train_joint_policy(self, train, target):
        title = "Role {} - Loss : {}"
        for k in self.horizon:
            print("Horizon: ", k)
            for epoch in range(self.n_epoch):
                total_loss_perbatch = [[] for _ in range(len(self.agents))]
                for batch in iterate_minibatches(train, target, (self.batch_size), shuffle=False):
                    x_batch, y_batch = batch
                    for t in range(0, self.total_timesteps - k, k):
                        x_updated = np.copy(x_batch)
                        x_prev = None
                        for i in range(k):
                            x_curr = x_updated[:, t + i:t + i + 1, :]
                            for role_idx, agent in enumerate(self.agents):
                                x_up = feature_roll(role_idx, x_curr, x_prev=x_prev)
                                pred = agent(x_up, training=True)
                                x_updated[:, t + i + 1, role_idx * 2:role_idx * 2 + 2] = pred

                            x_prev = x_curr

                        for role_idx, agent in enumerate(self.agents):
                            total_loss = []
                            x_prev = None
                            for i in range(k):
                                x_curr = x_updated[:, t + i:t + i + 1, :]
                                x = feature_roll(role_idx, x_curr, x_prev=x_prev)
                                y = y_batch[:, t + i, role_idx * 2:role_idx * 2 + 2]
                                loss = agent.train_on_batch(x, y)
                                x_prev = x_curr
                                total_loss.append(loss)

                            total_loss_perbatch[role_idx].append(tf.reduce_mean(total_loss))

                    for role_idx, agent in enumerate(self.agents):
                        agent.reset_states()

                print("Epoch ", epoch)
                for idx in range(len(self.agents)):
                    print(title.format(idx, tf.reduce_mean(total_loss_perbatch[idx])))
