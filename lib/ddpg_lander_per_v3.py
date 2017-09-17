import tensorflow as tf
import numpy as np
import gym
from gym import wrappers
import tflearn

from sum_tree import SumTree

import math
import random
from onstein_uhlenbeck_process import OrnsteinUhlenbeckProcess as OUP

###
# ==================================
# Lessons learned
# ==================================
# TWO SIDEd MEMORY.  INITIALLY TRY WITH lots of noise (say 1).
# TWO DIFFERENT SUMTREES AND PULL 1/2 from each or a Ratio from each.  see if moving ration to positive bettern than 50%
# Learning too abrupt/too divergent if 50% positive
# Learning too divergent with almost all successes
# Learning not too good with high error of 1.  try higher of 100
# Tried 100 and learned better but still very varied


# Improvement Ideas
# collect rewards and successes all at once and put into a random batch
# update network once
# switch back to 50%
# change weighting to be all same and remove updating and see if better
# in future but not for this problem, need to switch to rank based prioritization
# then consider if two rank based algorithms needed


# ==========================
#   Training Parameters
# ==========================
# Max training steps
MAX_EPISODES = 50000
# Max episode length
MAX_EP_STEPS = 1000
# Base learning rate for the Actor network
ACTOR_LEARNING_RATE = 0.0001
# Base learning rate for the Critic Network
CRITIC_LEARNING_RATE = 0.001
# Discount factor
GAMMA = 0.99
# Soft target update param
TAU = 0.001
#TAU = 0.01
# Exploration parameters
THETA = 2
SIGMA = .9
MEAN = 0
EXPLORATION_INITIAL_WEIGHT = 1
EXPLORATION_DECREMENT_RATE = .04
#  Prioritized Experience Replay
E = 100 # forces adding some error.  not really needed since put a np.maximun(error, 1) to ensure at least 1 appears
# Number of neuron on each hidden layer
H1 = 400
H2 = 300
# which episode to start training on data
START_LEARNING = 3


# ===========================
#   Utility Parameters
# ===========================
# Render gym env during training
RENDER_ENV = True
RENDER_FREQUENCY = 10

# Use Gym Monitor
GYM_MONITOR_EN = False
# Gym environment
# ENV_NAME = 'MountainCarContinuous-v0'
ENV_NAME = 'LunarLanderContinuous-v2'
# Directory for storing gym results
MONITOR_DIR = './results/gym_ddpg'
# Directory for storing tensorboard summary results
# SUMMARY_DIR = './results/tf_ddpg'
RANDOM_SEED = 1234
# Size of replay buffer
BUFFER_SIZE = 20000
SUCCESS_RATIO = .5 # percentage from batch to pull from positive rewards
MINIBATCH_SIZE = 64



# ===========================
#   Prioritized Experience Replay
# ===========================

class Memory:   # stored in SumTree

    def __init__(self, capacity):
        self.tree = SumTree(capacity)
        self.e = E

    def _getPriority(self, error):
        return error + self.e

    def add(self, error, sample):
        p = self._getPriority(error)
        self.tree.add(p, sample)

    def sample(self, n):
        batch = []
        segment = self.tree.total() / n

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            batch.append( (idx, data) )

        return batch

    def update(self, idx, error):
        p = self._getPriority(error)
        self.tree.update(idx, p)


# ===========================
#   Actor and Critic DNNs
# ===========================




class ActorNetwork(object):
    """
    Input to the network is the state, output is the action
    under a deterministic policy.
    The output layer activation is a tanh to keep the action
    between -2 and 2
    """

    def __init__(self, sess, state_dim, action_dim, action_bound, learning_rate, tau):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.action_bound = action_bound
        self.learning_rate = learning_rate
        self.tau = tau

        # Actor Network
        self.inputsmod, self.out, self.scaled_out = self.create_actor_network()

        self.network_params = tf.trainable_variables()

        # Target Network
        self.target_inputs, self.target_out, self.target_scaled_out = self.create_actor_network()

        self.target_network_params = tf.trainable_variables()[
                                     len(self.network_params):]

        # Op for periodically updating target network with online network
        # weights
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) +
                                                  tf.multiply(self.target_network_params[i], 1. - self.tau))
             for i in range(len(self.target_network_params))]

        # This gradient will be provided by the critic network
        self.action_gradient = tf.placeholder(tf.float32, [None, self.a_dim])

        # Combine the gradients here
        self.actor_gradients = tf.gradients(
            self.scaled_out, self.network_params, -self.action_gradient)

        # Optimization Op
        self.optimize = tf.train.AdamOptimizer(self.learning_rate). \
            apply_gradients(zip(self.actor_gradients, self.network_params))

        self.num_trainable_vars = len(
            self.network_params) + len(self.target_network_params)

    def create_actor_network(self):
        inputs = tflearn.input_data(shape=[None, self.s_dim])
        net = tflearn.fully_connected(inputs, H1, activation='relu')
        net = tflearn.fully_connected(net, H2, activation='relu')
        # Final layer weights are init to Uniform[-3e-3, 3e-3]
        w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        out = tflearn.fully_connected(
            net, self.a_dim, activation='tanh', weights_init=w_init)
        # Scale output to -action_bound to action_bound
        scaled_out = tf.multiply(out, self.action_bound)
        return inputs, out, scaled_out

    def train(self, inputs, a_gradient):
        self.sess.run(self.optimize, feed_dict={
            self.inputsmod: inputs,
            self.action_gradient: a_gradient
        })

    def predict(self, inputs):
        return self.sess.run(self.scaled_out, feed_dict={
            self.inputsmod: inputs
        })

    def predict_target(self, inputs):
        return self.sess.run(self.target_scaled_out, feed_dict={
            self.target_inputs: inputs
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

    def get_num_trainable_vars(self):
        return self.num_trainable_vars


class CriticNetwork(object):
    """
    Input to the network is the state and action, output is Q(s,a).
    The action must be obtained from the output of the Actor network.
    """

    def __init__(self, sess, state_dim, action_dim, learning_rate, tau, num_actor_vars):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.learning_rate = learning_rate
        self.tau = tau

        # Create the critic network
        self.inputs, self.action, self.out = self.create_critic_network()

        self.network_params = tf.trainable_variables()[num_actor_vars:]

        # Target Network
        self.target_inputs, self.target_action, self.target_out = self.create_critic_network()

        self.target_network_params = tf.trainable_variables()[(len(self.network_params) + num_actor_vars):]

        # Op for periodically updating target network with online network
        # weights with regularization
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) + tf.multiply(self.target_network_params[i], 1. - self.tau))
             for i in range(len(self.target_network_params))]

        # Network target (y_i)
        self.predicted_q_value = tf.placeholder(tf.float32, [None, 1])

        # Define loss and optimization Op
        self.loss = tflearn.mean_square(self.predicted_q_value, self.out)
        self.optimize = tf.train.AdamOptimizer(
            self.learning_rate).minimize(self.loss)

        # Get the gradient of the net w.r.t. the action.
        # For each action in the minibatch (i.e., for each x in xs),
        # this will sum up the gradients of each critic output in the minibatch
        # w.r.t. that action. Each output is independent of all
        # actions except for one.
        self.action_grads = tf.gradients(self.out, self.action)

    def create_critic_network(self):
        inputs = tflearn.input_data(shape=[None, self.s_dim])
        action = tflearn.input_data(shape=[None, self.a_dim])
        net = tflearn.fully_connected(inputs, H1, activation='relu')

        # Add the action tensor in the 2nd hidden layer
        # Use two temp layers to get the corresponding weights and biases
        t1 = tflearn.fully_connected(net, H2)
        t2 = tflearn.fully_connected(action, H2)

        net = tflearn.activation(
            tf.matmul(net, t1.W) + tf.matmul(action, t2.W) + t2.b, activation='relu')

        # linear layer connected to 1 output representing Q(s,a)
        # Weights are init to Uniform[-3e-3, 3e-3]
        w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        out = tflearn.fully_connected(net, 1, weights_init=w_init)
        return inputs, action, out

    def train(self, inputs, action, predicted_q_value):
        return self.sess.run([self.out, self.optimize], feed_dict={
            self.inputs: inputs,
            self.action: action,
            self.predicted_q_value: predicted_q_value
        })

    def predict(self, inputs, action):
        return self.sess.run(self.out, feed_dict={
            self.inputs: inputs,
            self.action: action
        })

    def predict_target(self, inputs, action):
        return self.sess.run(self.target_out, feed_dict={
            self.target_inputs: inputs,
            self.target_action: action
        })

    def action_gradients(self, inputs, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.inputs: inputs,
            self.action: actions
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

# ===========================
#   Tensorflow Summary Ops
# ===========================


def build_summaries():
    episode_reward = tf.Variable(0.)
    tf.summary.scalar("Reward", episode_reward)
    episode_ave_max_q = tf.Variable(0.)
    tf.summary.scalar("Qmax Value", episode_ave_max_q)

    summary_vars = [episode_reward, episode_ave_max_q]
    summary_ops = tf.summary.merge_all()

    return summary_ops, summary_vars

# ===========================
#   Agent Training
# ===========================



def train(sess, env, actor, critic):

    # Set up summary Ops
    # summary_ops, summary_vars = build_summaries()

    sess.run(tf.global_variables_initializer())
    # writer = tf.summary.FileWriter(SUMMARY_DIR, sess.graph)

    # Initialize target network weights
    actor.update_target_network()
    critic.update_target_network()

    # Initialize replay memory

    replay_buffer_successes = Memory(BUFFER_SIZE)
    replay_buffer_failures = Memory(BUFFER_SIZE)

    for i in range(MAX_EPISODES):

        s = env.reset()

        ep_reward = 0
        ep_ave_max_q = 0

        exploration_weight = np.maximum(0, EXPLORATION_INITIAL_WEIGHT - i * EXPLORATION_DECREMENT_RATE)

        for j in range(MAX_EP_STEPS):

            a = actor.predict(np.reshape(s, (1, actor.s_dim)))

            exploration_noise = exploration_weight * OUP.function(a, MEAN, THETA, SIGMA)
            a = a + exploration_noise

            max_value =  np.array([1., 1.])
            min_value =  np.array([-1., -1.])
            a = np.minimum(a, max_value)
            a = np.maximum(a, min_value)

            s2, r, terminal, info = env.step(a[0])

            if i != 0 and i % RENDER_FREQUENCY == 0:
                env.render()

            a2 = actor.predict(np.reshape(s2, (1, actor.s_dim)))
            a2_target = actor.predict_target(np.reshape(s2, (1, actor.s_dim)))
            if terminal:
                error = abs(critic.predict(np.reshape(s2, (1, actor.s_dim)), a2) - r)[0][0]
            else:
                error = abs(critic.predict(np.reshape(s2, (1, actor.s_dim)), a2) - (r + GAMMA * critic.predict_target(np.reshape(s2, (1, actor.s_dim)), a2_target)))[0][0]

            # print("error", error)
            # print("reward", r)


            # import ipdb; ipdb.set_trace()
            if r > 0:
                replay_buffer_successes.add(error,  (np.reshape(s, (actor.s_dim,)), np.reshape(a, (actor.a_dim,)), r,
                                  terminal, np.reshape(s2, (actor.s_dim,))))
            else:
                replay_buffer_failures.add(error,  (np.reshape(s, (actor.s_dim,)), np.reshape(a, (actor.a_dim,)), r,
                                                     terminal, np.reshape(s2, (actor.s_dim,))))

            # Keep adding experience to the memory until
            # there are at least minibatch size samples

            if i > START_LEARNING:
            # if replay_buffer.total > MINIBATCH_SIZE:
            #     print("sample", replay_buffer.sample(MINIBATCH_SIZE))
            #     import ipdb; ipdb.set_trace()

                batch_number_successes = round(SUCCESS_RATIO * MINIBATCH_SIZE)
                batch_number_failures = MINIBATCH_SIZE - batch_number_successes

                batch_successes = replay_buffer_successes.sample(batch_number_successes)
                batch_failures = replay_buffer_failures.sample(batch_number_failures)

                # print(i)
                # print(batch_successes)
                # print(batch_failures)
                # errors, batch = replay_buffer.sample(MINIBATCH_SIZE)
                # errors, s_batch, a_batch, r_batch, t_batch, s2_batch = \
                #     replay_buffer.sample(MINIBATCH_SIZE)
                errors_batch_successes = np.array([ o[0] for o in batch_successes ])
                s_batch_successes = np.array([ o[1][0] for o in batch_successes ])
                a_batch_successes = np.array([ o[1][1] for o in batch_successes ])
                r_batch_successes = np.array([ o[1][2] for o in batch_successes ])
                t_batch_successes = np.array([ o[1][3] for o in batch_successes ])
                s2_batch_successes = np.array([ o[1][4] for o in batch_successes ])

                errors_batch_failures = np.array([ o[0] for o in batch_failures ])
                s_batch_failures = np.array([ o[1][0] for o in batch_failures ])
                a_batch_failures = np.array([ o[1][1] for o in batch_failures ])
                r_batch_failures = np.array([ o[1][2] for o in batch_failures ])
                t_batch_failures = np.array([ o[1][3] for o in batch_failures ])
                s2_batch_failures = np.array([ o[1][4] for o in batch_failures ])
                # print(batch)
                # print('errors', errors)
                # print('s_batch', s_batch)
                # print('a_batch', a_batch)
                # print('r_batch', r_batch)
                # print('t_batch', t_batch)
                # print('s2_batch', s2_batch)
                # import ipdb; ipdb.set_trace()

            #     replay_buffer.sample(MINIBATCH_SIZE)


                # SUCCESSES
                # Calculate targets
                target_q = critic.predict_target(
                    s2_batch_successes, actor.predict_target(s2_batch_successes))

                y_i = []
                for k in range(batch_number_successes):
                    if t_batch_successes[k]:
                        y_i.append(r_batch_successes[k])
                    else:
                        y_i.append(r_batch_successes[k] + GAMMA * target_q[k])

                # Update the critic given the targets
                predicted_q_value, _ = critic.train(
                    s2_batch_successes, a_batch_successes, np.reshape(y_i, (batch_number_successes, 1)))

                updated_errors = (np.reshape(y_i, (batch_number_successes, 1)) - predicted_q_value)
                # print(updated_errors)
                #update errors
                for index in range(len(errors_batch_successes)):
                    idx = errors_batch_successes[index]
                    replay_buffer_successes.update(idx, np.amax(abs(updated_errors[index][0])))

                #To be done make this include negatrives
                ep_ave_max_q += np.amax(predicted_q_value)


                # Update the actor policy using the sampled gradient
                a_outs = actor.predict(s2_batch_successes)
                grads = critic.action_gradients(s2_batch_successes, a_outs)
                actor.train(s2_batch_successes, grads[0])

                # Update target networks
                actor.update_target_network()
                critic.update_target_network()

                # FAILURES
                # Calculate targets
                target_q = critic.predict_target(
                    s2_batch_failures, actor.predict_target(s2_batch_failures))

                y_i = []
                for k in range(batch_number_failures):
                    if t_batch_failures[k]:
                        y_i.append(r_batch_failures[k])
                    else:
                        y_i.append(r_batch_failures[k] + GAMMA * target_q[k])

                # Update the critic given the targets
                predicted_q_value, _ = critic.train(
                    s2_batch_failures, a_batch_failures, np.reshape(y_i, (batch_number_failures, 1)))

                updated_errors = (np.reshape(y_i, (batch_number_failures, 1)) - predicted_q_value)
                # print(updated_errors)
                #update errors
                for index in range(len(errors_batch_failures)):
                    idx = errors_batch_failures[index]
                    replay_buffer_failures.update(idx, np.amax(abs(updated_errors[index][0])))

           #     TODO make this include negatrives
                # ep_ave_max_q += np.amax(predicted_q_value)

                # Update the actor policy using the sampled gradient
                a_outs = actor.predict(s_batch_failures)
                grads = critic.action_gradients(s2_batch_failures, a_outs)
                actor.train(s2_batch_failures, grads[0])

                # Update target networks
                actor.update_target_network()
                critic.update_target_network()



            s = s2
            ep_reward += r

            if terminal:
                #
                # summary_str = sess.run(summary_ops, feed_dict={
                #     summary_vars[0]: ep_reward,
                #     summary_vars[1]: ep_ave_max_q / float(j)
                # })

                # writer.add_summary(summary_str, i)
                # writer.flush()

                print( '| Reward: %.2i' % int(ep_reward), " | Episode", i, \
                    '| Qmax: %.4f' % (ep_ave_max_q / float(j)) )

                break


def main(_):
    with tf.Session() as sess:

        env = gym.make(ENV_NAME)
        np.random.seed(RANDOM_SEED)
        tf.set_random_seed(RANDOM_SEED)
        env.seed(RANDOM_SEED)

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        action_bound = env.action_space.high
        # print("action_bound")
        # print(action_bound)
        # Ensure action bound is symmetric
        assert (np.array_equal(env.action_space.low, np.array([-1, -1])))
        assert (np.array_equal(env.action_space.high, np.array([1, 1])))

        actor = ActorNetwork(sess, state_dim, action_dim, action_bound,
                             ACTOR_LEARNING_RATE, TAU)

        critic = CriticNetwork(sess, state_dim, action_dim,
                               CRITIC_LEARNING_RATE, TAU, actor.get_num_trainable_vars())

        if GYM_MONITOR_EN:
            if not RENDER_ENV:
                env = wrappers.Monitor(
                    env, MONITOR_DIR, video_callable=False, force=True)
            else:
                env = wrappers.Monitor(env, MONITOR_DIR, force=True)

        train(sess, env, actor, critic)

        if GYM_MONITOR_EN:
            env.monitor.close()

if __name__ == '__main__':
    tf.app.run()