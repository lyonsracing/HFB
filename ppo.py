import gym
import time

import numpy as np
import time
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import MultivariateNormal

class PPO:
	"""
		This is the PPO class we will use as our model in main.py
	"""
	def __init__(self, policy_class, env, human_feedback=0, **hyperparameters):
		"""
			Initializes the PPO model, including hyperparameters.
			Parameters:
				policy_class - the policy class to use for our actor/critic networks.
				env - the environment to train on.
				hyperparameters - all extra arguments passed into PPO that should be hyperparameters.
			Returns:
				None
		"""
		# Make sure the environment is compatible with our code
		assert(type(env.observation_space) == gym.spaces.Box)
		assert(type(env.action_space) == gym.spaces.Box)

		# Initialize hyperparameters for training with PPO
		self._init_hyperparameters(hyperparameters)

		# Extract environment information
		self.env = env
		self.obs_dim = env.observation_space.shape[0]
		self.act_dim = env.action_space.shape[0]
		# Including human feedback
		self.human_feedback = human_feedback

		 # Initialize actor and critic networks
		self.actor = policy_class(self.obs_dim, self.act_dim)   # ALG STEP 1
		self.critic = policy_class(self.obs_dim, 1)

		# Initialize optimizers for actor and critic
		self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
		self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)

		# Initialize the covariance matrix used to query the actor for actions
		self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.5)
		self.cov_mat = torch.diag(self.cov_var)

		# This logger will help us with printing out summaries of each iteration
		self.logger = {
			'delta_t': time.time_ns(),
			't_so_far': 0,          # timesteps so far
			'i_so_far': 0,          # iterations so far
			'batch_lens': [],       # episodic lengths in batch
			'batch_rews': [],       # episodic returns in batch
			'actor_losses': [],     # losses of actor network in current iteration
		}

	def learn(self, total_timesteps):
		"""
			Train the actor and critic networks. Here is where the main PPO algorithm resides.
			Parameters:
				total_timesteps - the total number of timesteps to train for
			Return:
				None
		"""
		print(f"Learning... Running {self.max_timesteps_per_episode} timesteps per episode, ", end='')
		print(f"{self.timesteps_per_batch} timesteps per batch for a total of {total_timesteps} timesteps")
		t_so_far = 0 # Timesteps simulated so far
		i_so_far = 0 # Iterations ran so far
		while t_so_far < total_timesteps:                                                      # ALG STEP 2
			# Collecting our batch simulations here
			batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = self.rollout()    # ALG STEP 3

			# Calculate how many timesteps we collected this batch
			t_so_far += np.sum(batch_lens)

			# Increment the number of iterations
			i_so_far += 1

			# Logging timesteps so far and iterations so far
			self.logger['t_so_far'] = t_so_far
			self.logger['i_so_far'] = i_so_far

			# Calculate advantage at k-th iteration
			V, _ = self.evaluate(batch_obs, batch_acts)	
			A_k = batch_rtgs - V.detach()        # ALG STEP 5
			
			# Include human feedback
			A_k += self.human_feedback                                                          

			# One of the only tricks I use that isn't in the pseudocode. Normalizing advantages
			# isn't theoretically necessary, but in practice it decreases the variance of 
			# our advantages and makes convergence much more stable and faster. I added this because
			# solving some environments was too unstable without it.
			A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

			# This is the loop where we update our network for some n epochs
			for _ in range(self.n_updates_per_iteration):                                        # ALG STEP 6 & 7
				# Calculate V_phi and pi_theta(a_t | s_t)
				V, curr_log_probs = self.evaluate(batch_obs, batch_acts)

				# Calculate the ratio pi_theta(a_t | s_t) / pi_theta_k(a_t | s_t)
				# We just subtract the logs, which is the same as
				# dividing the values and then canceling the log with e^log.
				# For why we use log probabilities instead of actual probabilities,
				# here's a great explanation: 
				# https://cs.stackexchange.com/questions/70518/why-do-we-use-the-log-in-gradient-based-reinforcement-algorithms
				# TL;DR makes gradient ascent easier behind the scenes.
				ratios = torch.exp(curr_log_probs - batch_log_probs)

				# Calculate surrogate losses.
				surr1 = ratios * A_k
				surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k

				# Calculate actor and critic losses.
				# We take the negative min of the surrogate losses because we're trying to maximize
				# the performance function, but Adam minimizes the loss. So minimizing the negative
				# performance function maximizes it.
				actor_loss = (-torch.min(surr1, surr2)).mean()
				critic_loss = nn.MSELoss()(V, batch_rtgs + self.human_feedback)

				# Calculate gradients and perform backward propagation for actor network
				self.actor_optim.zero_grad()
				actor_loss.backward(retain_graph=True)
				self.actor_optim.step()

				# Calculate gradients and perform backward propagation for critic network
				self.critic_optim.zero_grad()
				critic_loss.backward()
				self.critic_optim.step()

				# Log actor loss
				self.logger['actor_losses'].append(actor_loss.detach())

			# Print a summary of our training so far
			self._log_summary()

			# Save our model if it's time
			if i_so_far % self.save_freq == 0:
				torch.save(self.actor.state_dict(), './ppo_actor.pth')
				torch.save(self.critic.state_dict(), './ppo_critic.pth')

	def rollout(self):
		"""
			This is where we collect the batch of data from simulation. Since this 
			is an on-policy algorithm, we'll need to collect a fresh batch
			of data each time we iterate the actor/critic networks.
			Parameters:
				None
			Return:
				batch_obs - the observations collected this batch. Shape: (number of timesteps, dimension of observation)
				batch_acts - the actions collected this batch. Shape: (number of timesteps, dimension of action)
				batch_log_probs - the log probabilities of each action taken this batch. Shape: (number of timesteps)
				batch_rtgs - the Rewards-To-Go of each timestep in this batch. Shape: (number of timesteps)
				batch_lens - the lengths of each episode this batch. Shape: (number of episodes)
		"""
		# Batch data. For more details, check function header.
		batch_obs = []
		batch_acts = []
		batch_log_probs = []
		batch_rews = []
		batch_rtgs = []
		batch_lens = []

		# Episodic data. Keeps track of rewards per episode, will get cleared
		# upon each new episode
		ep_rews = []

		t = 0 # Keeps track of how many timesteps we've run so far this batch

		# Keep simulating until we've run more than or equal to specified timesteps per batch
		while t < self.timesteps_per_batch:
			ep_rews = [] # rewards collected per episode

			# Reset the environment. Note that obs is short for observation. 
			obs = self.env.reset()
			done = False

			# Run an episode for a maximum of max_timesteps_per_episode timesteps
			for ep_t in range(self.max_timesteps_per_episode):
				# If render is specified, render the environment
				if self.render and (self.logger['i_so_far'] % self.render_every_i == 0) and len(batch_lens) == 0:
					self.env.render()

				t += 1 # Increment timesteps ran this batch so far

				#Debugging: check the caller of get action
				##print("Observation passed to get action: ", obs)

				# Calculate action and make a step in the env. 
				# Note that rew is short for reward.
				action, log_prob = self.get_action(obs)

				# Debugging: check action shape and type as well as observation space
				##print("original action shape: ", action.shape)
				if action.shape != (1,):
					action = action.reshape(-1)
				##print("reshaped action shape: ", action.shape)
				action = action.astype('float32')

				##print("action space: ", self.env.action_space)
				##print("observation space: ", self.env.observation_space)

				# Debugging: Check assertation error
				try:
					step_output = self.env.step(action)
				except AssertionError:
					##print("Offending observation: ",obs)
					##print("offending action: ", action)
					raise

				# Debugging value error
				##print("Step output: ", step_output)
				##print("Length of step output: ", len(step_output))

				# Debugging the step output before unpacking
				##print("Step output before unpacking: ", step_output)

				# Unpack the values from the step_output
				obs, rew, done, _, _ = step_output

				# Debugging the observation and reward after unpacking
				##print("observation after unpacking: ", obs)
				##print("reward after unpacking: ", rew)

				# Track observations in this batch
				batch_obs.append(obs)

				# Track recent reward, action, and action log probability
				ep_rews.append(rew)
				batch_acts.append(action)
				batch_log_probs.append(log_prob)

				# Debugging: print the step result
				step_result = self.env.step(action)
				##print("Step result: ", step_result)
				# Unpack the step result
				obs, rew, done, _= step_result[:4]
				# Debugging: print the observation and reward
				##print("Observation: ", obs)
				##print("Reward: ", rew)

				# Adding a delay or 0.1 seconds for training purposes
				time.sleep(0.1)
				# If the environment tells us the episode is terminated, break
				if done:
					break

			# Track episodic lengths and rewards
			batch_lens.append(ep_t + 1)
			batch_rews.append(ep_rews)

		# Reshape data as tensors in the shape specified in function description, before returning
		#batch_obs = torch.tensor(batch_obs, dtype=torch.float)
		#batch_acts = torch.tensor(batch_acts, dtype=torch.float)
		#batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float)
		#batch_rtgs = self.compute_rtgs(batch_rews)                                    # ALG STEP 4

		##Debugging: check types and shapes before converting to tensor
		#print("Types of elements in batch_log_probs: ", [type(x) for x in batch_log_probs])
		#print("Shapes of elements in batch_log_probs: ", [x.shape if hasattr(x,'shape') else None for x in batch_log_probs])
		#print("Type of elements in batch_obs: ", [type(x) for x in batch_obs])
		#print("Shapes of elements in batch_obs: ", [x.shape if hasattr(x, 'shape') else None for x in batch_obs])
		#print("Types of elements in batch_acts: ", [type(x) for x in batch_acts])
		#print("Shapes of elemtns in batch_acts: ", [x.shape if hasattr(x, 'shape') else None for x in batch_acts])
		
		## Trouble shooting the long delay between training iterations
		## Reshape the data as tensors in the shape specified in frunction 
		## description before running
		batch_obs = torch.tensor(np.array(batch_obs), dtype=torch.float)
		batch_acts = torch.tensor(np.array(batch_acts), dtype=torch.float)
		batch_log_probs = torch.stack(batch_log_probs).squeeze()
		batch_rtgs = self.compute_rtgs(batch_rews) ##Assuming this function already returns a tensor

		# Log the episodic returns and episodic lengths in this batch.
		self.logger['batch_rews'] = batch_rews
		self.logger['batch_lens'] = batch_lens

		return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens

	def compute_rtgs(self, batch_rews):
		"""
			Compute the Reward-To-Go of each timestep in a batch given the rewards.
			Parameters:
				batch_rews - the rewards in a batch, Shape: (number of episodes, number of timesteps per episode)
			Return:
				batch_rtgs - the rewards to go, Shape: (number of timesteps in batch)
		"""
		# The rewards-to-go (rtg) per episode per batch to return.
		# The shape will be (num timesteps per episode)
		batch_rtgs = []

		# Iterate through each episode
		for ep_rews in reversed(batch_rews):

			discounted_reward = 0 # The discounted reward so far

			# Iterate through all rewards in the episode. We go backwards for smoother calculation of each
			# discounted return (think about why it would be harder starting from the beginning)
			for rew in reversed(ep_rews):
				discounted_reward = rew + discounted_reward * self.gamma
				batch_rtgs.insert(0, discounted_reward)

		# Convert the rewards-to-go into a tensor
		batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)

		return batch_rtgs
	
	def get_action(self, obs):
		# Debugging: check for empty or non values
		if obs is None or len(obs) == 0:
			##print("Observation is empty or None.")
			return None, None
		# Debugging: check the shape pf the original obs
		##print("shape of original obs: ", np.shape(obs))
		# Exrtact the numpy array from the tuple
		if isinstance(obs, tuple):
			obs_array, _ = obs
		else:
			obs_array = obs

		##obs_array = obs

		# Debugging: checking for the shape and type of obs
		##print("Content of obs: ", obs)
		##print("Types of elements in obs: ", [type(element) for element in obs])
		##print("Shape of obs: ", np.shape(obs_array))
		##print("Type of obs: ", type(obs))
		
		# Debugging: check the shape of obs_array
		##print(np.shape(obs_array))
		##print(type(obs_array))
		##print(obs_array)

		# Convert the NumPy array to a tensor
		obs_tensor = torch.tensor(obs_array, dtype=torch.float32)
		
		# Add a batch dimension if its not there
		if len(obs_tensor.shape) == 1:
			obs_tensor = obs_tensor.unsqueeze(0)

		## Reshape the tensor to ensure it has the correct shape
		obs_tensor = obs_tensor.view(-1, 3)  # Reshape to ensure it has shape (batch_size, 3)

		# Query the actor network for the main action
		mean = self.actor(obs_tensor)
		# Create a distribution with the mean action and stf from the covariance matrix above 
		dist = MultivariateNormal(mean, self.cov_mat)
		# Sample an action from the distribution
		action = dist.sample()
		# Calculate the log probability for that action
		log_prob = dist.log_prob(action)
		# Return the sampled action and the log probability of that action in our distribution
		return action.detach().numpy(), log_prob.detach()

	def _init_hyperparameters(self, hyperparameters):
		"""
			Initialize default and custom values for hyperparameters
			Parameters:
				hyperparameters - the extra arguments included when creating the PPO model, should only include
									hyperparameters defined below with custom values.
			Return:
				None
		"""
		# Initialize default values for hyperparameters
		# Algorithm hyperparameters
		self.timesteps_per_batch = 4800                 # Number of timesteps to run per batch
		self.max_timesteps_per_episode = 1600           # Max number of timesteps per episode
		self.n_updates_per_iteration = 5                # Number of times to update actor/critic per iteration
		self.lr = 0.005                                 # Learning rate of actor optimizer
		self.gamma = 0.95                               # Discount factor to be applied when calculating Rewards-To-Go
		self.clip = 0.2                                 # Recommended 0.2, helps define the threshold to clip the ratio during SGA

		# Miscellaneous parameters
		self.render = True                              # If we should render during rollout
		self.render_every_i = 10                        # Only render every n iterations
		self.save_freq = 10                             # How often we save in number of iterations
		self.seed = None                                # Sets the seed of our program, used for reproducibility of results

		# Change any default values to custom values for specified hyperparameters
		for param, val in hyperparameters.items():
			exec('self.' + param + ' = ' + str(val))

		# Sets the seed if specified
		if self.seed != None:
			# Check if our seed is valid first
			assert(type(self.seed) == int)

			# Set the seed 
			torch.manual_seed(self.seed)
			print(f"Successfully set seed to {self.seed}")
			
	def _log_summary(self):
		"""
			Print to stdout what we've logged so far in the most recent batch.
			Parameters:
				None
			Return:
				None
		"""
		# Calculate logging values. I use a few python shortcuts to calculate each value
		# without explaining since it's not too important to PPO; feel free to look it over,
		# and if you have any questions you can email me (look at bottom of README)
		delta_t = self.logger['delta_t']
		self.logger['delta_t'] = time.time_ns()
		delta_t = (self.logger['delta_t'] - delta_t) / 1e9
		delta_t = str(round(delta_t, 2))

		t_so_far = self.logger['t_so_far']
		i_so_far = self.logger['i_so_far']
		avg_ep_lens = np.mean(self.logger['batch_lens'])
		avg_ep_rews = np.mean([np.sum(ep_rews) for ep_rews in self.logger['batch_rews']])
		avg_actor_loss = np.mean([losses.float().mean() for losses in self.logger['actor_losses']])

		# Round decimal places for more aesthetic logging messages
		avg_ep_lens = str(round(avg_ep_lens, 2))
		avg_ep_rews = str(round(avg_ep_rews, 2))
		avg_actor_loss = str(round(avg_actor_loss, 5))

		# Print logging statements
		print(flush=True)
		print(f"-------------------- Iteration #{i_so_far} --------------------", flush=True)
		print(f"Average Episodic Length: {avg_ep_lens}", flush=True)
		print(f"Average Episodic Return: {avg_ep_rews}", flush=True)
		print(f"Average Loss: {avg_actor_loss}", flush=True)
		print(f"Timesteps So Far: {t_so_far}", flush=True)
		print(f"Iteration took: {delta_t} secs", flush=True)
		print(f"------------------------------------------------------", flush=True)
		print(flush=True)

		# Reset batch-specific logging data
		self.logger['batch_lens'] = []
		self.logger['batch_rews'] = []
		self.logger['actor_losses'] = []

	# Within the 'learn' function, 'evaluate' is being called.
	# An 'evaluate' function is developed to evaluate the current policy
	# This function evaluates:
	# The Value estimation: estimated value of each state in the 
	# batch 'V' according to the current critic network.
	# The log probability calculation: log probability of taking each action
	# in the batch 'log_probs' according to the current policy (actor network)
	def evaluate(self, batch_obs, batch_acts):
		"""Calculate the value of an observation state and the log probabilities
		of an action taken in that state, under the current policy"""
		# Query critic network for the value function
		V = self.critic(batch_obs).squeeze()
		# Calculate log probability of a set of actions under the current policy
		mean = self.actor(batch_obs)
		dist =MultivariateNormal(mean, self.cov_mat)
		log_probs = dist.log_prob(batch_acts)

		return V, log_probs
