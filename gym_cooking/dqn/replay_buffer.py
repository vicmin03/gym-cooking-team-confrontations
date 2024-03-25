import numpy as np




# class for replay buffer storing history of the environment and agent actions

class ReplayBuffer:
    def __init__(self, buffer_size, num_agents, obs_dims, batch_size):
        self.buffer_size = buffer_size
        self.num_agents = num_agents
        self.buffer = []
        self.max_index = 0
        self.obs_dims = obs_dims        # the dimensions of each observation stored in the buffer
        self.batch_size = batch_size


    def add(self, state, actions, rewards):
        # adds an observation to the replay buffer including: environment state, agent actions, corresponding rewards
        
        experience = (state, actions, rewards)

        # add new experience to buffer, wrapping round if buffer is full
        self.max_index += 1
        self.buffer[self.buffer_size % self.max_index] = experience


    # empties the buffer
    def clear(self):
        self.buffer = []
        self.max_index = 0

    # store details of a transition in buffer 
    def store_transition(self, raw_obs, state, action, reward, 
                               raw_obs_, state_, done):

        
        index = self.mem_cntr % self.mem_size

        for agent_idx in range(self.n_agents):
            self.actor_state_memory[agent_idx][index] = raw_obs[agent_idx]
            self.actor_new_state_memory[agent_idx][index] = raw_obs_[agent_idx]
            self.actor_action_memory[agent_idx][index] = action[agent_idx]

        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.mem_cntr += 1



    def sample_buffer(self):
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, self.batch_size, replace=False)

        states = self.state_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]

        actor_states = []
        actor_new_states = []
        actions = []
        for agent_idx in range(self.n_agents):
            actor_states.append(self.actor_state_memory[agent_idx][batch])
            actor_new_states.append(self.actor_new_state_memory[agent_idx][batch])
            actions.append(self.actor_action_memory[agent_idx][batch])

        return actor_states, states, actions, rewards, \
               actor_new_states, states_, terminal
