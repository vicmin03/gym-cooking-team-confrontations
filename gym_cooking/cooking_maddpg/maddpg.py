import torch as T

# PARAMETERS
    # alpha = learning rate
    # epsilon = for balancing exploration + learning
    # discount = rate of discounting future rewards
    # decay = rate of decaying epsilong (so more learning than exploration)

class MADDPG():
    def __init__(self, alpha, epsilon, discount, decay, num_agents):
        self.alpha = alpha
        self.epsilon = epsilon
        self.discount = discount
        self.decay = decay
        self.num_agents = num_agents
        self.agents = []
        self.num_actions = 5   # action space for all agents is [(0, 0), (0, 1), (0, -1), (1, 0), (-1, 0)]

    def learn(self, memory):
    # initialise the replay buffer to batch size before learning can take place
        if not memory.ready():
            return

        actor_states, states, actions, rewards, \
        actor_new_states, states_, dones = memory.sample_buffer()

        device = self.agents[0].actor.device

        # turn the batch of samples into tensors so they can be passed into the network 
        states = T.tensor(states, dtype=T.float).to(device)
        actions = T.tensor(actions, dtype=T.float).to(device)
        rewards = T.tensor(rewards).to(device)
        states_ = T.tensor(states_, dtype=T.float).to(device)       
        dones = T.tensor(dones).to(device)          # indicates if a terminal state

        all_agents_new_actions = []
        # all_agents_new_mu_actions = []
        old_agents_actions = []

        for agent_idx, agent in enumerate(self.agents):
            new_states = T.tensor(actor_new_states[agent_idx], 
                                 dtype=T.float).to(device)

            # gets the new actions possible for actor in new state
            new_pi = agent.target_actor.forward(new_states)

            all_agents_new_actions.append(new_pi)

            # actions possible for actor in current state
            mu_states = T.tensor(actor_states[agent_idx], 
                                 dtype=T.float).to(device)
            pi = agent.actor.forward(mu_states)
            # all_agents_new_mu_actions.append(pi)

            # the actions that each agent actually performs
            old_agents_actions.append(actions[agent_idx])

        # concatenate and combine so correct input format for nn
        new_actions = T.cat([acts for acts in all_agents_new_actions], dim=1)
        # mu = T.cat([acts for acts in all_agents_new_mu_actions], dim=1)
        old_actions = T.cat([acts for acts in old_agents_actions],dim=1)

        for agent_idx, agent in enumerate(self.agents):
            critic_value_ = agent.target_critic.forward(states_, new_actions).flatten()
            critic_value_[dones[:,0]] = 0.0
            # find critic value for the actions that agents actually took
            critic_value = agent.critic.forward(states, old_actions).flatten()

            target = rewards[:,agent_idx] + agent.gamma*critic_value_
            # calulating the loss between prediction and target
            critic_loss = F.mse_loss(target, critic_value)
            agent.critic.optimizer.zero_grad()
            critic_loss.backward(retain_graph=True)
            agent.critic.optimizer.step()

            actor_loss = agent.critic.forward(states, mu).flatten()
            actor_loss = -T.mean(actor_loss)
            agent.actor.optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)
            agent.actor.optimizer.step()

            agent.update_network_parameters()
