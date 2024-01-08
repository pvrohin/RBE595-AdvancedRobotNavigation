class BayesFilter:
    def __init__(self, states, observations, transition_model, observation_model):
        self.states = states
        self.observations = observations
        self.transition_model = transition_model
        self.observation_model = observation_model

    def update(self, belief, action, observation):
        new_belief = {}
        for state in self.states:
            new_belief[state] = 0.0
            for prev_state in self.states:
                transition_prob = self.transition_model[prev_state][state]
                observation_prob = self.observation_model[state][observation]
                new_belief[state] += belief[prev_state] * transition_prob * observation_prob
        return new_belief

    def localize(self, initial_belief, actions, observations):
        belief = initial_belief
        for action, observation in zip(actions, observations):
            belief = self.update(belief, action, observation)
        return belief

def main():
    # Create an instance of BayesFilter
    states = ['A', 'B', 'C']
    observations = ['X', 'Y', 'Z']
    transition_model = {
        'A': {'A': 0.7, 'B': 0.2, 'C': 0.1},
        'B': {'A': 0.3, 'B': 0.5, 'C': 0.2},
        'C': {'A': 0.1, 'B': 0.4, 'C': 0.5}
    }
    observation_model = {
        'A': {'X': 0.6, 'Y': 0.3, 'Z': 0.1},
        'B': {'X': 0.1, 'Y': 0.7, 'Z': 0.2},
        'C': {'X': 0.3, 'Y': 0.4, 'Z': 0.3}
    }
    bayes_filter = BayesFilter(states, observations, transition_model, observation_model)

    # Define initial belief, actions, and observations
    initial_belief = {'A': 0.4, 'B': 0.3, 'C': 0.3}
    actions = ['action1', 'action2', 'action3']
    observations = ['X', 'Y', 'Z']

    # Perform localization
    belief = bayes_filter.localize(initial_belief, actions, observations)

    # Print the final belief
    print(belief)

if __name__ == "__main__":
    main()
