import sys
from collections import defaultdict
import math


class POMDPViterbi:
    
    
    def __init__(self):
        # State probabilities: P(s)
        self.initial_state_probs = {}
        
        # Transition probabilities: P(s'|s,a)
        self.transition_probs = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        
        # Observation probabilities: P(o|s)
        self.observation_probs = defaultdict(lambda: defaultdict(float))
        
        # Lists to maintain order
        self.states = []
        self.observations = []
        self.actions = []
        
    def parse_state_weights(self, filename):
        """Parse initial state weights and normalize to probabilities"""
        with open(filename, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]
            
            # Skip header
            header = lines[0]
            
            # Parse count and default weight
            parts = lines[1].split()
            num_states = int(parts[0])
            default_weight = float(parts[1]) if len(parts) > 1 else 0
            
            # Parse state weights
            total_weight = 0
            state_weights = {}
            
            for i in range(2, min(2 + num_states, len(lines))):
                parts = lines[i].split()
                state = parts[0].strip('"')
                weight = float(parts[1])
                state_weights[state] = weight
                total_weight += weight
                if state not in self.states:
                    self.states.append(state)
            
            # Normalize to get probabilities
            if total_weight > 0:
                for state in state_weights:
                    self.initial_state_probs[state] = state_weights[state] / total_weight
            
    def parse_state_action_state_weights(self, filename):
        """Parse state transition weights and normalize to conditional probabilities"""
        with open(filename, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]
            
            # Skip header
            header = lines[0]
            
            # Parse counts and default weight
            parts = lines[1].split()
            num_triples = int(parts[0])
            num_states = int(parts[1])
            num_actions = int(parts[2])
            default_weight = float(parts[3]) if len(parts) > 3 else 0
            
            # First, collect all weights
            weights = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
            
            # Parse transition weights
            for i in range(2, min(2 + num_triples, len(lines))):
                parts = lines[i].split()
                if len(parts) >= 4:
                    state1 = parts[0].strip('"')
                    action = parts[1].strip('"')
                    state2 = parts[2].strip('"')
                    weight = float(parts[3])
                    
                    weights[state1][action][state2] = weight
                    
                    # Track states and actions
                    if state1 not in self.states:
                        self.states.append(state1)
                    if state2 not in self.states:
                        self.states.append(state2)
                    if action not in self.actions:
                        self.actions.append(action)
            
            # Add default weights for missing transitions
            for state1 in self.states:
                for action in self.actions:
                    # Calculate total weight for normalization
                    total_weight = sum(weights[state1][action].values())
                    
                    # Add default weight for missing transitions
                    for state2 in self.states:
                        if state2 not in weights[state1][action]:
                            weights[state1][action][state2] = default_weight
                            total_weight += default_weight
                    
                    # Normalize to get conditional probabilities P(s'|s,a)
                    if total_weight > 0:
                        for state2 in self.states:
                            self.transition_probs[state1][action][state2] = \
                                weights[state1][action][state2] / total_weight
                    
    def parse_state_observation_weights(self, filename):
        """Parse observation weights and normalize to conditional probabilities"""
        with open(filename, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]
            
            # Skip header
            header = lines[0]
            
            # Parse counts and default weight
            parts = lines[1].split()
            num_pairs = int(parts[0])
            num_states = int(parts[1])
            num_observations = int(parts[2])
            default_weight = float(parts[3]) if len(parts) > 3 else 0
            
            # First, collect all weights
            weights = defaultdict(lambda: defaultdict(float))
            
            # Parse observation weights
            for i in range(2, min(2 + num_pairs, len(lines))):
                parts = lines[i].split()
                if len(parts) >= 3:
                    state = parts[0].strip('"')
                    observation = parts[1].strip('"')
                    weight = float(parts[2])
                    
                    weights[state][observation] = weight
                    
                    # Track states and observations
                    if state not in self.states:
                        self.states.append(state)
                    if observation not in self.observations:
                        self.observations.append(observation)
            
            # Add default weights and normalize
            for state in self.states:
                total_weight = sum(weights[state].values())
                
                # Add default weight for missing observations
                for obs in self.observations:
                    if obs not in weights[state]:
                        weights[state][obs] = default_weight
                        total_weight += default_weight
                
                # Normalize to get conditional probabilities P(o|s)
                if total_weight > 0:
                    for obs in self.observations:
                        self.observation_probs[state][obs] = weights[state][obs] / total_weight
                        
    def parse_observation_actions(self, filename):
        """Parse the sequence of observations and actions"""
        observation_action_sequence = []
        
        with open(filename, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]
            
            # Skip header
            header = lines[0]
            
            # Parse count
            num_pairs = int(lines[1])
            
            # Parse observation-action pairs
            for i in range(2, min(2 + num_pairs, len(lines))):
                parts = lines[i].split()
                if len(parts) >= 1:
                    observation = parts[0].strip('"')
                    # Action might be implicit (last observation doesn't have action)
                    action = parts[1].strip('"') if len(parts) > 1 else None
                    observation_action_sequence.append((observation, action))
                    
        return observation_action_sequence
    
    def viterbi(self, observation_action_sequence):
        """
        Implement the Viterbi algorithm to find the most likely state sequence
        """
        T = len(observation_action_sequence)
        
        # Initialize Viterbi tables
        # viterbi[t][state] = maximum probability of being in state at time t
        viterbi = [{} for _ in range(T)]
        # backpointer[t][state] = best previous state
        backpointer = [{} for _ in range(T)]
        
        # Initialization (t=0)
        first_obs = observation_action_sequence[0][0]
        for state in self.states:
            # Initial probability * observation probability
            viterbi[0][state] = self.initial_state_probs.get(state, 0) * \
                               self.observation_probs[state].get(first_obs, 0)
            backpointer[0][state] = None
        
        # Recursion (t=1 to T-1)
        for t in range(1, T):
            current_obs = observation_action_sequence[t][0]
            prev_action = observation_action_sequence[t-1][1]
            
            if prev_action is None:
                # Handle case where there's no action (shouldn't happen in normal case)
                continue
                
            for curr_state in self.states:
                max_prob = 0
                best_prev_state = None
                
                # Find the best previous state
                for prev_state in self.states:
                    # Previous Viterbi probability * transition probability * observation probability
                    prob = viterbi[t-1][prev_state] * \
                           self.transition_probs[prev_state][prev_action].get(curr_state, 0) * \
                           self.observation_probs[curr_state].get(current_obs, 0)
                    
                    if prob > max_prob:
                        max_prob = prob
                        best_prev_state = prev_state
                
                viterbi[t][curr_state] = max_prob
                backpointer[t][curr_state] = best_prev_state
        
        # Termination - find the best final state
        best_final_state = None
        max_final_prob = 0
        
        for state in self.states:
            if viterbi[T-1][state] > max_final_prob:
                max_final_prob = viterbi[T-1][state]
                best_final_state = state
        
        # Backtrack to find the best path
        best_path = []
        state = best_final_state
        
        for t in range(T-1, -1, -1):
            best_path.append(state)
            state = backpointer[t][state]
        
        best_path.reverse()
        
        return best_path
    
    def write_output(self, state_sequence, filename):
        """Write the state sequence to output file"""
        with open(filename, 'w') as f:
            f.write("states\n")
            f.write(f"{len(state_sequence)}\n")
            for state in state_sequence:
                f.write(f'"{state}"\n')


def main():
    """Main function to run the POMDP Viterbi algorithm"""
    
    # Create POMDP solver instance
    solver = POMDPViterbi()
    
    # Parse input files
    solver.parse_state_weights("state_weights.txt")
    solver.parse_state_action_state_weights("state_action_state_weights.txt")
    solver.parse_state_observation_weights("state_observation_weights.txt")
    observation_action_sequence = solver.parse_observation_actions("observation_actions.txt")
    
    # Run Viterbi algorithm
    best_state_sequence = solver.viterbi(observation_action_sequence)
    
    # Write output
    solver.write_output(best_state_sequence, "states.txt")


if __name__ == "__main__":
    main()
