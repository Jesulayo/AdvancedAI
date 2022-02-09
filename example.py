class TransportationMDP(object):
    def __init__(self, N):
        self.N = N
    def startState(self):
        return 1
    def isEnd(self, state):
        return state == self.N
    def actions(self, state):
        result = []
        if state+1 <= self.N:
            result.append('walk')
        if state*2 <= self.N:
            result.append('tram')
        return result
    def succProbReward(self, state, action):
        #return list of (newState, prob, reward)
        #state = s, action = a, newState = s'
        #prob = T(s,a,s'), reward = Reward(s,a,s')
        result = []
        if action == 'walk':
            result.append((state+1, 1., -1.))
        elif action == 'tram':
            result.append((state*2, 0.5, -2.))
            result.append((state, 0.5, -2.))
        return result
    def discount(self):
        return 1
    def states(self):
        return range(1, self.N+1)

mdp = TransportationMDP(N=10)
print(mdp.actions(3))