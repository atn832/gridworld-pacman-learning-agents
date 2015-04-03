from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math
          
class QLearningAgent(ReinforcementAgent):
  """
    Q-Learning Agent
    
    Functions you should fill in:
      - getQValue
      - getAction
      - getValue
      - getPolicy
      - update
      
    Instance variables you have access to
      - self.epsilon (exploration prob)
      - self.alpha (learning rate)
      - self.gamma (discount rate)
    
    Functions you should use
      - self.getLegalActions(state) 
        which returns legal actions
        for a state
  """
  def __init__(self, **args):
    "You can initialize Q-values here..."
    ReinforcementAgent.__init__(self, **args)

    self.qValues = util.Counter()
  
  def getQValue(self, state, action):
    """
      Returns Q(state,action)    
      Should return 0.0 if we never seen
      a state or (state,action) tuple 
    """
    if not self.qValues[(state, action)]:
      return 0
    q = self.qValues[(state, action)]
    #print ("Q", state, action, q)
    return q

  def getValue(self, state):
    """
      Returns max_action Q(state,action)        
      where the max is over legal actions.  Note that if
      there are no legal actions, which is the case at the
      terminal state, you should return a value of 0.0.
    """
    maxQ = float("-inf")
    foundMax = False
    legalActions = self.getLegalActions(state)
    if not legalActions:
      return 0

    for action in legalActions:
      q = self.getQValue(state, action)
      if (q > maxQ):
        maxQ = q
        foundMax = True

    return maxQ
    
  def getPolicy(self, state):
    """
      Compute the best action to take in a state.  Note that if there
      are no legal actions, which is the case at the terminal state,
      you should return None.
    """
    legalActions = self.getLegalActions(state)
    #print ("get policy", state, "legal actions are", legalActions)
    if not legalActions:
      return None

    bestActions = None
    bestQ = float("-inf")
    for action in legalActions:
      q = self.getQValue(state, action)
      if q > bestQ:
        bestActions = [action]
        bestQ = q
      elif q == bestQ:
        bestActions.append(action)

    # return the action that maximizes Q value
    #print ("policy: best action for", state, " is ", bestAction)
    return random.choice(bestActions)
    
  def getAction(self, state):
    """
      Compute the action to take in the current state.  With
      probability self.epsilon, we should take a random action and
      take the best policy action otherwise.  Note that if there are
      no legal actions, which is the case at the terminal state, you
      should choose None as the action.
    
      HINT: You might want to use util.flipCoin(prob)
      HINT: To pick randomly from a list, use random.choice(list)
    """  
    legalActions = self.getLegalActions(state)
    if not legalActions:
      return None

    # Explore
    if util.flipCoin(self.epsilon):
      return random.choice(legalActions)

    # or pick the best action
    return self.getPolicy(state)
  
  def update(self, state, action, nextState, reward):
    """
      The parent class calls this to observe a 
      state = action => nextState and reward transition.
      You should do your Q-Value update here
      
      NOTE: You should never call this function,
      it will be called on your behalf
    """
    # Q(s, a) :alpha= reward + max_a'(q(s', a'))
    # print ("update", state, action, nextState, reward)
    max_Q_a2 = float("-inf")
    found = False
    legalActions = self.getLegalActions(nextState)
    for a2 in legalActions:
      q2 = self.getQValue(nextState, a2)
      if q2 > max_Q_a2:
        max_Q_a2 = q2
        found = True

    if not found:
      max_Q_a2 = 0
    self.qValues[(state, action)] = \
      (1 - self.alpha) * self.qValues[(state, action)] + \
      self.alpha * (reward + self.gamma * max_Q_a2)
    
class PacmanQAgent(QLearningAgent):
  "Exactly the same as QLearningAgent, but with different default parameters"
  
  def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
    """
    These default parameters can be changed from the pacman.py command line.
    For example, to change the exploration rate, try:
        python pacman.py -p PacmanQLearningAgent -a epsilon=0.1
    
    alpha    - learning rate
    epsilon  - exploration rate
    gamma    - discount factor
    numTraining - number of training episodes, i.e. no learning after these many episodes
    """
    args['epsilon'] = epsilon
    args['gamma'] = gamma
    args['alpha'] = alpha
    args['numTraining'] = numTraining
    print ("init", QLearningAgent)
    QLearningAgent.__init__(self, **args)

  def getAction(self, state):
    """
    Simply calls the getAction method of QLearningAgent and then
    informs parent of action for Pacman.  Do not change or remove this
    method.
    """
    action = QLearningAgent.getAction(self,state)
    self.doAction(state,action)
    return action
    
class ApproximateQAgent(PacmanQAgent):
  """
     ApproximateQLearningAgent
     
     You should only have to overwrite getQValue
     and update.  All other QLearningAgent functions
     should work as is.
  """
  def __init__(self, extractor='IdentityExtractor', **args):
    self.featExtractor = util.lookup(extractor, globals())()
    print ("Using extractor", extractor)
    PacmanQAgent.__init__(self, **args)

    # You might want to initialize weights here.
    self.weights = util.Counter()
    
  def getQValue(self, state, action):
    """
      Should return Q(state,action) = w * featureVector
      where * is the dotProduct operator
    """
    features = self.featExtractor.getFeatures(state, action)
    # print ("update with features. action is", action, features)

    #for feature in features:
     # print ("feature", feature, "value: ", features[feature])
    
    return sum(self.weights[feature] * features[feature] for feature in features)
    
  def update(self, state, action, nextState, reward):
    """
       Should update your weights based on transition  
    """
    features = self.featExtractor.getFeatures(state, action)
    correction = (reward + self.gamma * self.getValue(nextState)) - self.getQValue(state, action)
    for feature in features.keys():
        self.weights[feature] = self.weights[feature] + self.alpha * correction * features[feature]

    # return PacmanQAgent.update(self, state, action, nextState, reward)
    
  def final(self, state):
    "Called at the end of each game."
    # call the super-class final method
    PacmanQAgent.final(self, state)
    
    # did we finish training?
    if self.episodesSoFar == self.numTraining:
      # you might want to print your weights here for debugging
      "*** YOUR CODE HERE ***"
      pass
