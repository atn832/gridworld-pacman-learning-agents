import mdp, util

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
  """
      * Please read learningAgents.py before reading this.*

      A ValueIterationAgent takes a Markov decision process
      (see mdp.py) on initialization and runs value iteration
      for a given number of iterations using the supplied
      discount factor.
  """
  def __init__(self, mdp, discount = 0.9, iterations = 100):
    """
      Your value iteration agent should take an mdp on
      construction, run the indicated number of iterations
      and then act according to the resulting policy.
    
      Some useful mdp methods you will use:
          mdp.getStates()
          mdp.getPossibleActions(state)
          mdp.getTransitionStatesAndProbs(state, action)
          mdp.getReward(state, action, nextState)
    """
    self.mdp = mdp
    self.discount = discount
    self.iterations = iterations
    self.values = util.Counter() # A Counter is a dict with default 0

    print ("iterations", iterations)
    self.qValues = util.Counter()
  
  def getValue(self, state):
    """
      Return the value of the state (computed in __init__).
    """
    maxQ = float("-inf")
    possibleActions = self.mdp.getPossibleActions(state)
    for action in possibleActions:
      q = self.getQValue(state, action)
      if (q > maxQ):
        maxQ = q

    return maxQ

  def getQValueR(self, state, action, i):
    #find max value,
    #sum(s to s') (proba(s,a,s') * (reward(s,a,s') + gamma * max(a') [q(s', a', i-1)])
    if i == 0:
      return self.values[state]#self.getValue(state)
    elif self.qValues[(state, action, i)]:
      return self.qValues[(state, action, i)]
    sum = 0
    statesAndProbs = self.mdp.getTransitionStatesAndProbs(state, action)
    for stateAndProb in statesAndProbs:
      nextState = stateAndProb[0]
      prob = stateAndProb[1]
      reward = self.mdp.getReward(state, action, nextState)

      # look for max a2
      maxQ_i_1 = float("-inf")
      foundSome = False
      actionsFromNextState = self.mdp.getPossibleActions(nextState)
      for a2 in actionsFromNextState:
        q2 = self.getQValueR(nextState, a2, i - 1)
        if q2 > maxQ_i_1:
          maxQ_i_1 = q2
          foundSome = True

      if not foundSome:
        maxQ_i_1 = 0

      sum = sum + prob * (reward + self.discount * maxQ_i_1)

    self.qValues[(state, action, i)] = sum
    return sum

  def getQValue(self, state, action):
    """
      The q-value of the state action pair
      (after the indicated number of value iteration
      passes).  Note that value iteration does not
      necessarily create this quantity and you may have
      to derive it on the fly.
    """
    return self.getQValueR(state, action, self.iterations);

  def getPolicy(self, state):
    """
      The policy is the best action in the given state
      according to the values computed by value iteration.
      You may break ties any way you see fit.  Note that if
      there are no legal actions, which is the case at the
      terminal state, you should return None.
    """
    actions = self.mdp.getPossibleActions(state)

    if not actions:
      return None

    bestAction = actions[0]
    bestQ = float("-inf")
    for a in actions:
      q = self.getQValue(state, a)
      if q > bestQ:
        bestAction = a
        bestQ = q

    # return the action that maximizes Q value
    print ("best action for", state, " is ", bestAction)
    return bestAction

  def getAction(self, state):
    "Returns the policy at the state (no exploration)."
    return self.getPolicy(state)
  
