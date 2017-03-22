import numpy as np

np.random.seed(1505367203)
print "Learnning rate:"+str(np.math.exp(np.random.uniform(-7.5,-4.5)))
print "Number of layers:"+str(np.random.choice(range(1,6)))
print "Number of hidden units per layer:"+str(np.random.choice(range(100,501)))
print "Weight Decay Coefficient:"+str(np.math.exp(np.random.uniform(-9,-6)))
print "Drop Out:"+str(np.random.choice(["Yes","No"]))

