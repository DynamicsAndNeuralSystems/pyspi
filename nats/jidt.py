from jpype import *
import numpy as np

class jidt():

    def __init__(self):
        jarloc = "./nats/lib/jidt/infodynamics.jar"
        startJVM(getDefaultJVMPath(), "-ea", "-Djava.class.path=" + jarloc)

    def te(self,x,y):
        calc_class = JPackage("infodynamics.measures.continuous.kernel").TransferEntropyCalculatorKernel
        calc = calc_class()
        calc.setProperty("NORMALISE", "true") # Normalise the individual variables
        calc.initialise(1, 0.5)
        calc.setObservations(JArray(JDouble,1)(y.tolist()), JArray(JDouble,1)(x.tolist()))
        return calc.computeAverageLocalOfObservations()

    def mi(self,x,y):
        calc_class = JPackage("infodynamics.measures.continuous.kernel").MutualInfoCalculatorMultiVariateKernel
        calc = calc_class()
        calc.setProperty("NORMALISE", "true") # Normalise the individual variables
        calc.initialise(1, 1, 0.5)
        calc.setObservations(JArray(JDouble,1)(y.tolist()), JArray(JDouble,1)(x.tolist()))
        return calc.computeAverageLocalOfObservations()
