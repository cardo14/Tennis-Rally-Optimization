# Tennis-Rally-Optimization


Inspired by game theoretical optimization of pitch sequencing in baseball and rally optimization in table tennis, we use a Hidden Markov Model and Recurrent Neural Network to estimate win probability based on shot selection using Jeff Sackman's match charting data.


models/markov_kartik contains a basic starting point Markov model that updates win probability as a point moves through states, where each shot represents a state in the point.


