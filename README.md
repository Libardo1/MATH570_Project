Project Initial Report

Gissella Bejarano & Erik Langert



Dataset Description


Australian Sign Languages Signs (UCI)


Contains data collected using (hardware):


Two Fifth Dimension Technologies gloves 


Two Ascension Flock-of-Birds magnetic position trackers


Data Characteristics - Multivariate, Time Series


Number of Instances (n) - 2565


Number of Attributes (p) - 22 (all per time step)


Number of Classes   (K)  - 95 (27 samples per class)


Detailed Description

x: Continuous. x position between -1 and 1.

Y: Continuous. y position between -1 and 1.     

z:  Continuous. z position between -1 and 1. 

roll: Continuous.  Roll with 0 meaning "palm down", rotating clockwise through to a maximum of 1 (not included), which is also "palm down". 

thumb: Continuous.  Thumb bend. has a value of 0 (straight) to 1 (fully bent). 

Fore: Continuous. Forefinger bend. has a value of 0 (straight) to 1 (fully bent). 

index: Index finger bend. has a value of 0 (straight) to 1 (fully bent). 

ring: Continuous. Ring finger bend. has a value of 0 (straight) to 1 (fully bent). 

Suggested Values to Ignore: pitch:  Has a value of -1, indicating that it is not available for this data. yaw: Has a value of -1, indicating that it is not available for this data little: In this case, it is a copy of ring bend. keycode: Indicates which key was pressed on the glove. gs1: Glove state 1. gs2: Glove state 2. . 

Research Questions

Can we infer class (word) of a new sign not used in training? What should we expect our classification accuracy to be?

How to best leverage the timed nature of our data? Can the sequential aspect of our data give us better classification results.

Are there correlations between our variables which improve time / space complexity for our models.

The providers of the dataset state some features are unimportant. Can we show any other features that are also extraneous?

Possible Methods

K Nearest Neighbor as a classifier.

Train Support Vector Machines per timestep per class.

Canonical Correlation Analysis of variables within classes (finger data vs. wrist data).

Principal Components Analysis to reduce dimensionality.
