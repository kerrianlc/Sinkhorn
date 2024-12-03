# Result directory

Contains some graphical results used in the report. All those plots can be regenerated
using notebooks and python scripts in the src directory.

## 7_to_0

Show vector field flow from a 7 test image (not present in the data)
to a 0 test image (not present in the data).

Hyperparameters : 
alpha = 0.4
beta = 0.2
gamma = 0.4

# horned_sinkhorn_custom_vector_field

Shows two different sinkhorn gradient descent.

* euclidian_plots.png : The sinkhorn gradient descent using euclidian cost function.
* manifold_aware_plots.png : The sinkhorn gradient descent using euclidian backward modified cost function.   
