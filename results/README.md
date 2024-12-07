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
* graph_data.png : shows the Graph data used for the backward manifold aware vector field.


# manifold_aware_visualisation

Shows the different ingredient to the custom vector field

* euclidian_grad.png : y-x vector field
* projection_field.png : pi(x) - x vector field
* manifold_geodesic_field.png : geotangent(pi(x)) in y's direction.
* manifold_aware_vector_field.png : the combinaison of all above with alpha = 0.4 (euc_grad), beta = 0.2 (proj_field), gamma = 0.4 (manifold_field)



## Moon sinkhorn

Moon distribution:
    Provides two visualization over the same point distributions for moon distribution.

    * euclidean_moon.png shows the Sinkhorn's final dual coefficients as the color for the two data distributions
    * geodesic_moon.png shows the geodesic-aware Sinkhorn's final dual coefficients as the color for the two data distributions

Circle distribution:
    Provides two visualization over the same point distributions for circle distribution.

    * euclidean_circles.png shows the Sinkhorn's final dual coefficients as the color for the two data distributions
    * geodesic_circles.png shows the geodesic-aware Sinkhorn's final dual coefficients as the color for the two data distributions