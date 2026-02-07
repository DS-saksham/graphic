# Visualization of Optimization Algorithms: Gradient Descent vs Adam

This project provides a 3D visualization of how different optimization algorithms—specifically Gradient Descent (GD) and Adam—traverse a non-linear loss surface. By animating their trajectories, we gain intuitive insights into their convergence speed, stability, and path smoothness.

## Authors

- **Saksham Humagain** – Computer Science, Kathmandu University  
- **Prajwal Ghimire** – Computer Science, Kathmandu University  

## Project Objective & Scope

The primary goal is to implement and visualize the behavior of optimizers on a complex, 3D non-linear loss surface.

### Scope

- Define a non-linear loss function with multiple local minima.  
- Implement Gradient Descent and Adam optimizers in Python.  
- Animate the trajectories on a 3D surface using Matplotlib.  
- Visually compare convergence speed, path smoothness, and stability.  

## Mathematical Foundation

### The Loss Function

The project uses a custom function to simulate a challenging optimization environment:

### Optimizers

- **Gradient Descent (GD):** Updates parameters in the direction of the negative gradient:



- **Adam Optimizer:** Combines momentum and adaptive learning rates for faster, smoother convergence:



## Implementation Details

The project is implemented using Python, leveraging **NumPy** for gradients and **Matplotlib** for 3D animation.

- **Gradients:** Computed analytically for the loss function.  
- **Visual Indicators:**  
  - Dots: Represent the current positions of the optimizers.  
  - Dashed Lines: Represent the path history on the surface.  
  - Colors: Red is used for GD and Blue is used for Adam.  

## Results & Observations

- Adam converges faster than GD, particularly on non-linear surfaces.  
- GD trajectories may oscillate in shallow regions before reaching the minimum.  
- Adam’s momentum and adaptive rates produce smoother trajectories.  
- Visualization provides insight into behavior that numerical results alone cannot offer.  

## References

- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.  
- Kingma, D. P., & Ba, J. (2015). *Adam: A Method for Stochastic Optimization*. ICLR.  
- Matplotlib and NumPy Documentation
