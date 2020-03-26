## Parameter $\lambda$ in regularization (neural network, logistic regression)
Large $\lambda$ in regularization terms penalizes the $\theta$ so that changes in $\theta$ are less impactfull meaning that algorithm is more prone to underfitting or more simplified(or at least thats how I understand it)\
Reversly, small $\lambda$ penalizes the $\theta$ by just a small ammount and it's more prone to overfitting.

## Support Vector Machines

Cost function of SVM looks a bit like this \\_ or _/ - very similar to logistic regression but it has a drop off at the end.

$J_\theta = \frac{1}{m}\sum_{i=1}^m -y^{(i)} \log\Big(\dfrac{1}{1 + e^{-\theta^Tx^{(i)}}}\Big) - (1 - y^{(i)})\log\Big(1 - \dfrac{1}{1 + e^{-\theta^Tx^{(i)}}}\Big)$

$\text{cost}_0(z) = \max(0, k(1+z))\newline\text{cost}_1(z) = \max(0, k(1-z))$

To make svm we modify  first term of logistic regression so that when z = $\theta^Tx$ is > 1 it outputs 0. Furthermore for values less than 1 we use a straight decreasing line instead of the sigmoid curve.

similarily we modify the second term of the cost function so that when z = $\theta^Tx$ is less then -1 it outputs 0. we also modify it so that for values of z greater than -1 we use a straight decreasing line.

**SVM with regularization**<br>
$J(\theta) = C\sum_{i=1}^m y^{(i)} \ \text{cost}_1(\theta^Tx^{(i)}) + (1 - y^{(i)}) \ \text{cost}_0(\theta^Tx^{(i)}) + \dfrac{1}{2}\sum_{j=1}^n \Theta^2_j$

The hypothesis of SVM is not interpreted as probability of y being 0 or 1. Instead it outputs either 0 or 1.

$h_\theta(x) =\begin{cases}    1 & \text{if} \ \Theta^Tx \geq 0 \\    0 & \text{otherwise}\end{cases}$ 

**SVM is a large margin classifier**

if y = 1 we want $\theta^Tx$ >= 1 (not just >= 0)
if y = 0 we want $\theta^Tx$ <= -1 (not just < 0)

when we set constant C to a very large value (e.g. 100 000), our optimizing function will contrain theta such that our equation equals. If C is very large we must choose $\theta$ so that

$\sum_{i=1}^m y^{(i)}\text{cost}_1(\Theta^Tx) + (1 - y^{(i)})\text{cost}_0(\Theta^Tx) = 0$

this reduces cost funtion to:

$J(\theta) = C \cdot 0 + \dfrac{1}{2}\sum_{j=1}^n \Theta^2_j \newline= \dfrac{1}{2}\sum_{j=1}^n \Theta^2_j$

in svm the decision boundry has the special property that is as far away as possible from both the positive and negative examples. The distance of the decision boundry to the nearest example is called the margin.

Large maring is only achieved when C is large. 

Data is linearly spearable when a straight line can separate the positive and negative examples.

If we have outliers that we dont want to affect the decision boundry with, we can reduce C.

## KERNELS

Given x compute new feature depending on proximity to landmarks l1, l2, l3 ...<br />
To do this. we find the similarity of x and some landmark li.<br /> 
**Gaussian Kernel:**

$f_i = similarity(x, l^{(i)}) = \exp(-\dfrac{||x - l^{(i)}||^2}{2\sigma^2})$

$f_i = similarity(x, l^{(i)}) = \exp(-\dfrac{\sum^n_{j=1}(x_j-l_j^{(i)})^2}{2\sigma^2})$

properties:
if x very near li then fi = 1 (close to 1)
if x is far from li then fi = 0 (close to 0)

Each feature gives us the features in our hypothesis

$h_\theta(x) = \theta_1f_1+\theta_2f_2 +\theta_3f_3 + ...$

$\sigma^2$ - parameter of gaussian kernel, can be modified to increase or decrease drop off of our feature fi.

**How to choose landmarks**<br />
one way is to put them is exact same locations as all the training examples. This gives us m landmarks. One per example.<br />
vertical vector: 
$x^{(i)} \rightarrow \begin{bmatrix}f_1^{(i)} = similarity(x^{(i)}, l^{(1)}) \newline f_2^{(i)} = similarity(x^{(i)}, l^{(2)}) \newline\vdots \newline f_m^{(i)} = similarity(x^{(i)}, l^{(m)}) \newline\end{bmatrix}$

now to get theta we can use svm minimaztion but with fi substituted in for xi