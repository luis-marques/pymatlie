## Installation
Either install via `pip` through
```
pip install pymatlie
```
or, for development, clone the repository and install in editable mode using
```
$ python -m venv venv
$ source venv/bin/activate
$ pip install -e .
```

## Usage

We use $g \in G$ for a group element, where $G$ is the Matrix Lie Group,
$\xi \in \mathbb R^m$ for a column vector in the Lie Algebra and 
$\xi^\wedge \in \mathfrak g$ for the matrix representation of said vector.

```python
from pymatlie.se2 import SE2

g = SE2.random() # Sample random group element

# Moving from the Group to the Lie Algebra
xi_hat = SE2.log(g)
xi     = SE2.vee(xi_hat)
# or alternatively
xi     = SE2.Log(g)

# Moving from the Lie Algebra to the Group
xi_hat = SE2.hat(xi) # Or SE3.wedge(xi)
g       = SE2.expm(xi_hat)
# or alternatively
g       = SE2.exp(xi)

# There are methods for generating common matrices
e       = SE2.get_identity()
Jl      = SE2.left_jacobian(xi)
Jr      = SE2.right_jacobian(xi)
Ad      = SE2.adjoint_matrix(g)
ad      = SE2.ad_operator(g)
```