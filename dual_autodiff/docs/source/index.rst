Welcome to dual_autodiff's Documentation
=====================================

``dual_autodiff`` is a Python package that implements automatic differentiation using dual numbers. This implementation is particularly useful for computing derivatives of functions accurately, without the errors associated with numerical differentiation or the complexity of symbolic differentiation.

Why Dual Numbers?
----------------

Imagine you want to find the derivative of a function. You could:

1. Calculate it by hand (tedious and error-prone)
2. Use numerical approximations (subject to numerical errors)
3. Use symbolic differentiation (computationally expensive)
4. Use dual numbers (fast and accurate!)

Dual numbers extend real numbers similar to how complex numbers do, but instead of i² = -1, we have ε² = 0. This special property makes them perfect for computing derivatives automatically.

Quick Start
----------

Installation
~~~~~~~~~~~

Install the package using pip:

.. code-block:: bash

   pip install -e .

Basic Usage
~~~~~~~~~~

Here's a simple example:

.. code-block:: python

   import dual_autodiff as df

   # Define a function to differentiate
   def f(x):
       return x**2 + 2*x + 1

   # Calculate derivative at x = 2
   derivative = df.autodiff(f, 2.0)
   print(f"The derivative at x = 2 is {derivative}")  # Should print 6.0

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   theory
   tutorial
   api
   examples
   advanced

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`