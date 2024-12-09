.. _theory:

Theory of Dual Numbers
======================

Overview
--------
Dual numbers are an algebraic extension of the real numbers that introduce a
new element :math:`\varepsilon` (often denoted "e" in code) with the defining
property:

.. math::

   \varepsilon^2 = 0.

In other words, while :math:`\varepsilon` is not zero itself, its square—and
thus any higher power—is identically zero. A dual number therefore takes the form

.. math::

   z = a + b\varepsilon,

where :math:`a` and :math:`b` are real numbers.

This construction forms a commutative ring, often interpreted as the real numbers
augmented with an infinitesimal direction. Dual numbers are a fundamental tool
in automatic differentiation, where they represent both a function value and its
first-order derivative simultaneously.

Algebraic Structure
-------------------
**Addition:** Given two dual numbers :math:`z_1 = a + b\varepsilon` and
:math:`z_2 = c + d\varepsilon`, their sum is defined component-wise:

.. math::

   z_1 + z_2 = (a + c) + (b + d)\varepsilon.

**Multiplication:** Multiplying two dual numbers uses distributivity and the rule
:math:`\varepsilon^2 = 0`:

.. math::

   z_1 \cdot z_2 = (a + b\varepsilon)(c + d\varepsilon)
   = ac + (ad + bc)\varepsilon + bd\,\varepsilon^2.

Since :math:`\varepsilon^2 = 0`, the last term vanishes, leaving

.. math::

   z_1 \cdot z_2 = ac + (ad + bc)\varepsilon.

**Conjugation:** There is a natural notion of a "conjugate" or "augmentation map"
that projects a dual number onto its real part:

.. math::

   \overline{z} = a.

While not strictly necessary for basic arithmetic, this projection often aids in
conceptualizing the structure as "real part plus infinitesimal part."

Relationship to Automatic Differentiation
-----------------------------------------
The primary motivation for dual numbers in many computational contexts is their
ability to encode both a function’s value and its first-order derivative at a
point. If we view :math:`\varepsilon` as an infinitesimal increment, then for a
differentiable function :math:`f: \mathbb{R} \to \mathbb{R}`, we can write:

.. math::

   f(a + \varepsilon) = f(a) + f'(a)\varepsilon.

In essence, evaluating the function at a dual number "simulates" the function’s
value and derivative. By propagating dual numbers through arithmetic operations,
one obtains the derivative via a single function evaluation, a method known as
**forward-mode automatic differentiation**.

Properties and Extensions
-------------------------
1. **Associativity and Commutativity:**  
   Dual numbers form a ring under the usual definitions of addition and 
   multiplication. Associativity and commutativity hold for both operations, and 
   distributivity of multiplication over addition is satisfied.

2. **Subalgebra of Formal Power Series:**  
   The dual numbers can be viewed as a quotient of the ring of polynomials
   :math:`\mathbb{R}[x]` by the ideal :math:`(x^2)`. Equivalently, they are like 
   formal power series truncated at the :math:`x^2` term.

3. **Invertibility:**  
   A dual number :math:`a + b\varepsilon` is invertible if and only if :math:`a \neq 0`.
   In such a case, its inverse can be computed as:

   .. math::

      \frac{1}{a + b\varepsilon} = \frac{1}{a} - \frac{b}{a^2}\varepsilon,

   since higher-order terms vanish.

4. **No Ordering:**  
   Unlike real numbers, the set of dual numbers does not inherit an order relation 
   that aligns with the arithmetic. While we can still compare the real parts 
   separately, the infinitesimal portion does not lend itself to a total order.

Conclusion
----------
The dual numbers :math:`a + b\varepsilon` elegantly capture both a value and a 
directional derivative, making them an invaluable tool in automatic 
differentiation. Their algebraic simplicity—stemming from the nilpotent element 
:math:`\varepsilon`—renders them a natural extension to the real numbers 
for tasks requiring first-order sensitivity analysis.
