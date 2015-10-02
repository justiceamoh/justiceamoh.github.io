---
title:  "A Coarse Cough Detection"
date:   2015-01-04 10:18:00
description: A First-Pass Cough Detection Block
published: true
---

## Introduction
This page is to give a cursory overview of automatic differentiation. Most of the content on here was lifted from the following resources:
- [Colah's blog on backpropagation][1]
- [Wikipedia page on AD][2]

## Computing Derivatives
- Analytical Method
- Divided Difference Approximation (Numerical Differentiation)
- Symbolic Method
- **Automatic Differentiation**

## Automatic Differentiation (AD)
AD is a set of techniques to numerically evaluate derivative:
- express functions as sequence of elementary arithmetic operations (+,-,×)
- apply **chain rule** repeatedly to these operations

![formula](http://latex.codecogs.com/gif.latex?%5Cfrac%7B%5Cpartial%20y%7D%7B%5Cpartial%20x%7D%3D%5Cfrac%7B%5Cpartial%20y%7D%7B%5Cpartial%20w%7D%5Ctimes%20%5Cfrac%7B%5Cpartial%20w%7D%7B%5Cpartial%20x%7D "\\frac{\\partial y}{\\partial x}=\\frac{\\partial y}{\\partial w}\\times \\frac{\\partial w}{\\partial x}")

Other names/variants: 
- Algorithmic/Computational Differentiation
- Reverse-Mode Differentiation (Backpropagation)

<!-- Pros:
- Chain rule: Decomposition of differentials 
- Fast at computing partial derivatives of a function wrt many inputs. 
 -->

## Computational Graphs
Math expressions can be thought of as **computational graphs**. 

Consider `e=(a+b)∗(b+1)`:
- 3 operations: 2 `+` and 1 `×`
- Intermediary variables: `c` & `d`
    + `c = a + b`
    + `d = b + 1`
    + `e = c * d`

<en-media type="image/png" hash="cb4c9c499a76ffbc8c44f27faa43316b"/>

- Functional programming, think of nodes as functions
- To evaluate expression:
    + Set input variables to values
    + Compute nodes up through graph 
- Example: set `a=2` and `b=1`, expect `e=6`

<en-media type="image/png" hash="474982020654627f39142ee4193611dc"/>


## Derivatives on Graphs
The partial derivatives are the edges of the graph:
- how is `c` affected by `a`?
- what is the partial derivative of `c` wrt to `a`? 

To evaluate the partial derivatives:
- Sum rule
- Product Rule

<en-media type="image/png" hash="28b669fd7cc592a4046a87d0ed6daa45"/>

Here is the computational graph with the derivatives label on each edge:

<en-media type="image/png" hash="2685be4fd083d82d5d364fd0d056858e"/>

To evaluate derivatives, just multiply values of edges connecting nodes. 
![formula](http://latex.codecogs.com/gif.latex?%5Cfrac%7B%5Cpartial%20e%7D%7B%5Cpartial%20a%7D%20%3D%20%5Cfrac%7B%5Cpartial%20c%7D%7B%5Cpartial%20a%7D%20%5Ccdot%20%5Cfrac%7B%5Cpartial%20e%7D%7B%5Cpartial%20c%7D%20%3D%201%5Cast%202%3D2 "\\frac{\\partial e}{\\partial a} = \\frac{\\partial c}{\\partial a} \\cdot \\frac{\\partial e}{\\partial c} = 1\\ast 2=2")

For indirectly connected nodes, **sum over all possible paths** (*multivariate chain rule*). For example:

<en-media type="image/png" hash="31874606fe6ab2e64c497c6922635cc3"/>


## Factoring Paths
Summing over all paths can easily lead to combinatorial explosion. 
<en-media type="image/png" hash="8877dd1bd004654d949917b5bfda90df"/>

In the above, there are 3 × 3 = 9 paths to sum over for derivative of Z wrt X. Paths grow exponential with complex graphs.
<en-media type="image/png" hash="e1784a5f7263bb740bf8bc30c155d869"/>


Instead of summing over all, we can factor them:

<en-media type="image/png" hash="cf51353678a9bb66f8c7c12533b411b3"/>

This is what AD algorithm does. It **efficiently computes the sum by factoring the paths** by merging edges together at every node. In AD, **each edge is touched only once**! 

There are two modes of the AD algorithm:
+ Forward-Mode Accumulation (Forward Pass)
+ Reverse-Mode Accumulation (Backpropagation)


## Forward-Mode Accumulation
<en-media type="image/png" hash="309601e3198240cb38d7f6cd2941db7e"/>

- **Feedforward**: start at input of graph and work towards end.
- At each node, sum all the paths feeding in 
- Finds how one input affects every node

## Reverse-Mode Accumulation (Backpropagation)
<en-media type="image/png" hash="6a9bf789bb038665296d26d234b744d4"/>

- **Backprop**: Backward propagation of derivatives
- Start at graph output and move towards the beginning
- At each node, sum all paths originating  
- Finds how **every node** affects one output


## Computational Victories (4N ops)
Consider the example:

<en-media type="image/png" hash="2685be4fd083d82d5d364fd0d056858e"/>

Using the *forward-mode* from b up gives derivative of every node wrt b. 

<en-media type="image/png" hash="a5a76bd3ec2de78c1cebb567c13e9713"/>

What of the reverse-mode from e down? Gives us derivatives of e wrt to **every node**!

<en-media type="image/png" hash="2625875347c22318802ac0fdf2a8ad11"/>

We get both de/da and de/db.  

> Forward-mode differentiation gave us the derivative of our output with respect to a single input, but reverse-mode differentiation **gives us all of them**.



[1]: http://colah.github.io/posts/2015-08-Backprop/
[2]: https://en.wikipedia.org/wiki/Automatic_differentiation
