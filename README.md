# Neural Differential Equation Investigations

This repository holds some preliminary investigations into Neural Differential Equations, particularly around Neural ODEs and basic engineering / implementation details.

The work followed here stems from Chen, R.; Rubanova, Y.; Bettencourt, J.; and Duvenaud, D. *Neural Ordinary Differential Equations*. 2019. [See arXiv preprint.](https://arxiv.org/pdf/1806.07366)

The file `annotatedExample.py` works from the [example given here](https://github.com/rtqichen/torchdiffeq/blob/master/examples/ode_demo.py), with comments to help label exactly what is done at each step, what parts are simulation, and what parts are modelling.

`exampleExtension.ipynb` takes the above example and investigates several parts of it to gain an understanding of what works and what doesn't when using these models. Some experiments include:
- Altering the network inputs (e.g. providing more than the minimum set of y augmentations, such as many exponents rather than only y**3)
- Changing the optimizer
- Changing the loss function