# ARCHITECTURE

`fsic` takes an object-oriented approach to modelling, in which individual
objects represent complete model instances, handling:

1. Data management, to access and modify values
2. Solution and solution control, to solve the model
3. The representation and storage of the equations that make up the model (and,
   by [2], their solution)

The rationale for an object-oriented approach rests on the idea that [1] and
[2] are generic operations not unique to any individual model. Only [3] is
model-specific, provided appropriate information can be extracted to inform [1]
(the list of variables in the model) and [2] (the mechanisms by which the model
is solved).


## Structure

`fsic` defines two main model classes:

1. *Model*s, which represent individual and standalone models, e.g. for a
   single country or region, and derived from the `BaseModel` class.
2. *Linker*s, which can store multiple model instances, transferring results
   between them, e.g. to link countries by trade, and based on the `BaseLinker`
   class.

