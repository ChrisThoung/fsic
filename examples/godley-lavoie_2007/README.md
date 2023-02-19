# Godley and Lavoie (2007) *Monetary economics*

fsic implementations of stock-flow consistent models from Godley and Lavoie
(2007).

There is one script per model, with naming convention as follows:

	X_yyy.py

where `X` is the chapter number and `yyy` is the model name e.g. '3_sim.py'.


## Usage

To simulate each model:

1. Run the model script with *this folder* as the current working directory.

See the docstrings of the individual scripts for details of the outputs.


## Contents

Implemented models have an accompanying link in the table below. Other models
are in *italics*.

| Chapter                                                           | Model                                                          |
| ----------------------------------------------------------------- | -------------------------------------------------------------- |
|  1. Introduction                                                  | -                                                              |
|  2. Balance sheets, transaction matrices and the monetary circuit | -                                                              |
|  3. The simplest model with government money                      | [SIM](3_sim.py), *SIMEX*, *SIMEXF*                             |
|  4. Government money with portfolio choide                        | [PC](4_pc.py), *PCEX*, *PCEX1*, *PCEX2*, *PCNEO*               |
|  5. Long-term bonds, capital gains and liquidity preference       | [LP](5_lp1.py), *LP2*, *LP3*, *LPNEO*                          |
|  6. Introducing the open economy                                  | [REG](6_reg.py), [OPEN](6_open.py), *OPENG*, *OPENM*, *OPENM3* |
|  7. A simple model with private bank money                        | [BMW](7_bmw.py), *BMWK*                                        |
|  8. Time, inventories, profits and pricing                        | -                                                              |
|  9. A model with private bank money, inventories and inflation    | [DIS](9_dis.py), *DISINF*                                      |
| 10. A model with both inside and outside money                    | *INSOUT*                                                       |
| 11. A growth model prototype                                      | *GROWTH*                                                       |
| 12. A more advanced open economy mode                             | *OPENFIX*, *OPENFIXR*, *OPENFIXG*, *OPENFLEX*                  |
| 13. General conclusion                                            | -                                                              |


## References

Godley, W., Lavoie, M. (2007),
*Monetary economics: an integrated approach to
credit, money, income, production and wealth*,
Palgrave Macmillan
