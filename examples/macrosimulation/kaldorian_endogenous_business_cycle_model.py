# -*- coding: utf-8 -*-
"""
kaldorian_endogenous_business_cycle_model
=========================================
fsic implementation of 'A Kaldorian endogenous business cycle model' from
Prante and Kohler (2023).

References:

    Prante, F., Kohler, K. (2023)
    'DIY macroeconomic model simulation: A Kaldorian endogenous business cycle model'
    https://macrosimulation.org/a_kaldor_model
"""

import string
from typing import List, NamedTuple

import matplotlib.pyplot as plt
import networkx as nx

import fsic


SCRIPT = """
# Output
Y = Y[-1] + {alpha} * (I[-1] - S[-1])

# Capital stock
K = (1 - {delta}) * K[-1] + I[-1]

# Saving
S = {sigma} * Y

# Investment
I = {sigma} * Y_E + {gamma} * (({sigma} * Y_E / {delta}) - K) + np.arctan(Y - Y_E)
"""

SYMBOLS = fsic.parse_model(SCRIPT)
KaldorianEndogenousBusinessCycleModel = fsic.build_model(SYMBOLS)


if __name__ == '__main__':
    # 1. Solve the model ------------------------------------------------------
    # NB Prante and Kohler (2023) initialise the endogenous variables to 1,
    #    rather than 0. The results would otherwise be identical.
    model = KaldorianEndogenousBusinessCycleModel(
        range(200), alpha=1.2, delta=0.2, sigma=0.4, Y_E=10, gamma=0.6
    )

    model.solve()

    # 2. Create charts and a graph representation of the model ----------------

    # Reindex the model to match the span of the plots in Prante and Kohler
    # (2023)
    model_plots = model.reindex(range(10, 100 + 1))

    _, axes = plt.subplots(2, 2, figsize=(14, 13))
    plt.suptitle(r'Kaldorian endogenous business cycle model')

    # Output and capital stock as the left-hand side charts
    axes[0, 0].set_title('Output')
    axes[0, 0].plot(model_plots.span, model_plots.Y, label='_', color='#FF992E')
    axes[0, 0].set_xlabel('Time')
    axes[0, 0].set_ylabel('Output (Y)')

    axes[1, 0].set_title('Capital stock')
    axes[1, 0].plot(model_plots.span, model_plots.K, label='_', color='#4563F2')
    axes[1, 0].set_xlabel('Time')
    axes[1, 0].set_ylabel('Capital stock (K)')

    # Saving and investment in a single chart
    axes[0, 1].set_title('Saving and investment')
    axes[0, 1].set_xlabel('Time')

    # Saving and mean
    axes[0, 1].plot(
        model_plots.span,
        model_plots.S,
        label='Saving (S)',
        color='#33C3F0'
    )  # fmt: skip
    axes[0, 1].plot(
        model_plots.span,
        [model_plots.S.mean()] * len(model_plots.span),
        label='_',
        linewidth=0.5,
        linestyle='--',
        color='#33C3F0',
    )

    # Investment and mean
    axes[0, 1].plot(
        model_plots.span,
        model_plots.I,
        label='Investment (I)',
        color='#FF4F2E'
    )  # fmt: skip
    axes[0, 1].plot(
        model_plots.span,
        [model_plots.I.mean()] * len(model_plots.span),
        label='_',
        linewidth=0.5,
        linestyle='--',
        color='#FF4F2E',
    )
    axes[0, 1].legend(loc='lower right')

    # Graph representation of the model
    G = fsic.tools.symbols_to_graph(SYMBOLS)

    # Remove parameters
    G.remove_nodes_from([x for x in G.nodes if str(x)[0] in string.ascii_lowercase])

    # Settings to display the variables (nodes) of the model (graph)
    class NodeSetting(NamedTuple):
        position: List[float]
        label: List[str]
        colour: List[str]

    node_settings = {
        'Y[t]':    NodeSetting([3.00, 3.00], r'$Y_t$',       '#FF992E'),
        'Y[t-1]':  NodeSetting([2.75, 3.00], r'$Y_{t-1}$',   '#FF992E'),
        'Y_E[t]':  NodeSetting([3.00, 3.25], r'$Y^E_t$',     '#77C3AF'),
        'I[t]':    NodeSetting([3.25, 3.25], r'$I_t$',       '#FF4F2E'),
        'I[t-1]':  NodeSetting([2.75, 3.25], r'$I_{t-1}$',   '#FF4F2E'),
        'S[t]':    NodeSetting([3.25, 2.75], r'$S_t$',       '#33C3F0'),
        'S[t-1]':  NodeSetting([2.75, 2.75], r'$S_{t-1}$',   '#33C3F0'),
        'K[t]':    NodeSetting([3.00, 3.50], r'$K_t$',       '#4563F2'),
        'K[t-1]':  NodeSetting([2.75, 3.50], r'$K_{t-1}$',   '#4563F2'),
    }  # fmt: skip

    nx.draw_networkx(G, ax=axes[1, 1],
                     pos={k: v.position for k, v in node_settings.items()},
                     node_color=[node_settings[n].colour for n in G.nodes],
                     labels={k: v.label for k, v in node_settings.items()})  # fmt: skip

    axes[1, 1].set_title('Model structure')

    plt.savefig('kaldorian_endogenous_business_cycle_model.png')
