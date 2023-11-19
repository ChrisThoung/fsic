# -*- coding: utf-8 -*-
"""
new_keynesian_3_equation_model
==============================
fsic implementation of 'A New Keynesian 3-equation model' from Prante and
Kohler (2023).

After model setup, the script has three steps:

1. Solve the model for equilibrium
2. Run the scenarios
    a. Increase in autonomous demand
    b. Increase in target inflation
    c. Increase in equilibrium output
3. Create a chart of output and a graph representation of the model

References:

    Prante, F., Kohler, K. (2023)
    'DIY macroeconomic model simulation: A New Keynesian 3-equation model'
    https://macrosimulation.org/a_new_keynesian_3_equation_model
"""

from typing import List, NamedTuple

import matplotlib.pyplot as plt
import networkx as nx

import fsic


SCRIPT = '''
# 1. IS curve
y = {A} - {a_1} * r[-1]

# 2. Phillips Curve
pi = pi[-1] + {a_2} * (y - {y_e})

# 3. Equilibrium interest rate
r_s = ({A} - {y_e}) / {a_1}

# 4. Monetary policy response
r = r_s + {a_3} * (pi - {pi_T})
'''

SYMBOLS = fsic.parse_model(SCRIPT)
NewKeynesian3EquationModel = fsic.build_model(SYMBOLS)


if __name__ == '__main__':
    # 1. Solve the model for equilbrium ---------------------------------------
    solution = NewKeynesian3EquationModel(range(50),
                                          a_1=0.3, a_2=0.7,
                                          A=10, pi_T=2, y_e=5)

    # Add a separate variable, `b`, and calculate the adjustment parameter,
    # `a_3`
    solution.add_variable('b', 1)
    solution.a_3 = solution.eval('(a_1 * (1 / (b * a_2) + a_2)) ** -1')

    solution.solve()

    # Copy the equilibrium results to a new object
    equilibrium = NewKeynesian3EquationModel(
        range(1, 50 + 1),
        **dict(zip(solution.names, solution.values[:, -1])))
    equilibrium.solve()


    # 2. Run the scenarios ----------------------------------------------------

    # a. Increase in autonomous demand
    higher_aggregate_demand = equilibrium.copy()
    higher_aggregate_demand['A', 5:] = 12
    higher_aggregate_demand.solve()

    # b. Increase in target inflation
    higher_inflation_target = equilibrium.copy()
    higher_inflation_target['pi_T', 5:] = 2.5
    higher_inflation_target.solve()

    # c. Increase in equilibrium output
    higher_equilibrium_output = equilibrium.copy()
    higher_equilibrium_output['y_e', 5:] = 7
    higher_equilibrium_output.solve()


    # 3. Create a chart of output and a graph representation of the model -----
    _, (chart, graph) = plt.subplots(1, 2, figsize=(14, 6.5))
    plt.suptitle(r'New Keynesian three-equation model')

    # Chart of output under different scenarios
    chart.plot(equilibrium.span, equilibrium.y,
               label='Initial equilibrium',
               linewidth=0.5, linestyle='--', color='k')

    chart.plot(higher_aggregate_demand.span, higher_aggregate_demand.y,
               label='Increase in autonomous demand', color='#33C3F0')
    chart.plot(higher_inflation_target.span, higher_inflation_target.y,
               label='Higher inflation target', color='#FF4F2E')
    chart.plot(higher_equilibrium_output.span, higher_equilibrium_output.y,
               label='Increase in equilibrium output', color='#4563F2')

    chart.set_xlim(1, 15)
    chart.set_xlabel('Time')

    chart.set_ylim(3, 8)
    chart.set_ylabel('Real output (y)')

    chart.legend(loc='lower right')
    chart.set_title('Output under different scenarios')

    # Graph representation of the model
    G = fsic.tools.symbols_to_graph(SYMBOLS)

    # Remove the `a_` parameters
    G.remove_nodes_from([x for x in G.nodes if x.startswith('a_')])

    # Add a further edge to show how the interest rate affects output with a
    # lag
    G.add_edge('r[t]', 'r[t-1]')

    # Settings to display the variables (nodes) of the model (graph)
    class NodeSetting(NamedTuple):
        position: List[float]
        label: List[str]
        colour: List[str]

    node_settings = {
        'y[t]':    NodeSetting([2.00, 3.00], r'$y_t$',       '#FF992E'),
        'pi[t]':   NodeSetting([3.00, 3.00], r'$\pi_t$',     '#FF992E'),
        'r[t]':    NodeSetting([3.00, 2.00], r'$r_t$',       '#FF992E'),
        'r[t-1]':  NodeSetting([2.50, 2.50], r'$r_{t-1}$',   '#33C3F0'),

        'pi[t-1]': NodeSetting([3.25, 3.25], r'$\pi_{t-1}$', '#33C3F0'),
        'pi_T[t]': NodeSetting([3.25, 1.75], r'$\pi^T_{t}$', '#4563F2'),

        'r_s[t]':  NodeSetting([3.50, 2.50], r'$r_{s,t}$',   '#77C3AF'),
        'y_e[t]':  NodeSetting([3.25, 2.75], r'$y^e_{t}$',   '#4563F2'),

        'A[t]':    NodeSetting([3.50, 3.50], r'$A_t$',       '#4563F2'),
    }

    nx.draw_networkx(G, ax=graph,
                     pos={k: v.position for k, v in node_settings.items()},
                     node_color=[node_settings[n].colour for n in G.nodes],
                     labels={k: v.label for k, v in node_settings.items()})

    graph.set_title('Model structure')

    plt.savefig('new_keynesian_3_equation_model.png')
