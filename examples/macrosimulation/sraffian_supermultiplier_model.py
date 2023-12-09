# -*- coding: utf-8 -*-
"""
sraffian_supermultiplier_model
==============================
fsic implementation of 'A Sraffian supermultiplier model' from Prante and
Kohler (2023).

After model setup, the script has three steps:

1. Set the model to be at equilbrium
2. Run the scenarios
    a. Increase in autonomous demand growth
    b. Increase in profit share
    c. Increase in normal capacity utilisation
3. Create charts and a graph representation of the model

References:

    Prante, F., Kohler, K. (2023)
    'DIY macroeconomic model simulation: A Sraffian supermultiplier model'
    https://macrosimulation.org/a_sraffian_supermultiplier_model
"""

from typing import List, NamedTuple

import matplotlib.pyplot as plt
import networkx as nx

import fsic


SCRIPT = '''
# 1. Profit rate
r = pi * u

# 2. Saving rate
s = -z + {s_r} * r

# 3. Consumption rate
c = u - s

# 4. Investment rate (growth of the capital stock)
g = g_0 + {g_1} * (u - {u_n})

# 5. Capacity utilisation
u = c + g

# 6. Expected growth of the capital stock
g_0 = g_0[-1] + ({mu} * (g[-1] - g_0[-1])) * {d}

# 7. Autonomous demand rate
z = z[-1] + z[-1] * (g_z[-1] - g[-1]) * {d}
'''

SYMBOLS = fsic.parse_model(SCRIPT)
SraffianSupermultiplierModel = fsic.build_model(SYMBOLS)


if __name__ == '__main__':
    # 1. Set the model to be at equilbrium ------------------------------------
    equilibrium = SraffianSupermultiplierModel(range(1000),
                                            g_1=0.2, s_r=0.8, mu=0.08, d=0.1,
                                            pi=0.35, g_z=0.02, u_n=0.75, strict=True)

    equilibrium.s = equilibrium.g = equilibrium.g_0 = equilibrium.g_z
    equilibrium.u = equilibrium.u_n
    equilibrium.c = equilibrium.eval('u_n - s')
    equilibrium.r = equilibrium.eval('pi * u_n')
    equilibrium.z = equilibrium.eval('s_r * r - g_z')

    equilibrium.solve()


    # 2. Run the scenarios ----------------------------------------------------

    # a. Increase in autonomous demand growth
    higher_autonomous_demand = equilibrium.copy()
    higher_autonomous_demand['g_z', 50:] = 0.03
    higher_autonomous_demand.solve(max_iter=500)

    # b. Increase in profit share
    higher_profit_share = equilibrium.copy()
    higher_profit_share['pi', 50:] = 0.4
    higher_profit_share.solve(max_iter=500)

    # c. Increase in normal capacity utilisation
    higher_capacity_utilisation = equilibrium.copy()
    higher_capacity_utilisation['u_n', 50:] = 0.8
    higher_capacity_utilisation.solve(max_iter=500)


    # 3. Create charts and a graph representation of the model ----------------
    _, axes = plt.subplots(2, 2, figsize=(14, 13))
    plt.suptitle('Sraffian supermultiplier model')

    # Chart results under different scenarios
    def plot(variable, title, ylabel, ylim, axis):
        """Plot `variable` on `axis`."""
        axis.plot(equilibrium.span, equilibrium[variable],
                  label='Initial equilibrium',
                  linewidth=0.5, linestyle='--', color='k')

        axis.plot(higher_autonomous_demand.span, higher_autonomous_demand[variable],
                  label='Increase in autonomous demand growth', color='#33C3F0')
        axis.plot(higher_profit_share.span, higher_profit_share[variable],
                  label='Higher profit share', color='#FF4F2E')
        axis.plot(higher_capacity_utilisation.span, higher_capacity_utilisation[variable],
                  label='Increase in capacity utilisation', color='#4563F2')

        axis.set_xlim(0, 700)
        axis.set_xlabel('Time')

        axis.set_ylim(*ylim)
        axis.set_ylabel(ylabel)

        axis.set_title(title)

    plot('u', 'Capacity utilisation', 'Capacity utilisation (u)', (0.2, 1.0), axes[0, 0])
    plot('g', 'Capital stock growth', 'Capital stock growth (g)', (-0.05, 0.05), axes[0, 1])
    plot('z', 'Autonomous demand growth', 'Autonomous demand growth (z)', (0.15, 0.25), axes[1, 0])

    axes[0, 0].legend(loc='lower right')

    # Graph representation of the model
    G = fsic.tools.symbols_to_graph(SYMBOLS)

    # Remove `d`
    G.remove_nodes_from([x for x in G.nodes if x == 'd[t]'])

    # Settings to display the variables (nodes) of the model (graph)
    class NodeSetting(NamedTuple):
        position: List[float]
        label: List[str]
        colour: List[str]

    node_settings = {
        'c[t]':     NodeSetting([2.25, 2.25], r'$c_t$',       '#FF992E'),

        'g[t]':     NodeSetting([1.50, 2.50], r'$g_t$',       '#FF992E'),
        'g[t-1]':   NodeSetting([1.00, 1.50], r'$g_{t-1}$',   '#4563F2'),

        'g_0[t]':   NodeSetting([1.00, 2.00], r'$g_{0,t}$',   '#FF992E'),
        'g_0[t-1]': NodeSetting([0.75, 1.79], r'$g_{0,t-1}$', '#777777'),

        'g_1[t]':   NodeSetting([1.79, 2.75], r'$g_{1,t}$',   '#777777'),
        'g_z[t-1]': NodeSetting([1.29, 1.25], r'$g_{z,t-1}$', '#4563F2'),

        'mu[t]':    NodeSetting([0.75, 2.21], r'$\mu_t$',     '#777777'),
        'pi[t]':    NodeSetting([1.50, 2.00], r'$\pi_t$',     '#777777'),
        'r[t]':     NodeSetting([1.75, 2.25], r'$r_t$',       '#FF992E'),

        's[t]':     NodeSetting([2.00, 2.00], r'$s_t$',       '#FF992E'),
        's_r[t]':   NodeSetting([2.25, 1.75], r'$s_{r,t}$',   '#4563F2'),

        'u[t]':     NodeSetting([2.00, 2.50], r'$u_t$',       '#FF992E'),
        'u_n[t]':   NodeSetting([1.21, 2.75], r'$u_{n,t}$',   '#4563F2'),

        'z[t]':     NodeSetting([1.50, 1.50], r'$z_t$',       '#FF992E'),
        'z[t-1]':   NodeSetting([1.71, 1.25], r'$z_{t-1}$',   '#33C3F0'),
    }

    nx.draw_networkx(G, ax=axes[1, 1],
                     pos={k: v.position for k, v in node_settings.items()},
                     node_color=[node_settings[n].colour for n in G.nodes],
                     labels={k: v.label for k, v in node_settings.items()})

    axes[1, 1].set_title('Model structure')

    plt.savefig('sraffian_supermultiplier_model.png')
