import numpy as np
import plotly.graph_objects as go

from .models import Graph, NearestTarget


def plot_graph(g: Graph, c: np.ndarray | None = None,
               fig: go.Figure | None = None) -> go.Figure:

    nan = np.nan * np.ones((g.num_edges, 1))
    x = np.hstack([g.vertices[:, 0][g.edges], nan])
    y = np.hstack([g.vertices[:, 1][g.edges], nan])

    fig = go.Figure() if fig is None else fig

    fig.add_trace(go.Scattergl(
        x=np.ravel(x),
        y=np.ravel(y),
        mode='markers+lines',
        line={'width': 1},
        marker={'color': 'black', 'size': 3},
    ))
    fig.update_yaxes(scaleanchor="x", scaleratio=1)

    return fig


def plot_distances(g: Graph, sources: np.ndarray, targets: np.ndarray,
                   source_target: NearestTarget, fig: go.Figure | None = None
                   ) -> go.Figure:
    cmap = np.arange(targets.shape[0])
    np.random.shuffle(cmap)

    # plt.scatter(g.vertices[:, 0], g.vertices[:, 1], s=3, c='black')
    # plt.scatter(sources[:, 0], sources[:, 1], c=cmap[source_target.index])
    # plt.scatter(targets[:, 0], targets[:, 1], s=10, c=cmap,
    #             linewidths=2, edgecolors='black')
    # plt.show()

    fig = go.Figure() if fig is None else fig

    fig.add_trace(go.Scattergl(
        x=g.vertices[:, 0],
        y=g.vertices[:, 1],
        mode='markers',
        marker={'color': 'black', 'size': 3},
    ))
    fig.add_trace(go.Scattergl(
        x=sources[:, 0],
        y=sources[:, 1],
        mode='markers',
        marker={'color': cmap[source_target.index]},
    ))
    fig.add_trace(go.Scattergl(
        x=targets[:, 0],
        y=targets[:, 1],
        mode='markers',
        marker={'color': cmap, 'size': 10,
                'line': {'width': 2, 'color': 'black'}},
    ))
    fig.update_yaxes(scaleanchor="x", scaleratio=1)

    return fig
