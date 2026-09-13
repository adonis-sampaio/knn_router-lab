"""Plotly figures for the app."""
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


def utility_bars(utility_by_model: dict, selected: str) -> go.Figure:
    models = list(utility_by_model)
    vals = [utility_by_model[m] for m in models]
    colors = ["#00CC96" if m == selected else "#636EFA" for m in models]
    fig = go.Figure(go.Bar(x=models, y=vals, marker_color=colors))
    fig.update_layout(title="Predicted utility by model (higher = better)",
                      yaxis_title="û(x, m)", margin=dict(t=40, b=20))
    return fig


def scatter_2d(coords_2d, best_models, models, query_coord=None, neighbor_coords=None):
    """UMAP projection of the support set, colored by best model."""
    df = {"x": coords_2d[:, 0], "y": coords_2d[:, 1], "best_model": best_models}
    fig = px.scatter(df, x="x", y="y", color="best_model",
                     category_orders={"best_model": models},
                     opacity=0.5, title="Embedding space (UMAP)")
    if neighbor_coords is not None:
        fig.add_scatter(x=neighbor_coords[:, 0], y=neighbor_coords[:, 1],
                        mode="markers", marker=dict(size=10, symbol="x",
                        color="black"), name="kNN neighbors")
    if query_coord is not None:
        fig.add_scatter(x=[query_coord[0]], y=[query_coord[1]], mode="markers",
                        marker=dict(size=16, color="red", symbol="star"),
                        name="Your query")
    return fig


def sample_complexity_plot(df) -> go.Figure:
    """Line chart of mean utility vs. training fraction per router."""
    fig = px.line(df, x="fraction", y="mean_utility", color="router",
                  markers=True, title="Sample complexity (Theorem 1)")
    fig.update_xaxes(title="Fraction of support set used for training")
    fig.update_yaxes(title="Mean utility on held-out queries")
    return fig
