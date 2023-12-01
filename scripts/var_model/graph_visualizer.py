import dash
from dash import dcc, html
import networkx as nx
import plotly.graph_objects as go
from dash.dependencies import Input, Output
from scripts.var_model.var_model import load_adj
# Create a Dash web application
app = dash.Dash(__name__)
# Create a sample graph using NetworkX
knn, _, net, G = load_adj(show=False)
# Get positions for the nodes (using spring_layout for simplicity)
pos = nx.spring_layout(G)

# Create nodes and edges for Plotly graph object
edge_trace = go.Scatter(
    x=[pos[k][0] for k in list(G.nodes())],
    y=[pos[k][1] for k in list(G.nodes())],
    line=dict(width=0.5, color='#888'),
    hoverinfo='none',
    mode='lines')

node_trace = go.Scatter(
    x=[pos[k][0] for k in G.nodes()],
    y=[pos[k][1] for k in G.nodes()],
    mode='markers',
    hoverinfo='text',
    marker=dict(
        showscale=True,
        colorscale='YlGnBu',
        size=10,
        colorbar=dict(
            thickness=15,
            title='Node Connections',
            xanchor='left',
            titleside='right'
        )
    )
)
node_trace.marker.color = [len(net[k]) for k in G.nodes()]
node_trace.text = [f'Node {k}<br># of connections: {len(net[k])}' for k in G.nodes()]

# Create layout for Plotly graph object
layout = go.Layout(
    showlegend=False,
    hovermode='closest',
    margin=dict(b=0, l=0, r=0, t=0),
    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
)
all_node_ids = list(G.nodes())

# Combine nodes and edges into a Plotly graph object
fig = go.Figure(data=[edge_trace, node_trace], layout=layout)

# Define layout of the Dash app
app.layout = html.Div([
    dcc.Graph(id="network-graph",figure=fig),
    dcc.Dropdown(
        id='node-dropdown',
        options=[{'label': str(node), 'value': node} for node in all_node_ids],
        multi=True,
        value=all_node_ids,  # Default: All nodes selected
        style={'width': '50%'}
    ),
])

@app.callback(
    Output('network-graph', 'figure'),
    [Input('node-dropdown', 'value')]
)
def update_graph(selected_nodes):
    # Create a subgraph containing only the selected nodes and their edges
    ls = []
    for x in selected_nodes:
        seconds = set()
        one_hop = knn[x]["1_hop"]["nodes"]
        for nd in one_hop:
            seconds.update(knn[nd]["1_hop"]["nodes"])
        ls.extend(list(seconds - set(one_hop)))
    H = G.subgraph(list(set(selected_nodes + ls)))
    # Create nodes and edges for Plotly graph object
    edge_trace = go.Scatter(
        x=[pos[k][0] for k in H.nodes()],
        y=[pos[k][1] for k in H.nodes()],
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')
    edge_trace.text = [f"Edge {str(net[k[0], k[1]])}" for k in H.edges()]

    node_trace = go.Scatter(
        x=[pos[k][0] for k in H.nodes()],
        y=[pos[k][1] for k in H.nodes()],
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            size=10,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            )
        )
    )

    node_trace.marker.color = [len(net[k]) for k in H.nodes()]
    node_trace.text = [f'Node {k}<br># of connections: {len(net[k])}' for k in H.nodes()]

    # Create layout for Plotly graph object
    layout = go.Layout(
        showlegend=False,
        hovermode='closest',
        margin=dict(b=0, l=0, r=0, t=0),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )

    # Combine nodes and edges into a Plotly graph object
    fig = go.Figure(data=[edge_trace, node_trace], layout=layout)
    return fig

# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True)
