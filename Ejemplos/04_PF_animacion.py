from dash import Dash, dcc, html, Input, Output, State, ctx
import plotly.graph_objects as go
import numpy as np
import random

# Inicializar la app Dash
app = Dash(__name__)

# Variables globales
num_particles = 50
particle_size = 6
target_size = 12
measurement_size = 40

# Inicialización de partículas, objetivo y medición
particles = np.random.rand(num_particles, 2) * 600  # Posiciones iniciales
weights = np.ones(num_particles) / num_particles
target = np.array([300.0, 200.0])
measurement = np.array([300.0, 200.0])

# Crear el layout de la aplicación
app.layout = html.Div([
    html.Div([
        html.Button("Start", id="start-button", n_clicks=0),
        html.Button("Reset", id="reset-button", n_clicks=0),
        html.Label("Speed:"),
        dcc.Slider(
            id="speed-slider",
            min=1,
            max=100,
            step=1,
            value=50,
            tooltip={"placement": "bottom", "always_visible": True},
        ),
        html.Button("Apply Speed", id="apply-speed-button", n_clicks=0),
        html.Div(id="speed-display", style={"margin-top": "10px"}),
    ], style={"margin-bottom": "20px"}),
    dcc.Graph(id="particle-filter-plot")
])


# Funciones para manejar partículas
def move_particles(speed):
    global particles
    noise = (np.random.rand(num_particles, 2) - 0.5) * (speed / 10)  # Escalar por velocidad
    particles += noise
    particles[:, 0] = np.clip(particles[:, 0], 0, 600)
    particles[:, 1] = np.clip(particles[:, 1], 0, 400)


def update_target_and_measurement():
    global target, measurement
    target += (np.random.rand(2) - 0.5) * 5
    target = np.clip(target, [0, 0], [600, 400])
    measurement = target + (np.random.rand(2) - 0.5) * 20
    measurement = np.clip(measurement, [0, 0], [600, 400])


def update_weights():
    global weights
    distances = np.linalg.norm(particles - measurement, axis=1)
    weights = 1 / (1 + distances)
    weights /= weights.sum()


def resample_particles():
    global particles
    indices = np.random.choice(range(num_particles), size=num_particles, p=weights)
    particles = particles[indices] + (np.random.rand(num_particles, 2) - 0.5) * 10


# Función para crear el gráfico inicial
def create_plot():
    fig = go.Figure()

    # Partículas
    fig.add_trace(go.Scatter(
        x=particles[:, 0],
        y=particles[:, 1],
        mode='markers',
        marker=dict(size=particle_size, color='blue', opacity=0.5),
        name='Particles'
    ))

    # Target
    fig.add_trace(go.Scatter(
        x=[target[0]],
        y=[target[1]],
        mode='markers',
        marker=dict(size=target_size, color='red'),
        name='Target'
    ))

    # Measurement
    fig.add_trace(go.Scatter(
        x=[measurement[0]],
        y=[measurement[1]],
        mode='markers',
        marker=dict(size=measurement_size, color='gray', opacity=0.3),
        name='Measurement'
    ))

    fig.update_layout(
        width=600,
        height=400,
        xaxis=dict(range=[0, 600]),
        yaxis=dict(range=[0, 400]),
        showlegend=True,
        title='Particle Filter Simulation'
    )
    return fig


# Callback para manejar los botones y sliders
@app.callback(
    [Output("particle-filter-plot", "figure"),
     Output("speed-display", "children")],
    [Input("start-button", "n_clicks"),
     Input("reset-button", "n_clicks"),
     Input("apply-speed-button", "n_clicks")],
    [State("speed-slider", "value")],
    prevent_initial_call=True
)
def update_simulation(start_clicks, reset_clicks, apply_speed_clicks, speed):
    global particles, target, measurement, weights

    # Obtener el elemento que activó el callback
    triggered_id = ctx.triggered_id

    if triggered_id == "reset-button":
        particles = np.random.rand(num_particles, 2) * 600  # Resetear partículas
        target = np.array([300.0, 200.0])
        measurement = np.array([300.0, 200.0])
        weights = np.ones(num_particles) / num_particles

    elif triggered_id == "start-button":
        move_particles(speed)
        update_target_and_measurement()
        update_weights()
        if random.random() < 0.1:  # Resample cada cierto tiempo
            resample_particles()

    elif triggered_id == "apply-speed-button":
        move_particles(speed)
        update_target_and_measurement()
        update_weights()
        resample_particles()

    return create_plot(), f"Speed: {speed}%"


# Ejecutar la app
if __name__ == "__main__":
    app.run_server(debug=True)
