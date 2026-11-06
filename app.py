import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="ADCS-Sim | Satellite Attitude Lab", page_icon="🛰️", layout="wide")

st.title("🛰️ ADCS-Sim: Satellite Attitude-Control Simulator")
st.caption("Educational 3-axis simulation | Sensor modelling • Attitude estimation • PID control • Visualization")

st.info(
    "Demonstration model: this app uses simplified Euler-angle rigid-body dynamics and a "
    "Kalman-style attitude estimator. The magnetometer attitude reference is idealized for "
    "teaching; this is not a flight-qualified spacecraft simulator or a full quaternion MEKF."
)

with st.sidebar:
    st.header("Simulation settings")
    duration = st.slider("Simulation duration (seconds)", 10, 60, 30, 5)
    dt = 0.05
    st.caption("Time step: 0.05 s")
    st.subheader("Initial satellite orientation (degrees)")
    initial = np.array([
        st.slider("Roll", -90, 90, 35),
        st.slider("Pitch", -90, 90, -25),
        st.slider("Yaw", -180, 180, 60)
    ], dtype=float)
    st.subheader("Desired orientation (degrees)")
    target = np.array([
        st.slider("Target roll", -90, 90, 0),
        st.slider("Target pitch", -90, 90, 0),
        st.slider("Target yaw", -180, 180, 0)
    ], dtype=float)
    st.subheader("PID tuning")
    kp = st.slider("Proportional gain (Kp)", 0.1, 8.0, 2.5, 0.1)
    ki = st.slider("Integral gain (Ki)", 0.0, 2.0, 0.08, 0.01)
    kd = st.slider("Derivative gain (Kd)", 0.0, 5.0, 1.8, 0.1)
    st.subheader("Actuators")
    use_wheel = st.checkbox("Enable reaction wheel", True)
    use_mtorquer = st.checkbox("Enable magnetorquer", True)
    max_torque = st.slider("Reaction-wheel torque limit (N·m)", 0.01, 0.20, 0.08, 0.01)
    st.subheader("Sensor noise")
    sensor_noise = st.slider("Attitude-reference noise (degrees)", 0.1, 5.0, 1.0, 0.1)
    gyro_noise = st.slider("Gyroscope noise (deg/s)", 0.01, 1.0, 0.15, 0.01)
    run = st.button("▶ Run simulation", type="primary", use_container_width=True)

def wrap_deg(a):
    return (a + 180.0) % 360.0 - 180.0

def rotation_matrix(euler_deg):
    roll, pitch, yaw = np.deg2rad(euler_deg)
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rx = np.array([[1,0,0],[0,cr,-sr],[0,sr,cr]])
    ry = np.array([[cp,0,sp],[0,1,0],[-sp,0,cp]])
    rz = np.array([[cy,-sy,0],[sy,cy,0],[0,0,1]])
    return rz @ ry @ rx

def cube_trace(attitude, name, color, opacity=1.0):
    # A simple 1-unit cube, rotated to show satellite attitude.
    verts = np.array([
        [-1,-0.65,-0.18], [1,-0.65,-0.18], [1,0.65,-0.18], [-1,0.65,-0.18],
        [-1,-0.65,0.18], [1,-0.65,0.18], [1,0.65,0.18], [-1,0.65,0.18]
    ], dtype=float)
    R = rotation_matrix(attitude)
    v = verts @ R.T
    faces = np.array([
        [0,1,2],[0,2,3],[4,6,5],[4,7,6],
        [0,4,5],[0,5,1],[1,5,6],[1,6,2],
        [2,6,7],[2,7,3],[3,7,4],[3,4,0]
    ])
    return go.Mesh3d(
        x=v[:,0], y=v[:,1], z=v[:,2],
        i=faces[:,0], j=faces[:,1], k=faces[:,2],
        name=name, color=color, opacity=opacity, flatshading=True
    )

def simulate():
    n = int(duration / dt) + 1
    t = np.linspace(0, duration, n)
    inertia = np.array([0.025, 0.030, 0.020])  # kg m^2, illustrative CubeSat values
    angle = initial.copy()
    omega = np.zeros(3)  # deg/s
    estimate = angle.copy()
    integral = np.zeros(3)
    prev_error = wrap_deg(target - estimate)
    # Independent-axis Kalman-style estimator states
    P = np.ones(3) * 4.0
    q = np.ones(3) * 0.03
    r = np.ones(3) * sensor_noise**2
    B = np.array([0.0, 0.0, 1.0])  # normalized illustrative geomagnetic-field direction

    rows = []
    rng = np.random.default_rng(7)
    for idx, ti in enumerate(t):
        gyro_meas = omega + rng.normal(0, gyro_noise, 3)
        attitude_meas = wrap_deg(angle + rng.normal(0, sensor_noise, 3))

        # Kalman-style estimate: gyro prediction followed by attitude-reference correction.
        estimate = wrap_deg(estimate + gyro_meas * dt)
        P = P + q
        K = P / (P + r)
        estimate = wrap_deg(estimate + K * wrap_deg(attitude_meas - estimate))
        P = (1.0 - K) * P

        error = wrap_deg(target - estimate)
        integral = np.clip(integral + error * dt, -100, 100)
        derivative = wrap_deg(error - prev_error) / dt
        pid_cmd = (kp * error + ki * integral + kd * derivative) * 0.001
        prev_error = error.copy()

        wheel_torque = np.zeros(3)
        if use_wheel:
            wheel_torque = np.clip(pid_cmd, -max_torque, max_torque)

        # Magnetorquer can only generate torque perpendicular to the local magnetic field.
        mag_torque = np.zeros(3)
        if use_mtorquer:
            desired_mag_torque = pid_cmd - wheel_torque
            mag_torque = desired_mag_torque - np.dot(desired_mag_torque, B) * B
            mag_torque = np.clip(mag_torque, -0.02, 0.02)

        total_torque = wheel_torque + mag_torque
        # Small-angle educational model: independent principal axes; cross-coupling omitted.
        alpha_deg = np.rad2deg(total_torque / inertia)
        if idx < n - 1:
            omega = omega + alpha_deg * dt
            angle = wrap_deg(angle + omega * dt)

        rows.append([
            ti, *angle, *estimate, *gyro_meas, *error,
            *wheel_torque, *mag_torque, *total_torque,
            float(np.linalg.norm(error))
        ])

    cols = ["Time","Roll","Pitch","Yaw","Est_Roll","Est_Pitch","Est_Yaw",
            "Gyro_Roll","Gyro_Pitch","Gyro_Yaw","Err_Roll","Err_Pitch","Err_Yaw",
            "Wheel_Roll","Wheel_Pitch","Wheel_Yaw","Mag_Roll","Mag_Pitch","Mag_Yaw",
            "Torque_Roll","Torque_Pitch","Torque_Yaw","Error_Magnitude"]
    return pd.DataFrame(rows, columns=cols), inertia

if run or "sim_data" not in st.session_state:
    data, inertia = simulate()
    st.session_state.sim_data = data
    st.session_state.inertia = inertia
else:
    data = st.session_state.sim_data
    inertia = st.session_state.inertia

# Metrics
final_error = data["Error_Magnitude"].iloc[-1]
settled_threshold = 2.0
settled = data.loc[data["Error_Magnitude"] < settled_threshold, "Time"]
settling_time = float(settled.iloc[0]) if len(settled) else None
c1, c2, c3, c4 = st.columns(4)
c1.metric("Final attitude error", f"{final_error:.2f}°")
c2.metric("Settling time (≤2°)", f"{settling_time:.2f} s" if settling_time is not None else "Not reached")
c3.metric("Peak angular speed", f"{np.sqrt(data[['Gyro_Roll','Gyro_Pitch','Gyro_Yaw']].pow(2).sum(axis=1)).max():.2f} °/s")
c4.metric("Peak control torque", f"{np.sqrt(data[['Torque_Roll','Torque_Pitch','Torque_Yaw']].pow(2).sum(axis=1)).max():.4f} N·m")

left, right = st.columns([1.15, 1])
with left:
    st.subheader("1. 3D satellite orientation")
    fig3d = go.Figure()
    fig3d.add_trace(cube_trace(initial, "Initial orientation", "#9CA3AF", 0.35))
    fig3d.add_trace(cube_trace(target, "Desired orientation", "#22C55E", 0.35))
    # Display final simulated orientation as the solid satellite.
    final_att = data[["Roll","Pitch","Yaw"]].iloc[-1].to_numpy()
    fig3d.add_trace(cube_trace(final_att, "Final simulated orientation", "#38BDF8", 1.0))
    fig3d.update_layout(
        height=390, margin=dict(l=0,r=0,t=25,b=0),
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z",
                   aspectmode="data", camera=dict(eye=dict(x=1.6,y=1.6,z=1.2))),
        legend=dict(orientation="h", y=-0.12)
    )
    st.plotly_chart(fig3d, use_container_width=True)

with right:
    st.subheader("2. Sensor and estimator readings")
    st.write("Gyroscope: angular-rate measurement (°/s)")
    st.write("Magnetometer: idealized attitude-reference measurement (°)")
    st.write("Estimator: gyro prediction + Kalman-style correction")
    sensor_fig = go.Figure()
    for axis, color in zip(["Roll","Pitch","Yaw"], ["#EF4444","#22C55E","#3B82F6"]):
        sensor_fig.add_trace(go.Scatter(x=data["Time"], y=data[axis], name=f"Actual {axis}", line=dict(color=color)))
        sensor_fig.add_trace(go.Scatter(x=data["Time"], y=data[f"Est_{axis}"], name=f"Estimated {axis}", line=dict(dash="dash", color=color)))
    sensor_fig.update_layout(height=330, margin=dict(l=0,r=0,t=20,b=0), xaxis_title="Time (s)", yaxis_title="Angle (°)", legend=dict(orientation="h"))
    st.plotly_chart(sensor_fig, use_container_width=True)

st.subheader("3. Attitude error and control performance")
err_fig = go.Figure()
for axis, color in zip(["Roll","Pitch","Yaw"], ["#EF4444","#22C55E","#3B82F6"]):
    err_fig.add_trace(go.Scatter(x=data["Time"], y=data[f"Err_{axis}"], name=f"{axis} error", line=dict(color=color)))
err_fig.add_hline(y=0, line_dash="dot", line_color="gray")
err_fig.update_layout(height=280, xaxis_title="Time (s)", yaxis_title="Error (°)", margin=dict(l=0,r=0,t=15,b=0))
st.plotly_chart(err_fig, use_container_width=True)

torque_fig = make_subplots(rows=1, cols=2, subplot_titles=("Actuator torque", "Angular velocity"))
for axis, color in zip(["Roll","Pitch","Yaw"], ["#EF4444","#22C55E","#3B82F6"]):
    torque_fig.add_trace(go.Scatter(x=data["Time"], y=data[f"Wheel_{axis}"], name=f"Wheel {axis}", line=dict(color=color)), row=1, col=1)
    torque_fig.add_trace(go.Scatter(x=data["Time"], y=data[f"Gyro_{axis}"], name=f"Gyro {axis}", line=dict(dash="dot", color=color)), row=1, col=2)
torque_fig.update_layout(height=330, margin=dict(l=0,r=0,t=45,b=0), legend=dict(orientation="h"))
torque_fig.update_xaxes(title_text="Time (s)", row=1, col=1)
torque_fig.update_xaxes(title_text="Time (s)", row=1, col=2)
torque_fig.update_yaxes(title_text="Torque (N·m)", row=1, col=1)
torque_fig.update_yaxes(title_text="Angular rate (°/s)", row=1, col=2)
st.plotly_chart(torque_fig, use_container_width=True)

st.subheader("4. Simulation data")
st.dataframe(data.iloc[::max(1, len(data)//20)].round(4), use_container_width=True, hide_index=True)
csv = data.to_csv(index=False).encode("utf-8")
st.download_button("⬇ Download simulation data (CSV)", csv, "adcs_sim_results.csv", "text/csv")

with st.expander("Model assumptions and limitations"):
    st.markdown("""
    - The satellite is represented by three Euler angles and three independent rotational axes.
    - The dynamics use illustrative principal moments of inertia; gyroscopic cross-coupling is omitted.
    - The estimator is a simple independent-axis Kalman-style demonstration, **not a full quaternion MEKF**.
    - The magnetometer attitude reference is idealized. A real system estimates attitude from vector observations and known reference fields; one magnetometer alone cannot determine full 3-axis attitude.
    - The reaction wheel and magnetorquer are simplified torque sources. Real actuators have momentum, saturation, geometry and environmental constraints.
    - Use this project for classroom demonstration, not real spacecraft design or operation.
    """)

st.success("Demonstration complete: satellite model, sensor/estimator, PID control, actuators and performance dashboard are shown above.")
