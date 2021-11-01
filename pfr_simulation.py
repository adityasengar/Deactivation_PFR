import numpy as np
import pandas as pd
import argparse
import os
import json
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def d_propylene_conc_dz(activity, propylene_conc, params):
    """
    Calculates the spatial derivative of the normalized propylene concentration
    using the 1MARI model.
    dx/dz' = -k_reaction * (x*λ₁ + x²*λ₁*ξ_m / x₀)
    """
    k_reaction = params.get('k_reaction', 5.0)
    xi_m = params.get('xi_m', 2.0)
    initial_x = params.get('initial_x', 1.0)
    
    x = propylene_conc
    x0 = initial_x

    term1 = -activity * x
    term2 = -activity * xi_m * x**2 / x0
    return k_reaction * (term1 + term2)

def solve_pfr_profile(activity_profile, params):
    """
    Solves the PFR profile for a given catalyst activity profile.
    """
    num_segments = params.get('num_segments', 100)
    total_length = params.get('total_length', 1.0)
    dz = total_length / num_segments
    x0 = params.get('initial_x', 1.0)

    z_profile = np.linspace(0, total_length, num_segments + 1)
    propylene_profile = np.zeros(num_segments + 1)
    propylene_profile[0] = x0
    
    for i in range(num_segments):
        dx_dz = d_propylene_conc_dz(activity_profile[i], propylene_profile[i], params)
        propylene_profile[i+1] = propylene_profile[i] + dx_dz * dz
        
    return z_profile, propylene_profile

def update_activity(activity_profile, propylene_profile, params):
    """
    Updates the catalyst activity profile over a time step dt.
    Uses a propylene-dependent deactivation model: d(lambda)/dt = -k_d * lambda * x.
    """
    dt = params.get('dt', 0.1)
    k_d = params.get('k_d', 1.0)
    
    # Deactivation rate depends on local propylene concentration
    d_lambda_dt = -k_d * activity_profile * propylene_profile
    new_activity_profile = activity_profile + d_lambda_dt * dt
    # Ensure activity doesn't go below zero
    new_activity_profile[new_activity_profile < 0] = 0
    return new_activity_profile

def run_time_dependent_simulation(params):
    """
    Runs the time-dependent PFR simulation.
    """
    num_segments = params.get('num_segments', 100)
    total_length = params.get('total_length', 1.0)
    lambda0 = params.get('initial_lambda', 1.0)
    t_final = params.get('t_final', 2.0)
    dt = params.get('dt', 0.1)

    activity_profile = np.full(num_segments + 1, lambda0)
    
    time_points = np.arange(0, t_final, dt)
    results_over_time = []

    print("Running time-dependent PFR simulation...")
    for t in time_points:
        z_profile, propylene_profile = solve_pfr_profile(activity_profile, params)
        
        results_over_time.append({
            'time': t,
            'reactor_length': z_profile,
            'propylene_concentration': propylene_profile,
            'catalyst_activity': activity_profile.copy()
        })

        activity_profile = update_activity(activity_profile, propylene_profile, params)

    return results_over_time

def plot_animation(results_over_time, params, output_path):
    """
    Creates an animation of the simulation results.
    """
    print(f"Generating animation to {output_path}...")
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    ax1.set_xlabel("Reactor Length (z')")
    
    color = 'tab:red'
    ax1.set_ylabel('Normalized Propylene Concentration', color=color)
    line1, = ax1.plot([], [], color=color, label='Propylene Conc.')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(0, params.get('initial_x', 1.0) * 1.1)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Fraction of Active Sites (λ)', color=color)
    line2, = ax2.plot([], [], color=color, label='Catalyst Activity')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(0, params.get('initial_lambda', 1.0) * 1.1)

    title = ax1.text(0.5, 1.05, '', transform=ax1.transAxes, ha="center")
    
    ax1.set_xlim(0, params.get('total_length', 1.0))

    def init():
        line1.set_data([], [])
        line2.set_data([], [])
        title.set_text('')
        return line1, line2, title

    def animate(i):
        data = results_over_time[i]
        line1.set_data(data['reactor_length'], data['propylene_concentration'])
        line2.set_data(data['reactor_length'], data['catalyst_activity'])
        title.set_text(f'PFR Profiles at t = {data["time"]:.2f}')
        return line1, line2, title

    ani = animation.FuncAnimation(fig, animate, frames=len(results_over_time),
                                  init_func=init, blit=True, interval=200)
    
    ani.save(output_path, writer='imagemagick')
    print(f"Animation saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="PFR Deactivation Simulation")
    parser.add_argument('--config', type=str, default='config/default.json', help="Path to config file.")
    # Add other arguments to override config file, setting defaults to None
    args_list = {
        'segments': int, 'length': float, 't_final': float, 'dt': float,
        'k_reaction': float, 'k_deactivation': float, 'xi_m': float,
        'output_dir': str
    }
    for arg, arg_type in args_list.items():
        parser.add_argument(f'--{arg}', type=arg_type, default=None)

    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config_params = json.load(f)
    
    params = {**config_params.get('simulation_params', {}), **config_params.get('model_params', {})}

    if args.segments: params['num_segments'] = args.segments
    if args.length: params['total_length'] = args.length
    if args.t_final: params['t_final'] = args.t_final
    if args.dt: params['dt'] = args.dt
    if args.k_reaction: params['k_reaction'] = args.k_reaction
    if args.k_deactivation: params['k_d'] = args.k_deactivation
    if args.xi_m: params['xi_m'] = args.xi_m

    output_dir = args.output_dir or config_params.get('output_params', {}).get('output_dir', 'results')
    plot_filename = config_params.get('output_params', {}).get('plot_filename', 'pfr_profile_animated.gif')
    
    os.makedirs(output_dir, exist_ok=True)
    
    results = run_time_dependent_simulation(params)
    
    print("\n--- Simulation Complete ---")
    
    plot_animation(results, params, os.path.join(output_dir, plot_filename))

if __name__ == "__main__":
    main()
