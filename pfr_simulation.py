import numpy as np
import pandas as pd
import argparse
import os
import json
import matplotlib.pyplot as plt

def F_x(propylene_conc, params):
    """
    Calculates an intermediate term for the deactivation rate expression.
    """
    k_d = params.get('k_d', 1.0)
    iC4_conc = params.get('iC4_conc', 1.0)
    x_k_T = params.get('x_k_T', 1.0)
    return (k_d / iC4_conc) * (x_k_T + propylene_conc)

def d_lambda_dz(activity, propylene_conc, params):
    """
    Calculates the spatial derivative of the fraction of active sites (lambda).
    This is a simplified model for catalyst deactivation along the reactor length.
    d(lambda)/dz = -F_x * lambda * (C_oligomer / C_iC4)
    """
    # Assuming oligomer concentration is proportional to propylene concentration
    oligomer_conc = propylene_conc * 0.1 
    iC4_conc = params.get('iC4_conc', 1.0)
    return -F_x(propylene_conc, params) * activity * (oligomer_conc / iC4_conc)

def d_propylene_conc_dz(activity, propylene_conc, params):
    """
    Calculates the spatial derivative of the normalized propylene concentration.
    This is a simplified model for the reaction rate.
    dx/dz = -k_reaction * lambda * x
    """
    k_reaction = params.get('k_reaction', 10.0)
    return -k_reaction * activity * propylene_conc

def run_pfr_simulation(params):
    """
    Runs the Plug Flow Reactor (PFR) simulation.
    """
    num_segments = params.get('num_segments', 100)
    total_length = params.get('total_length', 1.0)
    dz = total_length / num_segments

    # Initial conditions
    x0 = params.get('initial_x', 1.0)
    lambda0 = params.get('initial_lambda', 1.0)

    # Profiles along the reactor
    z_profile = np.linspace(0, total_length, num_segments + 1)
    propylene_profile = np.zeros(num_segments + 1)
    activity_profile = np.zeros(num_segments + 1)
    
    propylene_profile[0] = x0
    activity_profile[0] = lambda0
    
    print("Running PFR simulation...")
    # Integrate along the reactor length using Euler's method
    for i in range(num_segments):
        dx_dz = d_propylene_conc_dz(activity_profile[i], propylene_profile[i], params)
        dlambda_dz = d_lambda_dz(activity_profile[i], propylene_profile[i], params) 
        
        propylene_profile[i+1] = propylene_profile[i] + dx_dz * dz
        activity_profile[i+1] = activity_profile[i] + dlambda_dz * dz
        
    return pd.DataFrame({
        'reactor_length': z_profile,
        'propylene_concentration': propylene_profile,
        'catalyst_activity': activity_profile
    })

def plot_results(df, params, output_path):
    """
    Plots the simulation results.
    """
    print(f"Generating plot to {output_path}...")
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color = 'tab:red'
    ax1.set_xlabel("Reactor Length (z')")
    ax1.set_ylabel('Normalized Propylene Concentration', color=color)
    ax1.plot(df['reactor_length'], df['propylene_concentration'], color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Fraction of Active Sites (λ)', color=color)
    ax2.plot(df['reactor_length'], df['catalyst_activity'], color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    title = (
        'PFR Concentration and Activity Profiles\n'
        f"$k_{{reaction}} = {params.get('k_reaction')}$, $k_{{d}} = {params.get('k_d')}$"
    )
    plt.title(title)
    fig.tight_layout()
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="PFR Deactivation Simulation")
    parser.add_argument('--config', type=str, default='config/default.json', help="Path to config file.")
    parser.add_argument('--segments', type=int, default=None, help="Number of reactor segments for simulation.")
    parser.add_argument('--length', type=float, default=None, help="Total reactor length.")
    parser.add_argument('--k_reaction', type=float, default=None, help="Reaction rate constant.")
    parser.add_argument('--k_deactivation', type=float, default=None, help="Deactivation rate constant (k_d).")
    parser.add_argument('--output_dir', type=str, default=None, help="Directory to save outputs.")
    args = parser.parse_args()

    # Load params from config file
    with open(args.config, 'r') as f:
        config_params = json.load(f)
    
    params = {**config_params['simulation_params'], **config_params['model_params']}

    # Override params with command line arguments if provided
    if args.segments is not None:
        params['num_segments'] = args.segments
    if args.length is not None:
        params['total_length'] = args.length
    if args.k_reaction is not None:
        params['k_reaction'] = args.k_reaction
    if args.k_deactivation is not None:
        params['k_d'] = args.k_deactivation
    
    output_dir = args.output_dir if args.output_dir is not None else config_params['output_params']['output_dir']
    plot_filename = config_params['output_params']['plot_filename']
    
    os.makedirs(output_dir, exist_ok=True)
    
    results = run_pfr_simulation(params)
    
    print("\n--- Simulation Complete ---")
    print(results.tail())
    
    plot_results(results, params, os.path.join(output_dir, plot_filename))

if __name__ == "__main__":
    main()
