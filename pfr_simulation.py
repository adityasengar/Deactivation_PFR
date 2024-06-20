import numpy as np
import pandas as pd
import argparse
import os
import matplotlib.pyplot as plt

def F_x(x, params):
    k_d = params.get('k_d', 1.0)
    iC4_conc = params.get('iC4_conc', 1.0)
    x_k_T = params.get('x_k_T', 1.0)
    return (k_d / iC4_conc) * (x_k_T + x)

def d_lambda_dt(lambda_val, x, params):
    sum_C_i = x * 0.1 
    return -F_x(x, params) * lambda_val * (sum_C_i / params.get('iC4_conc', 1.0))

def d_C_prime_dz(lambda_val, x, params):
    k_reaction = params.get('k_reaction', 10.0)
    return k_reaction * lambda_val * x

def run_pfr_simulation(params):
    num_segments = params.get('num_segments', 100)
    total_length = params.get('total_length', 1.0)
    dz = total_length / num_segments
    x0 = params.get('initial_x', 1.0)
    lambda0 = params.get('initial_lambda', 1.0)
    z_profile = np.linspace(0, total_length, num_segments + 1)
    x_profile = np.zeros(num_segments + 1)
    lambda_profile = np.zeros(num_segments + 1)
    x_profile[0] = x0
    lambda_profile[0] = lambda0
    
    print("Running PFR simulation...")
    for i in range(num_segments):
        dx_dz = -d_C_prime_dz(lambda_profile[i], x_profile[i], params)
        dlambda_dz = -d_lambda_dt(lambda_profile[i], x_profile[i], params) 
        x_profile[i+1] = x_profile[i] + dx_dz * dz
        lambda_profile[i+1] = lambda_profile[i] + dlambda_dz * dz
        
    return pd.DataFrame({
        'reactor_length': z_profile,
        'propylene_conc': x_profile,
        'active_sites': lambda_profile
    })

def plot_results(df, output_path):
    print(f"Generating plot to {output_path}...")
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color = 'tab:red'
    ax1.set_xlabel('Reactor Length (z\')')
    ax1.set_ylabel('Propylene Concentration (x/x0)', color=color)
    ax1.plot(df['reactor_length'], df['propylene_conc'], color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Fraction of Active Sites (λ)', color=color)
    ax2.plot(df['reactor_length'], df['active_sites'], color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('PFR Concentration and Active Site Profiles')
    fig.tight_layout()
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="PFR Deactivation Simulation")
    parser.add_argument('--segments', type=int, default=100, help="Number of reactor segments for simulation.")
    parser.add_argument('--length', type=float, default=1.0, help="Total reactor length.")
    parser.add_argument('--k_reaction', type=float, default=5.0, help="Reaction rate constant.")
    parser.add_argument('--k_deactivation', type=float, default=1.0, help="Deactivation rate constant (k_d).")
    parser.add_argument('--output_dir', type=str, default="results", help="Directory to save outputs.")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    params = {
        'num_segments': args.segments, 'total_length': args.length,
        'initial_x': 1.0, 'initial_lambda': 1.0, 'k_d': args.k_deactivation,
        'iC4_conc': 10.0, 'x_k_T': 1.0, 'k_reaction': args.k_reaction
    }
    
    results = run_pfr_simulation(params)
    
    print("\n--- Simulation Complete ---")
    print(results.tail())
    
    plot_results(results, os.path.join(args.output_dir, "pfr_profile.png"))

if __name__ == "__main__":
    main()
