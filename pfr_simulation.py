import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

def F_x(x, params):
    """Simplified F(x) from power-law kinetics."""
    k_d = params.get('k_d', 1.0)
    iC4_conc = params.get('iC4_conc', 1.0)
    x_k_T = params.get('x_k_T', 1.0)
    return (k_d / iC4_conc) * (x_k_T + x)

def d_lambda_dt(lambda_val, x, params):
    """Rate of change of surface site concentration (dλ/dt')."""
    sum_C_i = x * 0.1 
    return -F_x(x, params) * lambda_val * (sum_C_i / params.get('iC4_conc', 1.0))

def d_C_prime_dz(lambda_val, x, params):
    """Change in product conc. along the reactor."""
    k_reaction = params.get('k_reaction', 10.0)
    return k_reaction * lambda_val * x

def run_pfr_simulation(params):
    """
    Simulates the PFR by treating it as a series of CSTRs.
    """
    # Simulation parameters
    num_segments = params.get('num_segments', 100) # Number of CSTRs in series
    total_length = params.get('total_length', 1.0)
    dz = total_length / num_segments
    
    # Initial conditions at reactor inlet (z=0)
    x0 = params.get('initial_x', 1.0)
    lambda0 = params.get('initial_lambda', 1.0)
    
    # Arrays to store profiles
    z_profile = np.linspace(0, total_length, num_segments + 1)
    x_profile = np.zeros(num_segments + 1)
    lambda_profile = np.zeros(num_segments + 1)
    
    x_profile[0] = x0
    lambda_profile[0] = lambda0
    
    print("Running PFR simulation...")
    # Iterate through each segment of the reactor
    for i in range(num_segments):
        # This is a simplified Euler integration for demonstration
        # A more complex solver would be used for higher accuracy
        
        # Calculate rates at the start of the segment
        dx_dz = -d_C_prime_dz(lambda_profile[i], x_profile[i], params) # Consumption of x
        dlambda_dz = -d_lambda_dt(lambda_profile[i], x_profile[i], params) # Deactivation along length
        
        # Update concentrations for the next segment
        x_profile[i+1] = x_profile[i] + dx_dz * dz
        lambda_profile[i+1] = lambda_profile[i] + dlambda_dz * dz
        
    results = pd.DataFrame({
        'reactor_length': z_profile,
        'propylene_conc': x_profile,
        'active_sites': lambda_profile
    })
    
    return results

def main():
    """Main function to run the PFR simulation."""
    # Define a default set of parameters
    default_params = {
        'num_segments': 100,
        'total_length': 1.0,
        'initial_x': 1.0,
        'initial_lambda': 1.0,
        'k_d': 1.0,
        'iC4_conc': 10.0,
        'x_k_T': 1.0,
        'k_reaction': 5.0
    }
    
    simulation_results = run_pfr_simulation(default_params)
    
    print("\n--- Simulation Results ---")
    print("Concentration profiles along the Plug Flow Reactor:")
    print(simulation_results.tail())

if __name__ == "__main__":
    main()