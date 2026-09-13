import numpy as np

def solve(num_vars, num_constraints, opt_type, obj_coeffs, A_orig, signs, b_orig):
    # 1. Analyze constraints to allocate variables
    num_slacks=signs.count('<=')
    num_surplus=signs.count('>=')
    num_artificial=signs.count('>=')+signs.count('=')
    
    total_vars=num_vars+num_slacks+num_surplus+num_artificial
    
    # Tableau dimensions: (Constraints + 1 Objective Row) x (Variables + 1 RHS Column)
    tableau=np.zeros((num_constraints + 1, total_vars + 1))
    
    # Track the basic variable for each constraint row (stores column index)
    basis=[-1]*num_constraints
    
    # Big-M Penalty
    M=1e6
    
    # 2. Setup Objective Row (Index -1)
    # The Simplex algorithm here is standardized to always MAXIMIZE.
    # For MAX Z: Z - cx = 0 (Coefficients are -c)
    # For MIN Z: Maximize -Z. -Z + cx = 0 (Coefficients are +c)
    if opt_type=='MAX':
        tableau[-1, :num_vars]=-np.array(obj_coeffs)
    else:
        tableau[-1, :num_vars]=np.array(obj_coeffs)
        
    # 3. Populate Tableau Constraints
    slack_idx=num_vars
    surplus_idx=num_vars+num_slacks
    art_idx=num_vars+num_slacks+num_surplus
    
    for i in range(num_constraints):
        tableau[i, :num_vars]=A_orig[i]
        tableau[i, -1]=b_orig[i]
        
        if signs[i]=='<=':
            tableau[i, slack_idx]=1
            basis[i]=slack_idx
            slack_idx+=1
            
        elif signs[i]=='>=':
            tableau[i, surplus_idx]=-1
            tableau[i, art_idx]=1
            tableau[-1, art_idx]=M  # Initial Big-M coefficient in Z-row
            basis[i]=art_idx
            surplus_idx+=1
            art_idx+=1
            
        elif signs[i]=='=':
            tableau[i, art_idx]=1
            tableau[-1, art_idx]=M  # Initial Big-M coefficient in Z-row
            basis[i]=art_idx
            art_idx+=1

    # 4. Standardize Z-row (Eliminate Artificial Variables from Basis)
    # To start the Simplex method, the Z-row must have 0s in basic columns.
    for i in range(num_constraints):
        if basis[i]>=num_vars+num_slacks+num_surplus:
            tableau[-1, :] -= M * tableau[i, :]

    print("\n--- Initial Simplex Tableau ---")
    print(np.round(tableau, 2))
    
    # 5. Simplex Algorithm Loop
    iteration=1
    while True:
        # Check optimality condition: all coefficients in Z-row (excluding RHS) must be >= 0
        z_row_coeffs = tableau[-1, :-1]
        if np.all(z_row_coeffs>=-1e-7):
            break
            
        # Determine entering variable (most negative in Z-row)
        pivot_col=np.argmin(z_row_coeffs)
        
        # Determine leaving variable (Minimum ratio test)
        col_vals=tableau[:-1, pivot_col]
        rhs_vals=tableau[:-1, -1]
        
        # Consider only positive entries in the pivot column for ratios
        valid_rows=np.where(col_vals > 1e-7)[0]
        if len(valid_rows)==0:
            print("\nResult: UNBOUNDED SOLUTION. The feasible region is infinite in the optimal direction.")
            return
            
        ratios=rhs_vals[valid_rows] / col_vals[valid_rows]
        pivot_row=valid_rows[np.argmin(ratios)]
        
        # Pivot Operations
        basis[pivot_row]=pivot_col
        pivot_element=tableau[pivot_row, pivot_col]
        
        # Normalize the pivot row
        tableau[pivot_row, :]/=pivot_element
        
        # Eliminate entering variable from all other rows (including Z-row)
        for i in range(num_constraints+1):
            if i!=pivot_row:
                tableau[i, :]-=tableau[i, pivot_col]*tableau[pivot_row, :]
                
        iteration+=1
        if iteration>1000:
            print("\nResult: FAILED TO CONVERGE within 1000 iterations.")
            return

    # 6. Extract and Format Results
    solution=np.zeros(total_vars)
    for i in range(num_constraints):
        solution[basis[i]]=tableau[i, -1]
        
    # Check for infeasibility: are any artificial variables strictly positive?
    artificial_sum=np.sum(solution[num_vars + num_slacks + num_surplus:])
    if artificial_sum>1e-5:
        print("\nResult: INFEASIBLE. Artificial variables remain in the optimal basis.")
        return
        
    # Read optimal objective value
    opt_val=tableau[-1, -1]
    # If the original problem was a MIN problem, we maximized -Z, so flip the sign back
    if opt_type=='MIN':
        opt_val=-opt_val

    print("\n=== OPTIMAL SOLUTION FOUND ===")
    print(f"Iterations: {iteration-1}")
    print(f"Optimal Objective Value ({opt_type}): {opt_val:.4f}\n")
    
    print("Variable Values:")
    for i in range(num_vars):
        print(f"  Decision x{i+1}: {solution[i]:.4f}")
        
    curr_idx = num_vars
    for i in range(num_slacks):
        print(f"  Slack s{i+1}: {solution[curr_idx]:.4f}")
        curr_idx+=1
        
    for i in range(num_surplus):
        print(f"  Surplus e{i+1}: {solution[curr_idx]:.4f}")
        curr_idx+=1
        
    for i in range(num_artificial):
        print(f"  Artificial a{i+1}: {solution[curr_idx]:.4f}")
        curr_idx+=1


def get_user_inputs():
    print("--- LPP Solver ---")
    num_vars=int(input("Enter number of decision variables: "))
    num_constraints=int(input("Enter number of constraints: "))
    opt_type=input("Optimization type (MIN/MAX): ").strip().upper()
    
    print(f"\nEnter objective function coefficients for {num_vars} variables (space-separated):")
    obj_coeffs=list(map(float, input().strip().split()))
    
    A_orig,signs,b_orig=[],[],[]
    
    print("\nFor each constraint, enter the coefficients, sign (<=, >=, =), and RHS value.")
    print(f"Example format for {num_vars} variables: " + " ".join(["1" for _ in range(num_vars)]) + " <= 4")
    
    for i in range(num_constraints):
        row_input=input(f"Constraint {i+1}: ").strip().split()
        A_orig.append(list(map(float, row_input[:num_vars])))
        signs.append(row_input[-2])
        b_orig.append(float(row_input[-1]))
        
    return num_vars, num_constraints, opt_type, obj_coeffs, A_orig, signs, b_orig


if __name__ == "__main__":
    try:
        inputs = get_user_inputs()
        solve(*inputs)
    except Exception as e:
        print(f"\nAn error occurred while parsing inputs: {e}")
        print("Ensure you enter numbers spaced out properly, like: 1 2 <= 5")
