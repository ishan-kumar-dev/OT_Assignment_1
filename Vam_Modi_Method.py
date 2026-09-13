import numpy as np

def calculate_penalties(costs, active_rows, active_cols):
    row_penalties = []
    for i in range(costs.shape[0]):
        if active_rows[i]:
            row_costs = [costs[i][j] for j in range(costs.shape[1]) if active_cols[j]]
            if len(row_costs) > 1:
                sorted_costs = sorted(row_costs)
                row_penalties.append((sorted_costs[1] - sorted_costs[0], i, 'row'))
            elif len(row_costs) == 1:
                row_penalties.append((row_costs[0], i, 'row'))
            
    col_penalties = []
    for j in range(costs.shape[1]):
        if active_cols[j]:
            col_costs = [costs[i][j] for i in range(costs.shape[0]) if active_rows[i]]
            if len(col_costs) > 1:
                sorted_costs = sorted(col_costs)
                col_penalties.append((sorted_costs[1] - sorted_costs[0], j, 'col'))
            elif len(col_costs) == 1:
                col_penalties.append((col_costs[0], j, 'col'))
                
    return row_penalties + col_penalties

def vam(costs, supply, demand):
    supply = supply.copy()
    demand = demand.copy()
    allocation = np.zeros(costs.shape)
    
    active_rows = [True] * len(supply)
    active_cols = [True] * len(demand)
    
    while any(active_rows) and any(active_cols):
        penalties = calculate_penalties(costs, active_rows, active_cols)
        if not penalties:
            break
            
        penalties.sort(key=lambda x: x[0], reverse=True)
        max_penalty = penalties[0]
        
        index = max_penalty[1]
        is_row = max_penalty[2] == 'row'
        
        if is_row:
            min_cost_col = min([j for j in range(costs.shape[1]) if active_cols[j]], 
                               key=lambda j: costs[index][j])
            r, c = index, min_cost_col
        else:
            min_cost_row = min([i for i in range(costs.shape[0]) if active_rows[i]], 
                               key=lambda i: costs[i][index])
            r, c = min_cost_row, index
            
        qty = min(supply[r], demand[c])
        allocation[r][c] = qty
        supply[r] -= qty
        demand[c] -= qty
        
        if supply[r] == 0:
            active_rows[r] = False
        if demand[c] == 0:
            active_cols[c] = False
            
    return allocation

def get_uv(allocation, costs):
    rows, cols = allocation.shape
    u = [None] * rows
    v = [None] * cols
    u[0] = 0
    
    basic_cells = [(i, j) for i in range(rows) for j in range(cols) if allocation[i][j] > 0]
    
    while None in u or None in v:
        for r, c in basic_cells:
            if u[r] is not None and v[c] is None:
                v[c] = costs[r][c] - u[r]
            elif v[c] is not None and u[r] is None:
                u[r] = costs[r][c] - v[c]
    return u, v

def get_loop(start_node, basic_cells):
    def dfs(current, path, is_row_move):
        if len(path) > 3 and path[0] == path[-1]:
            return path
        
        for r, c in basic_cells:
            if (r, c) not in path[:-1]:
                if is_row_move and r == current[0] and c != current[1]:
                    res = dfs((r, c), path + [(r, c)], not is_row_move)
                    if res: return res
                elif not is_row_move and c == current[1] and r != current[0]:
                    res = dfs((r, c), path + [(r, c)], not is_row_move)
                    if res: return res
        return None

    loop = dfs(start_node, [start_node], True)
    if not loop:
        loop = dfs(start_node, [start_node], False)
    return loop

def modi(allocation, costs):
    iteration = 1
    while True:
        rows, cols = allocation.shape
        u, v = get_uv(allocation, costs)
        
        penalties = np.zeros((rows, cols))
        min_penalty = 0
        entering_cell = None
        
        for i in range(rows):
            for j in range(cols):
                if allocation[i][j] == 0:
                    penalties[i][j] = costs[i][j] - (u[i] + v[j])
                    if penalties[i][j] < min_penalty:
                        min_penalty = penalties[i][j]
                        entering_cell = (i, j)
                        
        if min_penalty >= 0:
            print(f"\nOptimal solution reached after {iteration - 1} MODI iterations.")
            return allocation
            
        basic_cells = [(i, j) for i in range(rows) for j in range(cols) if allocation[i][j] > 0]
        basic_cells.append(entering_cell)
        
        loop = get_loop(entering_cell, basic_cells)
        minus_cells = loop[1:-1:2]
        theta = min(allocation[r][c] for r, c in minus_cells)
        
        for idx, (r, c) in enumerate(loop[:-1]):
            if idx % 2 == 0:
                allocation[r][c] += theta
            else:
                allocation[r][c] -= theta
                
        iteration += 1

def get_user_input():
    print("=== Transportation Problem Input ===")
    num_sources = int(input("Enter number of sources (rows): "))
    num_destinations = int(input("Enter number of destinations (columns): "))
    
    print("\nEnter the unit transportation cost matrix:")
    costs = []
    for i in range(num_sources):
        row = list(map(float, input(f"  Row {i+1} costs (separated by space): ").split()))
        if len(row) != num_destinations:
            raise ValueError(f"Expected {num_destinations} values for row {i+1}.")
        costs.append(row)
    costs = np.array(costs)
    
    print("\nEnter source capacities (Supply):")
    supply = np.array(list(map(float, input(f"  Enter {num_sources} values (separated by space): ").split())))
    if len(supply) != num_sources:
        raise ValueError(f"Expected {num_sources} supply values.")

    print("\nEnter destination requirements (Demand):")
    demand = np.array(list(map(float, input(f"  Enter {num_destinations} values (separated by space): ").split())))
    if len(demand) != num_destinations:
        raise ValueError(f"Expected {num_destinations} demand values.")

    # Check and handle unbalanced problem automatically by adding dummy rows/cols
    total_supply = np.sum(supply)
    total_demand = np.sum(demand)
    
    if total_supply != total_demand:
        print(f"\n[Note] Unbalanced problem detected (Total Supply: {total_supply}, Total Demand: {total_demand}).")
        if total_supply > total_demand:
            diff = total_supply - total_demand
            print(f"Adding Dummy Destination with demand = {diff}")
            demand = np.append(demand, diff)
            dummy_col = np.zeros((costs.shape[0], 1))
            costs = np.hstack((costs, dummy_col))
        else:
            diff = total_demand - total_supply
            print(f"Adding Dummy Source with supply = {diff}")
            supply = np.append(supply, diff)
            dummy_row = np.zeros((1, costs.shape[1]))
            costs = np.vstack((costs, dummy_row))
            
    return costs, supply, demand

if __name__ == "__main__":
    costs, supply, demand = get_user_input()

    print("\n--- Phase 1: Vogel's Approximation Method (VAM) ---")
    ibfs_allocation = vam(costs, supply, demand)
    ibfs_cost = np.sum(ibfs_allocation * costs)
    print("Initial Basic Feasible Solution Allocation Matrix:")
    print(ibfs_allocation)
    print(f"Total IBFS Cost: {ibfs_cost}")

    print("\n--- Phase 2: MODI Method (Optimality Test) ---")
    optimal_allocation = modi(ibfs_allocation, costs)
    optimal_cost = np.sum(optimal_allocation * costs)
    
    print("\nOptimal Shipment Plan Allocation Matrix:")
    print(optimal_allocation)
    print(f"Minimum Total Transportation Cost: {optimal_cost}")
