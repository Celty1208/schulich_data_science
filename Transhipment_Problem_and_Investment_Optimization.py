
# Part 1:
import pandas as pd
from gurobipy import GRB
import gurobipy as gb

# Import files
Refinement_Demand = pd.read_csv('/Users/celty/Desktop/OMIS 6000 OPERATIONAL/ASG 1/data pt_1/Refinement_Demand.csv')
CUMCF_Capacity = pd.read_csv('/Users/celty/Desktop/OMIS 6000 OPERATIONAL/ASG 1/data pt_1/Capacity_for_Direct_Production_Facilities.csv')
Distribution_Capacity = pd.read_csv('/Users/celty/Desktop/OMIS 6000 OPERATIONAL/ASG 1/data pt_1/Capacity_for_Transship_Distribution_Centers.csv')
IUG_Capacity = pd.read_csv('/Users/celty/Desktop/OMIS 6000 OPERATIONAL/ASG 1/data pt_1/Capacity_for_Transship_Production_Facilities.csv')
CUMCF_Cost = pd.read_csv('/Users/celty/Desktop/OMIS 6000 OPERATIONAL/ASG 1/data pt_1/Cost_Production_to_Refinement.csv')
IUG_Cost = pd.read_csv('/Users/celty/Desktop/OMIS 6000 OPERATIONAL/ASG 1/data pt_1/Cost_Production_to_Transshipment.csv')
Distribution_Cost = pd.read_csv('/Users/celty/Desktop/OMIS 6000 OPERATIONAL/ASG 1/data pt_1/Cost_Transshipment_to_Refinement.csv')


# Create Model
model = gb.Model("Minimize Transportation Cost")

# Create the a single class of decision variables
x = model.addVars(25, 5, lb=0, vtype=GRB.CONTINUOUS, name="CUMCF")
y = model.addVars(15, 2, lb=0, vtype=GRB.CONTINUOUS, name="IUG")
z = model.addVars(2, 5, lb=0, vtype=GRB.CONTINUOUS, name="Distribution")


# The objective function
model.setObjective(
    gb.quicksum(CUMCF_Cost.set_index(['ProductionFacility', 'RefinementCenter']).loc[(i+1, k+1), 'Cost'] * x[i, k] for i in range(25) for k in range(5)) +
    gb.quicksum(IUG_Cost.set_index(['ProductionFacility', 'TransshipmentHub']).loc[(i+1, j+1), 'Cost'] * y[i, j] for i in range(15) for j in range(2)) +
    gb.quicksum(Distribution_Cost.set_index(['TransshipmentHub', 'RefinementCenter']).loc[(j+1, k+1), 'Cost'] * z[j, k] for j in range(2) for k in range(5)), 
    gb.GRB.MINIMIZE)

# Constraints - Capacity from CUMCF
for i in range(25):
    model.addConstr(gb.quicksum(x[i, k] for k in range(5)) <= CUMCF_Capacity.loc[i, 'Capacity'])
    
# Constraints - Capacity from IUG
for i in range(15): 
    model.addConstr(gb.quicksum(y[i, j] for j in range(2)) <= IUG_Capacity.loc[i, 'Capacity'])
    
# Constraints - Capacity from Transshipment Hub
for j in range(2):
    total_input = gb.quicksum(y[i, j] for i in range(15))
    total_output = gb.quicksum(z[j, k] for k in range(5))
    model.addConstr(total_input == total_output)
    model.addConstr(total_output <= Distribution_Capacity.loc[j, 'Capacity'])

# Constraints - Demand from Refinement Center
for k in range(5):
    model.addConstr(gb.quicksum(x[i, k] for i in range(25)) + gb.quicksum(z[j, k] for j in range(2)) == Refinement_Demand.loc[k, 'Demand'])

# Optimally solve the problem
model.optimize()


# Print results
if model.status == gb.GRB.OPTIMAL:
    solution_x = model.getAttr('x', x)
    solution_y = model.getAttr('x', y)
    solution_z = model.getAttr('x', z)
    print("Optimal solution found:")
    for key, value in solution_x.items():
        print(f"CUMCF[{key}]: {value}")
    for key, value in solution_y.items():
        print(f"IUG[{key}]: {value}")
    for key, value in solution_z.items():
        print(f"Distribution[{key}]: {value}")
else:
    print("No optimal solution found.")


# Question: In the optimal solution, what proportion of canola oil is transshipped?
CUMCF_total = sum(x[i, k].X for i in range(25) for k in range(5))
UGI_total = sum(y[i, j].X for i in range(15) for j in range(2))

total_transport = CUMCF_total + UGI_total
transship_proportion = UGI_total / total_transport

print(transship_proportion)


# Minimize Transportation Cost_Objctive
model_1 = gb.Model("Minimize Transportation Cost_Objctive")

x_1 = model_1.addVars(25, 5, lb=0, vtype=GRB.CONTINUOUS, name="CUMCF")
y_1 = model_1.addVars(15, 2, lb=0, vtype=GRB.CONTINUOUS, name="IUG")
z_1 = model_1.addVars(2, 5, lb=0, vtype=GRB.CONTINUOUS, name="Distribution")

# Add a penalty cost for transshipped canola oil
Penalty_cost = 0.5
model_1.setObjective(
    gb.quicksum(CUMCF_Cost.set_index(['ProductionFacility', 'RefinementCenter']).loc[(i+1, k+1), 'Cost'] * x_1[i, k] for i in range(25) for k in range(5)) +
    gb.quicksum(IUG_Cost.set_index(['ProductionFacility', 'TransshipmentHub']).loc[(i+1, j+1), 'Cost'] * y_1[i, j] for i in range(15) for j in range(2)) +
    gb.quicksum(Distribution_Cost.set_index(['TransshipmentHub', 'RefinementCenter']).loc[(j+1, k+1), 'Cost'] * z_1[j, k] for j in range(2) for k in range(5)) +
    Penalty_cost * gb.quicksum(y_1[i, j] for i in range(15) for j in range(2)), 
    gb.GRB.MINIMIZE)

# Constraints - Capacity from CUMCF
for i in range(25):
    model_1.addConstr(gb.quicksum(x_1[i, k] for k in range(5)) <= CUMCF_Capacity.loc[i, 'Capacity'])
    
# Constraints - Capacity from IUG
for i in range(15): 
    model_1.addConstr(gb.quicksum(y_1[i, j] for j in range(2)) <= IUG_Capacity.loc[i, 'Capacity'])
    
# Constraints - Capacity from Transshipment Hub
for j in range(2):
    total_input = gb.quicksum(y_1[i, j] for i in range(15))
    total_output = gb.quicksum(z_1[j, k] for k in range(5))
    model_1.addConstr(total_input == total_output)
    model_1.addConstr(total_output <= Distribution_Capacity.loc[j, 'Capacity'])

# Constraints - Demand from Refinement Center
for k in range(5):
    model_1.addConstr(gb.quicksum(x_1[i, k] for i in range(25)) + gb.quicksum(z_1[j, k] for j in range(2)) == Refinement_Demand.loc[k, 'Demand'])

# Optimally solve the problem
model_1.optimize()

CUMCF_total = sum(x_1[i, k].X for i in range(25) for k in range(5))
UGI_total = sum(y_1[i, j].X for i in range(15) for j in range(2))

total_transport = UCCFM_total + UGI_total
transship_proportion = UGI_total / total_transport

print(transship_proportion)   


# Minimize Transportation Cost_Constraintmodel_2 = gb.Model("Minimize Transportation Cost_Constraint")
x_2 = model_2.addVars(25, 5, lb=0, vtype=GRB.CONTINUOUS, name="CUMCF")
y_2 = model_2.addVars(15, 2, lb=0, vtype=GRB.CONTINUOUS, name="IUG")
z_2 = model_2.addVars(2, 5, lb=0, vtype=GRB.CONTINUOUS, name="Distribution")

model_2.setObjective(
    gb.quicksum(CUMCF_Cost.set_index(['ProductionFacility', 'RefinementCenter']).loc[(i+1, k+1), 'Cost'] * x_2[i, k] for i in range(25) for k in range(5)) +
    gb.quicksum(IUG_Cost.set_index(['ProductionFacility', 'TransshipmentHub']).loc[(i+1, j+1), 'Cost'] * y_2[i, j] for i in range(15) for j in range(2)) +
    gb.quicksum(Distribution_Cost.set_index(['TransshipmentHub', 'RefinementCenter']).loc[(j+1, k+1), 'Cost'] * z_2[j, k] for j in range(2) for k in range(5)), 
    gb.GRB.MINIMIZE)

# Constraints - Capacity from CUMCF
for i in range(25):
    model_2.addConstr(gb.quicksum(x_2[i, k] for k in range(5)) <= CUMCF_Capacity.loc[i, 'Capacity'])
    
# Constraints - Capacity from IUG
for i in range(15): 
    model_2.addConstr(gb.quicksum(y_2[i, j] for j in range(2)) <= IUG_Capacity.loc[i, 'Capacity'])
    
# Constraints - Capacity from Transshipment Hub
for j in range(2):
    total_input = gb.quicksum(y_2[i, j] for i in range(15))
    total_output = gb.quicksum(z_2[j, k] for k in range(5))
    model_2.addConstr(total_input == total_output)
    model_2.addConstr(total_output <= Distribution_Capacity.loc[j, 'Capacity'])

# Constraints - Demand from Refinement Center
for k in range(5):
    model_2.addConstr(gb.quicksum(x_2[i, k] for i in range(25)) + gb.quicksum(z_2[j, k] for j in range(2)) == Refinement_Demand.loc[k, 'Demand'])

# Constraints - Transship Quantity limit
transship_limit = 2600
model_2.addConstr(gb.quicksum(y_2[i, j] for i in range(15) for j in range(2)) <= transship_limit)

model_2.optimize()

CUMCF_total = sum(x_2[i, k].X for i in range(25) for k in range(5))
UGI_total = sum(y_2[i, j].X for i in range(15) for j in range(2))

total_transport = CUMCF_total + UGI_total
transship_proportion = UGI_total / total_transport

print(transship_proportion)   


# (f)Based on the estimated distances from the production facilities to the refinement center in North America, here are the carbon cost:
# 
# 1. United States (i = 6-10): Closest - 0.3
# 2. Mexico (i = 11-15): Very Close - 0.5
# 3. Canada (i = 1-5): Very Close - 0.5
# 4. France (i = 21-25): Close - 0.8
# 5. China (i = 16-20): Fairly Far - 1
# 6. Germany (i = 36-40): Far - 1.5
# 7. Ukraine (i = 31-35): Far - 1.5
# 8. India (i = 26-30): Farthest - 2

def CUMCF_Cost_by_location(row):
    production_facility = row['ProductionFacility']
    base_cost = row['Cost']
    if 1 <= production_facility <= 5:  # Canada
        return base_cost + 0.5
    elif 6 <= production_facility <= 10:  # United States
        return base_cost + 0.3
    elif 11 <= production_facility <= 15:  # Mexico
        return base_cost + 0.5
    elif 16 <= production_facility <= 20:  # China
        return base_cost + 1
    elif 21 <= production_facility <= 25:  # France
        return base_cost + 0.8
    else:
        return base_cost
CUMCF_Cost['Cost'] = CUMCF_Cost.apply(CUMCF_Cost_by_location, axis=1)

def IUG_Cost_by_location(row):
    production_facility = row['ProductionFacility']
    base_cost = row['Cost']
    if 1 <= production_facility <= 5:  # India
        return base_cost + 2
    elif 6 <= production_facility <= 10:  # Ukraine
        return base_cost + 1.5
    elif 11 <= production_facility <= 15:  # Germany
        return base_cost + 1.5
    else:
        return base_cost
IUG_Cost['Cost'] = IUG_Cost.apply(IUG_Cost_by_location, axis=1)


# Minimize Transportation Cost
model_3 = gb.Model("Minimize Transportation Cost")

x_3 = model_3.addVars(25, 5, lb=0, vtype=GRB.CONTINUOUS, name="CUMCF")
y_3 = model_3.addVars(15, 2, lb=0, vtype=GRB.CONTINUOUS, name="IUG")
z_3 = model_3.addVars(2, 5, lb=0, vtype=GRB.CONTINUOUS, name="Distribution")

model_3.setObjective(
    gb.quicksum(CUMCF_Cost.set_index(['ProductionFacility', 'RefinementCenter']).loc[(i+1, k+1), 'Cost'] * x_3[i, k] for i in range(25) for k in range(5)) +
    gb.quicksum(IUG_Cost.set_index(['ProductionFacility', 'TransshipmentHub']).loc[(i+1, j+1), 'Cost'] * y_3[i, j] for i in range(15) for j in range(2)) +
    gb.quicksum(Distribution_Cost.set_index(['TransshipmentHub', 'RefinementCenter']).loc[(j+1, k+1), 'Cost'] * z_3[j, k] for j in range(2) for k in range(5)), 
    gb.GRB.MINIMIZE)

# Constraints - Capacity from CUMCF
for i in range(25):
    model_3.addConstr(gb.quicksum(x_3[i, k] for k in range(5)) <= CUMCF_Capacity.loc[i, 'Capacity'])
    
# Constraints - Capacity from IUG
for i in range(15): 
    model_3.addConstr(gb.quicksum(y_3[i, j] for j in range(2)) <= IUG_Capacity.loc[i, 'Capacity'])
    
# Constraints - Capacity from Transshipment Hub
for j in range(2):
    total_input = gb.quicksum(y_3[i, j] for i in range(15))
    total_output = gb.quicksum(z_3[j, k] for k in range(5))
    model_3.addConstr(total_input == total_output)
    model_3.addConstr(total_output <= Distribution_Capacity.loc[j, 'Capacity'])

# Constraints - Demand from Refinement Center
for k in range(5):
    model_3.addConstr(gb.quicksum(x_3[i, k] for i in range(25)) + gb.quicksum(z_3[j, k] for j in range(2)) == Refinement_Demand.loc[k, 'Demand'])

model_3.optimize()  


# Print Results
if model_3.status == gb.GRB.OPTIMAL:
    solution_x = model_3.getAttr('x', x_3)
    solution_y = model_3.getAttr('x', y_3)
    print("Optimal solution found:")
    for key, value in solution_x.items():
        print(f"CUMCF[{key}]: {value}")
    for key, value in solution_y.items():
        print(f"IUG[{key}]: {value}")
else:
    print("No optimal solution found.")




# Part 2

from gurobipy import GRB
import gurobipy as gb

# Create the optimization model
model = gb.Model("Tower Research")

# Create 2 classes of decision variables where each Python
B = model.addVars(6, lb=0, vtype=GRB.CONTINUOUS, name="Borrow")
w = model.addVars(4, lb=0, vtype=GRB.CONTINUOUS, name="Wealth")
R = model.addVars(4, lb=0, vtype=GRB.CONTINUOUS, name="Repay")

model.setObjective(R[3], GRB.MINIMIZE)

# Payback Constraints
model.addConstr(R[0] == 1.0175*B[0], "Period 2 Constraint")
model.addConstr(R[1] == 1.0225*B[1] + 1.0175*B[3], "Period 3 Constraint")
model.addConstr(R[2] == 1.0275*B[2] + 1.0225*B[4] + 1.0175*B[5], "Period 4 Constraint")
model.addConstr(R[3] == R[0]+R[1]+R[2], "Period Total Payback")

# Wealth Constraints
model.addConstr(w[0] == 140000 + 180000 - 300000 + B[0] + B[1] + B[2], "Period 1 Constraint")
model.addConstr(w[1] == w[0] + 260000 - 400000 + B[3] + B[4] - R[0], "Period 2 Constraint")
model.addConstr(w[2] == w[1] + 420000 - 350000 + B[5] - R[1], "Period 3 Constraint")
model.addConstr(w[3] == w[2] + 580000 - 200000 - R[2], "Period 4 Constraint")

# Other Constraints
model.addConstr(w[0] >= 25000, "constraint_1")
model.addConstr(w[1] >= 20000, "constraint_2")
model.addConstr(w[2] >= 35000, "constraint_3")
model.addConstr(w[3] >= 18000, "constraint_4")
model.addConstr(B[0] + B[1] + B[2] <= 250000, "constraint_5")
model.addConstr(B[3] + B[4] <= 150000, "constraint_6")
model.addConstr(B[5] <= 350000, "constraint_7")
model.addConstr(w[2] >= 0.65 * (w[0] + w[1]), "constraint_8")
model.optimize()
# Number of constraints in the model
print("Number of Constraints: ", model.numConstrs)
# The status of the model
print("Model Status: ", model.status)
# Value of the objective function
print("Total costs: ", model.objVal)
# Print the decision variables
print(model.printAttr('X'))

# June Constraint changed.
model.reset()
B = model.addVars(6, lb=0, vtype=GRB.CONTINUOUS, name="Borrow")
w = model.addVars(4, lb=0, vtype=GRB.CONTINUOUS, name="Wealth")
R = model.addVars(4, lb=0, vtype=GRB.CONTINUOUS, name="Repay")

model.setObjective(R[3], GRB.MINIMIZE)


# Payback Constraints
model.addConstr(R[0] == 1.0175*B[0], "Period 2 Constraint")
model.addConstr(R[1] == 1.0225*B[1] + 1.0175*B[3], "Period 3 Constraint")
model.addConstr(R[2] == 1.0275*B[2] + 1.0225*B[4] + 1.0175*B[5], "Period 4 Constraint")
model.addConstr(R[3] == R[0]+R[1]+R[2], "Period Total Payback")

# Wealth Constraints
model.addConstr(w[0] == 140000 + 180000 - 300000 + B[0] + B[1] + B[2], "Period 1 Constraint")
model.addConstr(w[1] == w[0] + 260000 - 400000 + B[3] + B[4] - R[0], "Period 2 Constraint")
model.addConstr(w[2] == w[1] + 420000 - 350000 + B[5] - R[1], "Period 3 Constraint")
model.addConstr(w[3] == w[2] + 580000 - 200000 - R[2], "Period 4 Constraint")

# Other Constraints
model.addConstr(w[0] >= 25000, "constraint_1")
model.addConstr(w[1] >= 27500, "constraint_2")
model.addConstr(w[2] >= 35000, "constraint_3")
model.addConstr(w[3] >= 18000, "constraint_4")
model.addConstr(B[0] + B[1] + B[2] <= 250000, "constraint_5")
model.addConstr(B[3] + B[4] <= 150000, "constraint_6")
model.addConstr(B[5] <= 350000, "constraint_7")
model.addConstr(w[2] >= 0.65 * (w[0] + w[1]), "constraint_8")
model.optimize()
# Number of constraints in the model
print("Number of Constraints: ", model.numConstrs)
# The status of the model
print("Model Status: ", model.status)
# Value of the objective function
print("Total costs: ", model.objVal)
# Print the decision variables
print(model.printAttr('X'))

# Dual Linear Programming
model.reset()
Y = model.addVars(8, lb=0, vtype=GRB.CONTINUOUS, name="Coeff")
# Redefine the objective function for the dual problem
Z = 5000*Y[0] + 140000*Y[1] + 85000*Y[2] - 312000*Y[3] - 250000*Y[4] - 150000*Y[5] - 350000*Y[6] - 15000*Y[7] 
model.setObjective(Z, GRB.MAXIMIZE)

# Redefine the dual constraints (assuming Y variables are already defined)
model.addConstr(Y[0] - 0.0175*Y[1] - 0.0175*Y[2] - 0.0175*Y[3] - Y[4] - 0.656125*Y[7] <= 1.0175, "Dual_Constraint_B0")
model.addConstr(Y[0] + Y[1] - 0.0225*Y[2] - 0.0225*Y[3] - Y[4] - 1.3325*Y[7] <= 1.0225, "Constraint_B1")
model.addConstr(Y[0] + Y[1] + Y[2] - 0.0275*Y[3] - Y[4] - 0.3*Y[7] <= 1.0275, "Constraint_B2")
model.addConstr(Y[1] - 0.0175*Y[2] - 0.0175*Y[3] - Y[5] - 0.6675*Y[7] <= 1.0175, "Constraint_B3")
model.addConstr(Y[1] + Y[2] - 0.0225*Y[3] - Y[5] + 0.35*Y[7] <= 1.0225, "Constraint_B4")
model.addConstr(Y[2] - 0.0175*Y[3] - Y[6] + Y[7] <= 1.0175, "Constraint_B5")

# Optimize the dual model
model.optimize()

# Output results
print("Number of Constraints in the Dual Model:", model.numConstrs)
print("Dual Model Status:", model.status)
print("Optimal Value of the Dual Objective Function:", model.objVal)
model.printAttr('X')




