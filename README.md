# OC-VTQ: Optimal Control of Vectored-Thrust Quadcopter 
An MIT final project for 16.32: Optimal Control and Estimation by Marcos Espitia-Alvarez

## Setup
Before anything, please navigate to a directory where
you would like to import the repository
### Step 1: Clone the Repository
```
git clone git@github.com:mespitiaalvarez/OC-VTQ.git
```

### Step 2: Navigate to py_sim inside OC-VTQ
```
cd ./OC-VTQ/py_sim
```

### Step 3: Create a python3 virual environment
```
python3 -m venv .venv
```

### Step 4: Activate the environment
Windows
```
.venv\Scripts\activate
```
or (Linux)
```
source .venv/bin/activate
```

### Step 5: Pip install packages 
These are listed in requirements, again ensure you have activated the environment
```
pip install -r requirements.txt
```

### Step 6: Guide to files in py_sim
There are many files in py_sim, each with a purpose
```
dynamics.py --> full Casadi Symbolic Definition of model

integrator.py --> Contains the RK45 modified definition as a Casadi function

optimization_problem.py --> Where optization problem and costs are defined. Solve takes in a cost function and other required states to solve opt problem. 

optimal_control.py --> Where inital and target states are defined. Also where ts and N live. This file imports from optimization_problem and runs solve function

utils.py --> Various utilities and definitions written either for Casadi or numpy

visualize.py --> Takes in csv number from data folder (created when opt_control run) and graphs out the 5 plots

nmpc.py --> An incomplete NMPC simulation where the sim state is also forward integrated 
```

### Step 7: Play with Optimal Control
In the optimal control file is where the solve function is actually called. Change the values of x_target to have different ref states for the optimization problem. Refer to dynamics.py or the paper to know which indicies correspond to which elements. Important one to know x[6:10] are the quaterion values being a,b,c,d. When you want to run, remember to be in the py_sim directory
```
python3 optimal_control.py
```
You should see a folder called data appear with 3 CSVs (x_opt, u_opt, x_ref)

### Step 8: Plot
Optimal_Control standarizes csv name outputs, they will be numbered. The naming structure is assumed in visualize. At the top of the main function just change the string called number to whatevr number (AS A STRING).
Example:
```
def main():
    # === Load NMPC Results === #
    use('TkAgg')  # Was using Linux Subsystem
    number = "3" <<<<<CHANGE THIS TO "4" or "5" or "1"

    str_xopt = "results/x_opt_" + number + ".csv"
    str_uopt = "results/u_opt_" + number + ".csv"
```