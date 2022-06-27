# Predator-Prey Swarm Silumator

## Simulator

### Running a simulation

Use the script 'simulate.py' to generate simulations.

Use:

```
  python3 simulate.py <name_of_config_file>.json --sim_name <name_of_the_simulation> --save
```

The simulation results can be found in the folder 'output/simulation_results'.

### Running an experiment

Use the script 'simulate.py' to run experiments.

Use:
```
  python3 simulate.py <name_of_experiment_config_file>.json --sim_name <name_of_the_experiment> --experiment --plot
```

### Visualizing a simulation

You can visualize simulations using the associated web-based visualization tool.

Use the following command to start the visualization tool:

```
./start_visualization_server.[sh|bat]
```

You can now navigate to 'localhost:5000', to access the tool.

TO DO: A more detailed description will be added.