# How to use these files

In order to run the python scripts, it's necessary to have MPCTools available here: https://bitbucket.org/rawlings-group/mpc-tools-casadi/src/master/

In bare_sim, you will find these files:
1) basic_sim.py - this will run a single MPC simulation and animate the results over a double gyre vector field
2) sim_ftle.py - this will run an MPC simulation and animate results over an FTLE field. *Note: you must first create an FTLE data file using sim_ftle.py before being able to plot over an FTLE field. This needs to be done once*
3) ftle_calc.py - creates a .npz file of FTLE field data
4) mpc_hist_lzy.py - this was used to generate the histogram plots (heading angle, x/y components, energy spent) in our paper
