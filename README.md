# How to use these files

Paper available here: https://arxiv.org/abs/2103.10556

In order to run the python scripts, it's necessary to have MPCTools available here: https://bitbucket.org/rawlings-group/mpc-tools-casadi/src/master/

A) In bare_sim, you will find these files:
  1) basic_sim.py - this will run a single MPC simulation and animate the results over a double gyre vector field. *Use this to get started and for sanity checks*
  2) sim_ftle.py - this will run an MPC simulation and animate results over an FTLE field. *Note: you must first create an FTLE data file using     ftle_calc.py before being able to plot over an FTLE field. This needs to be done once*
  3) ftle_calc.py - creates a .npz file of FTLE field data
  4) mpc_hist_lzy.py - this was used to generate the histogram plots (heading angle, x/y components, energy spent) in our paper
  5) abc_ctrl.py - this will run MPC on the ABC (Arnold-Beltrami-Childress) flow field
  6) abc_lcs.py - this is to create LCS data for plotting in abc_ctrl
  7) remote_read.py - downloads a Gulf of Mexico flow field off the internet (you should save the npz file so it can imported in gulf_ctrl.py and gilf_lcs.py
  8) gulf_ctrl.py and gulf_lcs.py - similar to the ABC flow files, but for the Gulf of Mexico test case

B) In rq_sweeps, you will find: (This was used to generate the appendix figures and Figure 2 in our paper)
  1) mpc_bundle_rq.py - This runs multiple simulations (default 100) and plots the full trajectories for each simulation as the cost function values change in an animation. There also exists an option to save data for later reading. 
  2) read_slide_rq.py - If data was saved, this python file can be used to re-read and visualize the results

C) In multi_sweep, you will find: (Figure 8 was generated using the files here)
  1) double_sweep.py - Generates data by running 100 simulations (default, can be changed) over different time horizons (default, can be changed) and saves it to a .npz file
  2) abc_dsweep.py - similar to double_sweep.py, but for the ABC flow field
  3) read_d_sweep.py - This python script can be used to visualize the .npz file generated by double_sweep.py
