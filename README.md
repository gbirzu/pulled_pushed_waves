# pulled_pushed_waves
LICENSE: The code located here is provided as-is, with no warranty, etc. (It's under GPL v2.) 
AUTHOR: Gabriel Birzu
CONTACT: gbirzu@gmail.com 

DESCRIPTION: This depository contains the tools and data to generate all the figures in Birzu, G, Hallatschek, O, and Korolev, KS, "Fluctuations uncover a distinct class of traveling waves", PNAS (2018). The averaged data used in the paper can be found in the data/ folder. The data for the individual Monte Carlo runs is not included. The source code for the program used perform the Montel Carlso simulations in the main text is included. The data from these runs was analyzed as described in the paper to generate the averages from the data/ folder. Additionally, Monte Carlso runs were used to generate Fig 3 from the main text. These are included in data/stochastic_simulations Scripts: 
    - eff_pop_coopeartive.cpp contains the source code to run stochastic simulations of a range expansions with a per capita growth rate of the form r(n) = r_0(1 - n/N)(1 + B*n/N).
    - paper_plots_final.py plots the figures from the main text
    - si_plots.py plots the figures from Supplemental Information
    - analytical_plots.py contains tools to make plots that use only analytical results from the paper
    - data_analysis_tools.py contains tools to analyze simulation data

REQUIREMENTS: Python (v2.7) with the NumPy (v1.11) library, and a C++ compiler that supports the C++11 standard (e.g., GNU's g++). 

INSTALLATION: Compile "eff_pop_cooperative.cpp" source file and name the program "pop_program_cooperative" 

USAGE: (1) Simulations can be performed by calling the program from the command line with:

./pop_program_cooperative 

This will run the program with the default parameters. Parameters can be changed using the input flags below.

-g zero density growth rate. Same as r_0 in Eq. (3) of the main text 
-m migration probability per generation
-K carrying capacity. Must be integer. Same as N in Eq. (3) of the main text
-B cooperativity. Same as B in Eq. (3) of the main text
-r run number. Used to keep track of independent runs and create unique names for the output files
-n deterministic flag. Any non zero value assigned to this flag will set gamma_n = 0
-w flag for saving profiles at intermediate times. Any non zero value assigned to this flag will cause the program to save profiles when the time elapsed is a power of 2

E.g. 

./pop_program_cooperative -g 0.1 -m 0.2 -K 1000 -B 2.5

would run a simulation with the growth function from the main text of the article (Eq. 3), using the parameters: r_0 = 0.1, m = 0.2, N = 1000, B = 2.5
