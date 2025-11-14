from numpy.random import rand

from deampy.sensitivity_analysis import SensitivityAnalysis

# sample from parameters
param1 = rand(100) * 20
param2 = rand(100) * -1
# sample from output as a function of parameters
output = param1 + 2*param2 + rand(100) * 10

# set up parameter values in a dictionary
dic_par_values = {'Par1': param1, 'Par2': param2}

# do sensitivity analysis
sa = SensitivityAnalysis(dic_parameter_values=dic_par_values, output_values=output)

sa.print_corr(corr='r')     # Pearson's
sa.print_corr(corr='rho')   # Spearman's
sa.print_corr(corr='p')     # partial correlation
sa.print_corr(corr='pr')    # partial rank correlation

# save results to a csv file
sa.export_to_csv(corrs=['r', 'rho', 'p', 'pr'], decimal=4)
