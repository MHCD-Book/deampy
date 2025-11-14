from deampy.sensitivity_analysis import SensitivityAnalysis


def do_sa(dic_parameter_values, outputs):
    """
    :param dic_parameter_values: (dictionary) of parameter values with parameter names as the key
    :param outputs: (list) of output values (e.g. cost or QALY observations)
    """
    sa = SensitivityAnalysis(dic_parameter_values=dic_parameter_values,
                             output_values=outputs)

    print('')
    sa.print_corr(corr='r')     # Pearson's
    sa.print_corr(corr='rho')   # Spearman's
    sa.print_corr(corr='p')     # partial correlation
    sa.print_corr(corr='pr')    # partial rank correlation

# set up parameter values in a dictionary
dic_parameter_values = {
    'par1': [1, 4, 3, 7, 1],
    'par2': [5, 7, 2, 8, 23],
    'par3': [5, 6, 7, 8, 9]
}
# set up the output values in a list
outputs = [7, 1, 2, 15, 5]
# call the sensitivity analysis function
do_sa(dic_parameter_values=dic_parameter_values, outputs=outputs)



# set up parameter values in a dictionary
dic_parameter_values = {
    'par1': [1, 1, 1, 1, 1],
    'par2': [5, 7, 2, 8, 23],
}
# set up the output values in a list
outputs = [7, 1, 2, 15, 5]
# call the sensitivity analysis function
do_sa(dic_parameter_values=dic_parameter_values, outputs=outputs)
