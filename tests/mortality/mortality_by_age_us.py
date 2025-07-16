import statistics as stats

from numpy.random import RandomState

from deampy.in_out_functions import read_csv_cols_to_dictionary
from deampy.models import MortalityModelByAge

dict_mortality_rates = read_csv_cols_to_dictionary(
    file_name='US Life Table 2023.csv',
    if_convert_float=True
)

model = MortalityModelByAge(
    age_breaks=range(0, 101),  # age groups
    mortality_rates=dict_mortality_rates['Rate'],  # mortality rates for each age group
    age_delta=1  # age interval to ensure that age breaks are equally spaced
)

rng = RandomState(seed=0)

# Sample time to death for different ages
a0 = [model.sample_time_to_death(current_age=0, rng=rng) for i in range(1000)]
a5 = [model.sample_time_to_death(current_age=5, rng=rng) for i in range(1000)]
a35= [model.sample_time_to_death(current_age=35, rng=rng) for i in range(1000)]
a99= [model.sample_time_to_death(current_age=99, rng=rng) for i in range(1000)]
a100= [model.sample_time_to_death(current_age=100, rng=rng) for i in range(1000)]

print('0 years old:', stats.mean(a0))
print('5 years old:', stats.mean(a5))
print('35 years old:', stats.mean(a35))
print('99 years old:', stats.mean(a99))
print('100 years old:', stats.mean(a100))