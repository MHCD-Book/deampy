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
print(model.sample_time_to_death(current_age=0, rng=rng))
print(model.sample_time_to_death(current_age=5, rng=rng))
print(model.sample_time_to_death(current_age=5.1, rng=rng))
print(model.sample_time_to_death(current_age=12, rng=rng))