from numpy.random import RandomState

from deampy.models import MortalityModelByAge

model = MortalityModelByAge(
    age_breaks=[0, 5, 10],  # age groups
    mortality_rates=[0.1, 0.2, 0.3],  # mortality rates for each age group
    age_delta=5  # age interval to ensure that age breaks are equally spaced
)

rng = RandomState(seed=0)

# Sample time to death for different ages
print(model.sample_time_to_death(current_age=0, rng=rng))
print(model.sample_time_to_death(current_age=5, rng=rng))
print(model.sample_time_to_death(current_age=5.1, rng=rng))
print(model.sample_time_to_death(current_age=12, rng=rng))