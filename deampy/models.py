import deampy.random_variates as RVGs
from deampy.data_structure import DataFrameOfObjects, OneDimDataFrame


class MortalityModelByAgeSex:
    """
    example:
        age,   sex,      mortality rate
        0,     0,        0.1,
        0,     1,        0.11,
        5,     0,        0.2,
        5,     1,        0.21,
        10,    0,        0.3
        10,    1,        0.31

    This class assumes that the first column contains age groups and the last column contains mortality rates
    """

    def __init__(self, mortality_rates, group_mins, group_maxs, group_delta, age_min, age_delta):
        """
        :param mortality_rates: (list of list) the table above
        :param group_mins: list of minimum value of x (in example above: [0, 0])
        :param group_maxs: list of maximum value of x (in example above: [10, 1])
        :param group_delta: list of interval between break points of x
                    if set to 'int', x is treated as categorical variable
                    (in example above: [5, 'int'])
        :param age_min:
        :param age_delta:
        """

        self.df = DataFrameOfObjects(list_x_min=group_mins,
                                     list_x_max=group_maxs,
                                     list_x_delta=group_delta)

        self.ageMin = age_min

        for df_row in self.df.get_rows():
            rates = []
            for row in mortality_rates:
                if df_row[:-1] == row[1:-1]:
                    rates.append(row[-1])
            self.df.set_obj(x_value=df_row[0:-1],
                            obj=RVGs.NonHomogeneousExponential(rates=rates, delta_t=age_delta))

    def sample_time_to_death(self, group, age, rng):

        if age < self.ageMin:
            raise ValueError('Current age cannot be smaller than the minimum age.')

        return self.df.get_obj(group).sample(rng=rng, arg=age)



class MortalityModelByAge:
    """
    example:
        age,   mortality rate
        0,     0.1,
        5,     0.2,
        10,    0.3
    """

    def __init__(self, age_breaks, mortality_rates, age_delta):
        """
        :param : (list of list) the table above
        :param age_delta:
        """

        if len(age_breaks) != len(mortality_rates):
            raise ValueError('The number of age breaks should be equal to the number of mortality rates.')

        # create the list of nonhomogenous exponential distributions
        y_objects = []
        for i, v in enumerate(mortality_rates):
            if v <= 0:
                raise ValueError('All y_values (rates of exponential distributions) should be greater than 0.')
            y_objects.append(RVGs.NonHomogeneousExponential(
                rates= mortality_rates[i:]
            ))

        self.ageMin = age_breaks[0]

        self.df = OneDimDataFrame(
            y_objects=y_objects,
            x_min=self.ageMin,
            x_max=age_breaks[-1],
            x_delta=age_delta)

    def sample_time_to_death(self, current_age, rng):

        if current_age < self.ageMin:
            raise ValueError('Current age cannot be smaller than the minimum age.')

        return self.df.get_obj(current_age).sample(rng=rng)
