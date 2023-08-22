import pandas as pd


# TODO: create test cases

df_holders = {}
holders_by_name = {}


class _DFHolder:
    """A df holder acts as a pointer to a dataframe.

    It can be used as an element within a dataframe without breaking .groupby and .unique
    
    The .df property is a copy of the relevant dataframe.
    _DFHolder objects which point to identical dataframes are the __same__ object.
    """

    def __init__(self, df):
        self._df = df.copy()

    def __hash__(self):
        return hash(tuple((col, tuple(self.df[col])) for col in self._df.columns))

    @property
    def df(self):
        return self._df.copy()

    def __str__(self):
        return f"dataframe holder: {self._df}"


def create_dataframe_holder(df):
    new_holder = _DFHolder(df)
    if hash(new_holder) in df_holders:
        return df_holders[hash(new_holder)]
    df_holders[hash(new_holder)] = new_holder
    return new_holder
