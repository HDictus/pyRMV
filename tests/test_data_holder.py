import pandas as pd
from analysis_neuro import df_holder

def test_df_holder_differs_by_df():
    df1 = pd.DataFrame({'a': range(1000)})
    df2 = pd.DataFrame({'a': range(200)})
    df3 = pd.DataFrame({'1234': range(4000)})
    hold1 = df_holder.create_dataframe_holder(df1)
    hold2 = df_holder.create_dataframe_holder(df2)
    hold3 = df_holder.create_dataframe_holder(df3)
    pd.testing.assert_frame_equal(hold1.df, df1)
    assert hold1 != hold2
    assert hold2 != hold3
    assert hold1 != hold3
    assert hold1 == df_holder.create_dataframe_holder(df1)
    assert hold1 == df_holder.create_dataframe_holder(df1.copy())