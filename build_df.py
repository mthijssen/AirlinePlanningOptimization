"""
This is a file where the dataframe from exercise 1A is made. Useful to clean up code. 
"""


import pandas as pd

def load_df():
    airport_data = pd.read_excel('DemandGroup12.xlsx', index_col=1, header=2)
    airport_data = airport_data.head(6).T

    demand_data = pd.read_excel('DemandGroup12.xlsx', index_col=1, header=11)

    pop = pd.read_excel('pop.xlsx', header=2)

    airport_data = airport_data.rename(columns={airport_data.columns[0]: "City"})

    df = pd.merge(airport_data, pop, on="City", how="left")
    df = df.drop(columns=["Unnamed: 3"], index=0).reset_index(drop=True)
    df = df.rename(columns={
        2021: "Population_2021",
        2024: "Population_2024",
        "2021.1": "GDP_2021",
        "2024.1": "GDP_2024"
    })

    cols_to_convert = ["Latitude (deg)", "Longitude (deg)", "Runway (m)", "Available slots"]
    df[cols_to_convert] = df[cols_to_convert].apply(
        pd.to_numeric, errors="coerce"
    )

    pop_growth = (df["Population_2024"] / df["Population_2021"]) ** (1/3)
    df["Population_2026"] = df["Population_2024"] * (pop_growth ** 2)

    gdp_growth = (df["GDP_2024"] / df["GDP_2021"]) ** (1/3)
    df["GDP_2026"] = df["GDP_2024"] * (gdp_growth ** 2)

    return df, demand_data