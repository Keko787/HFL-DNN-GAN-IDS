from ucimlrepo import fetch_ucirepo

def load_RTIoT2022():
    # fetch dataset
    rt_iot2022 = fetch_ucirepo(id=942)

    # data (as pandas dataframes)
    X = rt_iot2022.data.features
    y = rt_iot2022.data.targets

    # metadata
    print(rt_iot2022.metadata)

    # variable information
    print(rt_iot2022.variables)

    return X, y
