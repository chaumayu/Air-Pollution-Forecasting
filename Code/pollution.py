import pandas as pd

def read(path):
    # Load data
    df = pd.read_csv(path)
    print df.head()
    # Drop No column
    df.drop('No', axis=1, inplace=True)
    print df.head()
    # Drop first 24 from pressure column
    df = df[24:].reset_index()
    print df.head()

    return df

def year_month_day_hour(df):
    # year month day hour to datetime
    df['date'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
    # Drop year month day hour column
    df.drop(['year','month','day','hour'], axis=1, inplace=True)
    # Reorder columns
    df = df[['date', 'pm2.5', 'DEWP', 'TEMP', 'PRES', 'cbwd', 'Iws', 'Is', 'Ir']]
    print df.head()

    return df

def rename_column_name(df):
    df.columns = ['date', 'pollution', 'dew', 'temp', 'pressure', 'wind_direction', 'wind_speed', 'hrs_snow', 'hrs_rain']
    print df.head()

    return df

def check_na(df):
    print df.isnull().any()
    df['pollution'].fillna(0, inplace=True)

    return df

def main():
    raw_data = 'Data/PRSA_data_2010.1.1-2014.12.31.csv'
    data = read(raw_data)
    data = year_month_day_hour(data)
    data = rename_column_name(data)
    data = check_na(data)

    print 'Saving to csv'
    data.to_csv('Data/pollution.csv', index=False)

if __name__ == '__main__':
    main()
