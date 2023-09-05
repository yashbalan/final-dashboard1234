import pandas as pd
import streamlit as st
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta
import seaborn as sns
import re
import matplotlib.pyplot as plt
from PIL import Image
import folium
from streamlit_folium import folium_static
from folium import plugins


df1 = pd.DataFrame(pd.read_csv(
    'May Data/Ops_Session_Data.csv', encoding='latin1'))
df2 = pd.DataFrame(pd.read_csv(
    'May Data/past_bookings_May23.csv', encoding='latin1'))

df3 = pd.DataFrame(pd.read_csv(
    'May Data/possible_subscribers_May23.csv', encoding='latin1'))

def load_monthly_data(file_path):
    return pd.read_csv(file_path, encoding='latin1')

# Load June data
june_file_path = 'Roundtable Data/June Roundtable data.xlsx - Round table.csv'
df_june = pd.DataFrame(load_monthly_data(june_file_path))

# Load July data
july_file_path = 'Roundtable Data/Roundtable july1 (2).csv'
df_july = pd.DataFrame(load_monthly_data(july_file_path))

# Concatenate June and July data for df_month
df_month = pd.concat([df_june, df_july], ignore_index=True)

# Load June vehicles data
june_vehicles_file_path = 'KM Data/Vehicles-Daily-Report-01-Jun-2023-12-00-AM-to-30-Jun-2023-11-59-PM.xlsx - Vehicle Daily Report.csv'
df_vehicles_june = pd.DataFrame(load_monthly_data(june_vehicles_file_path))

# Load July vehicles data
july_vehicles_file_path = 'KM Data/Vehicles-Daily-Report-01-Jul-2023-12-00-AM-to-31-Jul-2023-11-59-PM.csv'
df_vehicles_july = pd.DataFrame(load_monthly_data(july_vehicles_file_path))

# Concatenate June and July data for df_vehicles_month
df_vehicles_month = pd.concat([df_vehicles_june, df_vehicles_july], ignore_index=True)


def load_rank_data(file_path):
    return pd.read_csv(file_path, encoding='latin1')

# Load June data
june_rank_file_path = r"Rank Data/June Rank Data.csv"
df_rank_june = pd.DataFrame(load_rank_data(june_rank_file_path))

# Load July data
july_rank_file_path = r"Rank Data/July Rank.csv"
df_rank_july = pd.DataFrame(load_rank_data(july_rank_file_path))

# Concatenate June and July data for the rank DataFrame
df_rank = pd.concat([df_rank_june, df_rank_july], ignore_index=True)

df_month.rename(
    columns={'Reach date ': 'Reach date'}, inplace=True)
df2["Customer Name"] = df2["firstName"].str.cat(df2["lastName"], sep=" ")
df_month["Customer Name"] = df_month["firstName"].str.cat(
    df_month["lastName"], sep=" ")
df_month['E-pod No.'] = df_month['E-pod No.'].str.upper()

df1 = df1.dropna(subset=["uid"])


def subtract_12_from_pm_time(time_str):
    time_pattern = r"(\d{1,2}):(\d{2})\s?(AM|PM|pm|am)"
    match = re.match(time_pattern, time_str)

    if match:
        hour = int(match.group(1))
        minute = int(match.group(2))
        am_pm = match.group(3)

        if am_pm.lower() == "pm" and hour < 12:
            hour += 12

        return "{:02d}:{:02d}".format(hour, minute)

    return time_str


df_month = df_month[pd.notna(df_month['fromTime'])]
df_month = df_month[df_month['Reach Time'] != 'Cancel']
df_month = df_month[df_month['Reach Time'] != 'CANCEL']

df1['Booking Session time'] = df1['Booking Session time'].apply(
    subtract_12_from_pm_time)
# df_month['Booked time'] = df_month['Booked time'].apply(
#     subtract_12_from_pm_time)
df2['updated'] = pd.to_datetime(df2['updated'], format='%d/%m/%Y, %H:%M:%S')
df2['fromTime'] = pd.to_datetime(df2['fromTime'], format='%Y-%m-%dT%H:%M')

df_month['updated'] = pd.to_datetime(
    df_month['updated'])
df_month['fromTime'] = pd.to_datetime(
    df_month['fromTime'], format='mixed')
df_month['Reach Time'] = pd.to_datetime(
    df_month['Reach Time'], format='mixed')


df1["KM Travelled for Session"] = df1["KM Travelled for Session"].str.replace(
    r'[a-zA-Z]', '', regex=True)
df1['EPOD Name'] = df1['EPOD Name'].str.extract(
    r'^(.*?)\s+\(.*\)$')[0]
df1['EPOD Name'] = df1['EPOD Name'].fillna('EPOD-006')

df2['time_diff'] = df2['fromTime'] - df2['updated']
df2['time_diff_hours'] = df2['time_diff'] / timedelta(hours=1)

# Initialize the 'cancelledPenalty' column with zeros
df2['cancelledPenalty'] = 0

# Set 'cancelledPenalty' to 1 for rows where 'canceled' is True and 'time_diff_hours' is less than 2
df2.loc[(df2['canceled'] == True) & (df2['time_diff_hours'] < 2), 'cancelledPenalty'] = 1

# Optionally, you can drop the 'time_diff' column if you don't need it anymore
df2.drop(columns=['time_diff'], inplace=True)

df_month['time_diff'] = df_month['fromTime'] - df_month['updated']
df_month['time_diff_hours'] = df_month['time_diff'] / timedelta(hours=1)

# Initialize the 'cancelledPenalty' column with zeros
df_month['cancelledPenalty'] = 0

# Set 'cancelledPenalty' to 1 for rows where 'canceled' is True and 'time_diff_hours' is less than 2
df_month.loc[(df_month['canceled'] == True) & (df_month['time_diff_hours'] < 2), 'cancelledPenalty'] = 1

# Optionally, you can drop the 'time_diff' column if you don't need it anymore
df_month.drop(columns=['time_diff'], inplace=True)

df1.drop(columns=['Customer Location City'], inplace=True)
df2.rename(columns={'location.state': 'Customer Location City'}, inplace=True)
df2['Customer Location City'].replace(
    {'Haryana': 'Gurugram', 'Uttar Pradesh': 'Noida'}, inplace=True)
df_month.rename(
    columns={'location.state': 'Customer Location City'}, inplace=True)
df_month['Customer Location City'].replace(
    {'Haryana': 'Gurugram', 'Uttar Pradesh': 'Noida'}, inplace=True)
df1['Actual Date'] = pd.to_datetime(df1['Actual Date'])

df1['KM Travelled for Session'] = df1['KM Travelled for Session'].replace(
    '', 0)
df1['KM Travelled for Session'].fillna(0, inplace=True)

df1['KM Travelled for Session'] = pd.to_numeric(
    df1['KM Travelled for Session'], errors='coerce')

df1['KM Travelled for Session'] = df1['KM Travelled for Session'].astype(int)
grouped = df1.groupby(['EPOD Name', df1['Actual Date'].dt.strftime(
    '%d/%m')])['KM Travelled for Session'].sum().reset_index()


pivot_df = grouped.pivot_table(index='EPOD Name', columns='Actual Date',
                               values='KM Travelled for Session', fill_value=0).reset_index()
pivot_df.columns.name = None
pivot_df.columns = [col.strftime('%d/%m') if isinstance(
    col, pd.Timestamp) else col.replace(' ', '_') for col in pivot_df.columns]

pivot_df = pivot_df.rename(columns={'EPOD_Name': 'Name'})
pivot_df['Name'] = pivot_df['Name'].str.replace('-', '').str.replace(' ', '')

df_vehicles_month['Name'] = df_vehicles_month['Name'].str.replace(
    '-', '').str.replace(' ', '')


def assign_city(row):
    if row['Name'] in ['EPOD001', 'EPOD004', 'EPOD002', 'EPOD003', 'EPOD008']:
        return 'Noida'
    elif row['Name'] in ['EPOD006', 'EPOD007']:
        return 'Delhi'
    elif row['Name'] in ['EPOD005', 'EPOD009', 'EPOD010', 'EPOD011', 'EPOD012']:
        return 'Gurugram'
    else:
        return 'Unknown'


pivot_df['Customer Location City'] = pivot_df.apply(assign_city, axis=1)
df_vehicles_month['Customer Location City'] = df_vehicles_month.apply(
    assign_city, axis=1)


def calculate_t_minus_15(row):
    booking_time_str = row['Booking Session time']
    arrival_time_str = row['E-pod Arrival Time @ Session location']

    booking_time = datetime.strptime(booking_time_str, "%H:%M")

    arrival_time = datetime.strptime(arrival_time_str, "%I:%M:%S %p")

    if arrival_time_str.endswith("PM") and booking_time.hour < 12:
        booking_time = booking_time + timedelta(hours=12)

    time_diff = booking_time - arrival_time

    if time_diff >= timedelta(minutes=15):
        return 1
    elif time_diff < timedelta(seconds=0):
        return 2
    else:
        return 0


df_month['t-15_kpi'] = np.nan


def check_booking_time(df):
    for index, row in df.iterrows():
        booking_time_str = row['fromTime']
        arrival_time_str = row['Reach Time']

        booking_time = pd.to_datetime(booking_time_str)
        arrival_time = pd.to_datetime(arrival_time_str)

        time_diff = booking_time - arrival_time

        if time_diff >= timedelta(minutes=15):
            df.loc[index, 't-15_kpi'] = 1
        elif time_diff < timedelta(seconds=0):
            df.loc[index, 't-15_kpi'] = 2
        else:
            df.loc[index, 't-15_kpi'] = 0

    return df


df1['t-15_kpi'] = df1.apply(calculate_t_minus_15, axis=1)

# Calculate Duration in df2
def calculate_duration_df2(row):
    start_time = pd.to_datetime(row['optChargeStartTime'], dayfirst=False, errors='coerce')
    end_time = pd.to_datetime(row['optChargeEndTime'], dayfirst=False, errors='coerce')

    # Check if either start_time or end_time is NaT (Not a Time) or invalid format
    if pd.isna(start_time) or pd.isna(end_time) or start_time >= end_time:
        return None

    return abs((end_time - start_time).total_seconds() / 60)

# Add 'Duration' column to df2
df2['Duration'] = df2.apply(calculate_duration_df2, axis=1)

# Merge df2 and df1 based on 'uid'
merged_df = pd.merge(df2, df1, on="uid", how="left")

# Calculate Duration in df_month after merging
def calculate_duration_df_month(row):
    start_time = pd.to_datetime(row['optChargeStartTime'], dayfirst=False, errors='coerce')
    end_time = pd.to_datetime(row['optChargeEndTime'], dayfirst=False, errors='coerce')

    # Check if either start_time or end_time is NaT (Not a Time) or invalid format
    if pd.isna(start_time) or pd.isna(end_time) or start_time >= end_time:
        return None

    return abs((end_time - start_time).total_seconds() / 60)

# Add 'Duration' column to df_month after merging
df_month['Duration'] = df_month.apply(calculate_duration_df_month, axis=1)


df_month = check_booking_time(df_month)


df3.set_index('uid', inplace=True)


merged_df = merged_df.join(df3['type'], on='location.user_id_x')
df_month = df_month.join(df3['type'], on='location.user_id')
df_month = df_month.rename(columns={'E-pod No.': 'Number'})
df_month['Number'] = df_month['Number'].astype(str)
df_month = df_month.merge(
    df_vehicles_month[['Number', 'Name']], on='Number', how='left')



df_month = df_month.rename(
    columns={"Name": "EPOD Name", "Full name": "Actual OPERATOR NAME", "optBatteryBeforeChrg": "Actual SoC_Start", "optBatteryAfterChrg": "Actual Soc_End", "KWH Charged": "KWH Pumped Per Session", "Booked time": "Booking Session time"})
#df_month['EPOD Name'] = df_month['EPOD Name'].fillna('EPOD012')

# Concatenate "Reach Date" and "Reach date" columns into "Actual Date"
df_month['Actual Date'] = df_month['Reach Date'].fillna(df_month['Reach date'])

# Drop the individual "Reach Date" and "Reach date" columns
df_month.drop(columns=['Reach Date', 'Reach date'], inplace=True)

df_month['E-pod Arrival Time @ Session location'] = df_month['Reach time'].fillna(df_month['Arrival Time'])

# Drop rows without a 'uid'
df_month.dropna(subset=['uid'], inplace=True)

grouped_df = merged_df.groupby("uid").agg(
    {"Actual SoC_Start": "min", "Actual Soc_End": "max"}).reset_index()


grouped_df = grouped_df.rename(
    columns={"Actual SoC_Start": "Actual SoC_Start", "Actual Soc_End": "Actual Soc_End"})

merged_df = pd.merge(merged_df, grouped_df, on="uid", how="left")

merged_df = merged_df.drop(["Actual SoC_Start_x", "Actual Soc_End_x"], axis=1)

merged_df = merged_df.rename(columns={"location.lat_y": "location.lat", "location.long_y": "location.long", "optChargeStartTime_x": "optChargeStartTime", "optChargeEndTime_x": "optChargeEndTime",
                             "Actual SoC_Start_y": "Actual SoC_Start", "Actual Soc_End_y": "Actual Soc_End", "Customer Name_x": "Customer Name", "canceled_x": "canceled"})

merged_df = merged_df.drop_duplicates(subset="uid", keep="first")

merged_df = merged_df.reset_index(drop=True)


# Now, we can subtract 'optChargeStartTime' from 'optChargeEndTime' and update the 'Duration' column in merged_df
merged_df['Duration'] = (pd.to_datetime(merged_df['optChargeEndTime'], dayfirst=False) - pd.to_datetime(merged_df['optChargeStartTime'], dayfirst=False)).dt.total_seconds() / 60

# Filter out rows with duration less than 0 and greater than 300
#merged_df = merged_df[(merged_df['Duration'] >= 0) & (merged_df['Duration'] <= 300)]


df_month['Day'] = pd.to_datetime(df_month['Booked date'], format='mixed').dt.day_name()

merged_df['Day'] = pd.to_datetime(merged_df['Actual Date']).dt.day_name()


cities = ["Gurugram", "Delhi", "Noida"]


merged_df["Actual SoC_Start"] = merged_df["Actual SoC_Start"].str.rstrip("%")
merged_df["Actual Soc_End"] = merged_df["Actual Soc_End"].str.rstrip("%")

#df_month.to_csv("samlldata.csv", index=False)
requiredColumns = ['uid', 'Actual Date', 'Customer Name', 'EPOD Name', 'Actual OPERATOR NAME', 'Duration', 'optChargeStartTime', 'optChargeEndTime', 'Day',
                   'E-pod Arrival Time @ Session location', 'Actual SoC_Start', 'Actual Soc_End', 'Booking Session time', 'Customer Location City', 'canceled', 'cancelledPenalty', 't-15_kpi', 'type', 'KWH Pumped Per Session', 'location.lat', 'location.long']

df_month = df_month[requiredColumns]
merged_df = merged_df[requiredColumns]

merged_df.to_csv('merged.csv')
dfs = [merged_df, df_month]
merged_df = pd.concat(dfs, ignore_index=True)

merged_df = merged_df[merged_df['Customer Location City'].isin(cities)]
#merged_df.to_csv(r"C:\Users\DELL\Downloads\finalstream\finalstream\iopdf29.csv")

df = merged_df


st.set_page_config(page_title="Hopcharge Dashboard",
                   page_icon=":bar_chart:", layout="wide")

st.markdown(
    """
            <style>
                .appview-container .main .block-container {{
                    padding-top: {padding_top}rem;
                    padding-bottom: {padding_bottom}rem;
                    }}

            </style>""".format(
        padding_top=1, padding_bottom=1
    ),
    unsafe_allow_html=True,
)
st.markdown(
    """
    <style>
    body {
        overflow-x: hidden;
        overflow-yr: hidden;
        color:black;
    }
    </style>
    """,
    unsafe_allow_html=True
)


df_vehicles_month = df_vehicles_month.drop(columns=["Number", "Year", "Make", "Model",
                                                  "Fuel Type", "Driver Name", "Driver Number", "Total"], axis=1)


def convert_to_datetime_with_current_year(date_str):
    date_str = date_str.strip()
    current_year = datetime.now().year
    date_str_with_year = date_str + "/" + str(current_year)

    datetime_obj = pd.to_datetime(date_str_with_year, format='%d/%m/%Y')
    return datetime_obj
    # except ValueError as e:
    #     print("Error:", e)
    #     return None


def convert_vehicle_data(df):

    id_vars = ['Name', 'Customer Location City']
    melted_df = pd.melt(df, id_vars=id_vars,
                        var_name='date', value_name='KM Travelled for Session')
    melted_df = melted_df.rename(
        columns={'date': 'Actual Date', "Name": "EPOD Name"})

    # melted_df['Actual Date'] = pd.to_datetime(
    #     melted_df['Actual Date'], format='%d/%m/%Y %H:%M', errors='coerce')
    return melted_df

vehicle_data_list = [pivot_df, df_vehicles_month]
melted_dfs = [convert_vehicle_data(df) for df in vehicle_data_list]
vehicle_df = pd.concat(melted_dfs, ignore_index=True)
vehicle_df = vehicle_df[vehicle_df['Customer Location City'].isin(cities)]
vehicle_df['KM Travelled for Session'] = vehicle_df['KM Travelled for Session'].replace(
    '-', '0', regex=True)
vehicle_df["Actual Date"] = vehicle_df["Actual Date"].apply(
    convert_to_datetime_with_current_year)

vehicle_df.to_csv('melted.csv')




#merged_df.to_csv(r"C:\Users\DELL\PycharmProjects\Excel\merdf2.csv")

image = Image.open(r'Hpcharge.png')
col1, col2, col3, col4, col5 = st.columns(5)
col3.image(image, use_column_width=False)
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["Executive Dashboard", "Charge Pattern Insights", "EPod Stats", "Operator Stats", "Subscription Insights", "Geographical Insights"])


with tab1:
    col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
    with col1:
        df['Actual Date'] = pd.to_datetime(df['Actual Date'], errors='coerce')
        min_date = df['Actual Date'].min().date()
        max_date = df['Actual Date'].max().date()
        start_date = st.date_input(
            'Start Date', min_value=min_date, max_value=max_date, value=min_date, key="ex-date-start")

    with col1:
        end_date = st.date_input(
            'End Date', min_value=min_date, max_value=max_date, value=max_date, key="ex-date-end")

    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    filtered_df = df[(df['Actual Date'] >= start_date)
                     & (df['Actual Date'] <= end_date)]




    filtered_df['Actual SoC_Start'] = pd.to_numeric(
        filtered_df['Actual SoC_Start'], errors='coerce')
    filtered_df['Actual Soc_End'] = pd.to_numeric(
        filtered_df['Actual Soc_End'], errors='coerce')


    # Process Actual SoC_Start and Actual Soc_End columns
    def process_soc(value):
        try:
            numeric_value = pd.to_numeric(value, errors='coerce')
            if numeric_value > 100:
                return int(str(numeric_value)[:2])  # Extract first 2 digits
            return numeric_value
        except:
            return np.nan


    filtered_df['Actual SoC_Start'] = filtered_df['Actual SoC_Start'].apply(process_soc)
    filtered_df['Actual Soc_End'] = filtered_df['Actual Soc_End'].apply(process_soc)

    record_count_df = filtered_df.groupby(
        ['EPOD Name', 't-15_kpi']).size().reset_index(name='Record Count')

    city_count_df = filtered_df.groupby(['Customer Location City', 't-15_kpi']).size().reset_index(
        name='Record Count')
    record_count_df = record_count_df.sort_values(by='Record Count')
    city_count_df = city_count_df.sort_values(by='Record Count')
    start_soc_stats = filtered_df.dropna(subset=['Actual SoC_Start']).groupby('EPOD Name')['Actual SoC_Start'].agg([
        'max', 'min', 'mean', 'median'])

    end_soc_stats = filtered_df.dropna(subset=['Actual Soc_End']).groupby('EPOD Name')['Actual Soc_End'].agg([
        'max', 'min', 'mean', 'median'])
    start_soc_stats = start_soc_stats.sort_values(by='EPOD Name')
    end_soc_stats = end_soc_stats.sort_values(by='EPOD Name')
    kpi_flag_data = filtered_df['t-15_kpi']

    before_time_count = (kpi_flag_data == 1).sum()
    on_time_count = (kpi_flag_data == 0).sum()
    delay_count = (kpi_flag_data == 2).sum()

    total_count = before_time_count + delay_count + on_time_count
    before_time_percentage = (before_time_count / total_count) * 100
    on_time_percentage = (on_time_count / total_count) * 100
    delay_percentage = (delay_count / total_count) * 100
    on_time_sla = (1 - (delay_percentage / 100))*100
    labels = ['T-15 Fulfilled', 'Delay']

    start_soc_avg = start_soc_stats['mean'].values.mean()
    start_soc_median = start_soc_stats['median'].values[0]

    end_soc_avg = end_soc_stats['mean'].values.mean()
    end_soc_median = end_soc_stats['median'].values[0]

    col2.metric("T-15 Fulfilled", f"{before_time_percentage.round(2)}%")
    col3.metric("On Time SLA", f"{on_time_sla.round(2)}%")
    #col4.metric("T-15 Not Fulfilled", f"{on_time_percentage.round(2)}%")
    col4.metric("Avg Start SoC", f"{start_soc_avg.round(2)}%")
    col5.metric("Avg End SoC", f"{end_soc_avg.round(2)}%")

    total_sessions = filtered_df['t-15_kpi'].count()
    fig = go.Figure(data=[go.Pie(labels=['T-15 Fulfilled', 'Delay', 'T-15 Not Fulfilled'],
                                 values=[before_time_count,
                                         delay_count, on_time_count],
                                 hole=0.6,
                                 sort=False,
                                 textinfo='label+percent+value',
                                 textposition='outside',
                                 marker=dict(colors=['green', 'red', 'yellow']))])

    fig.add_annotation(text='Total Sessions',
                       x=0.5, y=0.5, font_size=15, showarrow=False)

    fig.add_annotation(text=str(total_sessions),
                       x=0.5, y=0.45, font_size=15, showarrow=False)
    fig.update_layout(
        title='T-15 KPI (Overall)',
        showlegend=False,
        height=400,
        width=610
    )

    with col2:
        st.plotly_chart(fig, use_container_width=False)

    allowed_cities = df['Customer Location City'].dropna().unique()
    city_count_df = city_count_df[city_count_df['Customer Location City'].isin(
        allowed_cities)]

    fig_group = go.Figure()

    color_mapping = {0: 'red', 1: 'green', 2: 'yellow'}
    city_count_df['Percentage'] = city_count_df['Record Count'] / \
        city_count_df.groupby('Customer Location City')[
        'Record Count'].transform('sum') * 100

    fig_group = go.Figure()

    for flag in city_count_df['t-15_kpi'].unique():
        df_flag = city_count_df[city_count_df['t-15_kpi'] == flag]

        fig_group.add_trace(go.Bar(
            x=df_flag['Customer Location City'],
            y=df_flag['Percentage'],
            name=str(flag),
            text=df_flag['Percentage'].round(1).astype(str) + '%',
            marker=dict(color=color_mapping[flag]),
            textposition='auto'
        ))

    fig_group.update_layout(
        barmode='group',
        title='T-15 KPI (HSZ Wise)',
        xaxis={'categoryorder': 'total descending'},
        yaxis={'tickformat': '.2f', 'title': 'Percentage'},
        height=400,
        width=610,
        margin=dict(t=50, b=50, l=50, r=50),
        showlegend=False
    )

    with col5:
        st.plotly_chart(fig_group)

    filtered_city_count_df = city_count_df[city_count_df['t-15_kpi'] == 1]

    max_record_count_city = filtered_city_count_df.loc[
        filtered_city_count_df['Record Count'].idxmax(), 'Customer Location City']
    min_record_count_city = filtered_city_count_df.loc[
        filtered_city_count_df['Record Count'].idxmin(), 'Customer Location City']

    col6.metric("City with Maximum Sessions", max_record_count_city)
    col7.metric("City with Minimum Sessions", min_record_count_city)

    start_soc_max = start_soc_stats['max'].values.max()

    start_soc_min = start_soc_stats['min'].values.min()

    start_soc_avg = start_soc_stats['mean'].values.mean()
    start_soc_median = np.median(start_soc_stats['median'].values)

    gauge_range = [0, 100]

    start_soc_max_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=start_soc_max,
        title={'text': "Max Start SoC %", 'font': {'size': 15}},
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={'axis': {'range': gauge_range}},
    ))
    start_soc_max_gauge.update_layout(width=150, height=250)

    start_soc_min_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=start_soc_min,
        title={'text': "Min Start SoC %", 'font': {'size': 15}},
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={'axis': {'range': gauge_range}}
    ))
    start_soc_min_gauge.update_layout(width=150, height=250)

    start_soc_avg_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=start_soc_avg,
        title={'text': "Avg Start SoC %", 'font': {'size': 15}},
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={'axis': {'range': gauge_range}}
    ))
    start_soc_avg_gauge.update_layout(width=150, height=250)

    start_soc_median_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=start_soc_median,
        title={'text': "Median Start SoC %", 'font': {'size': 15}},
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={'axis': {'range': gauge_range}}
    ))
    start_soc_median_gauge.update_layout(width=150, height=250)
    start_soc_median_gauge.update_layout(
        # Adjust the margins as needed
        shapes=[dict(
            type='line',
            x0=1,
            y0=-2,
            x1=1,
            y1=2,
            line=dict(
                color="black",
                width=1,
            )
        )]
    )
    with col3:
        for i in range(1, 27):
            st.write("\n")
        with col2:
            st.write("#### Start SoC Stats")


    with col6:
        for i in range(1, 27):
            st.write("\n")
        st.write("#### End SoC Stats")

    # Create the layout using grid container
    st.markdown('<div class="grid-container">', unsafe_allow_html=True)

    col1, col2, col3, col4, col5, col6, col7, col8 = st.columns(8)
    with col1:
        st.plotly_chart(start_soc_min_gauge)
    with col2:
        st.plotly_chart(start_soc_max_gauge)
    with col3:
        st.plotly_chart(start_soc_avg_gauge)
    with col4:
        st.plotly_chart(start_soc_median_gauge)

    end_soc_max = end_soc_stats['max'].values.max()
    end_soc_min = end_soc_stats['min'].values.min()
    end_soc_avg = end_soc_stats['mean'].values.mean()
    end_soc_median = np.median(end_soc_stats['median'].values)

    end_soc_max_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=end_soc_max,
        title={'text': "Max End SoC %", 'font': {'size': 15}},
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={'axis': {'range': gauge_range}}
    ))

    end_soc_min_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=end_soc_min,
        title={'text': "Min End SoC %", 'font': {'size': 15}},
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={'axis': {'range': gauge_range}}
    ))
    end_soc_min_gauge.update_layout(
        shapes=[
            dict(
                type='line',
                xref='paper',
                yref='paper',
                x0=0,
                y0=-2,
                x1=0,
                y1=2,
                line=dict(
                    color='black',
                    width=1
                )
            )
        ]
    )
    end_soc_max_gauge.update_layout(width=150, height=250)
    end_soc_min_gauge.update_layout(width=150, height=250)

    end_soc_avg_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=end_soc_avg,
        title={'text': "Avg End SoC %", 'font': {'size': 15}},
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={'axis': {'range': gauge_range}}
    ))
    end_soc_avg_gauge.update_layout(width=150, height=250)

    end_soc_median_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=end_soc_median,
        title={'text': "Median End SoC %", 'font': {'size': 15}},
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={'axis': {'range': gauge_range}}
    ))
    end_soc_median_gauge.update_layout(width=150, height=250)

    with col5:
        st.plotly_chart(end_soc_min_gauge)

    with col6:
        st.plotly_chart(end_soc_max_gauge)
    with col7:
        st.plotly_chart(end_soc_avg_gauge)
    with col8:
        st.plotly_chart(end_soc_median_gauge)

    for city in allowed_cities:
        city_filtered_df = filtered_df[filtered_df['Customer Location City'] == city]

        city_start_soc_stats = city_filtered_df.dropna(subset=['Actual SoC_Start'])['Actual SoC_Start'].agg([
            'max', 'min', 'mean', 'median'])

        city_end_soc_stats = city_filtered_df.dropna(subset=['Actual Soc_End'])['Actual Soc_End'].agg([
            'max', 'min', 'mean', 'median'])

        city_start_soc_max = city_start_soc_stats['max'].max()
        city_start_soc_min = city_start_soc_stats['min'].min()
        city_start_soc_avg = city_start_soc_stats['mean'].mean()
        city_start_soc_median = np.median(city_start_soc_stats['median'])

        city_end_soc_max = city_end_soc_stats['max'].max()
        city_end_soc_min = city_end_soc_stats['min'].min()
        city_end_soc_avg = city_end_soc_stats['mean'].mean()
        city_end_soc_median = np.median(city_end_soc_stats['median'])

        city_start_soc_max_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=city_start_soc_max,
            title={'text': f"Start Max - {city}", 'font': {'size': 15}},
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={'axis': {'range': gauge_range}}
        ))
        city_start_soc_max_gauge.update_layout(width=150, height=250)
        city_start_soc_min_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=city_start_soc_min,
            title={'text': f"Start Min - {city}", 'font': {'size': 15}},
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={'axis': {'range': gauge_range}}
        ))
        city_start_soc_min_gauge.update_layout(width=150, height=250)
        city_start_soc_avg_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=city_start_soc_avg,
            title={'text': f"Start Avg - {city}", 'font': {'size': 15}},
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={'axis': {'range': gauge_range}}
        ))
        city_start_soc_avg_gauge.update_layout(width=150, height=250)
        city_start_soc_median_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=city_start_soc_median,
            title={'text': f"Start Median - {city}", 'font': {'size': 15}},
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={'axis': {'range': gauge_range}}
        ))
        city_start_soc_median_gauge.update_layout(
            # Adjust the margins as needed
            shapes=[dict(
                type='line',
                x0=1,
                y0=-2,
                x1=1,
                y1=2,
                line=dict(
                    color="black",
                    width=1,
                )
            )]
        )
        city_start_soc_median_gauge.update_layout(width=150, height=250)
        city_end_soc_max_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=city_end_soc_max,
            title={'text': f"End Max - {city}", 'font': {'size': 15}},
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={'axis': {'range': gauge_range}}
        ))
        city_end_soc_max_gauge.update_layout(width=150, height=250)
        city_end_soc_min_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=city_end_soc_min,
            title={'text': f"End Min - {city}", 'font': {'size': 15}},
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={'axis': {'range': gauge_range}}
        ))
        city_end_soc_min_gauge.update_layout(width=150, height=250)
        city_end_soc_min_gauge.update_layout(
            shapes=[
                dict(
                    type='line',
                    xref='paper',
                    yref='paper',
                    x0=0,
                    y0=-2,
                    x1=0,
                    y1=2,
                    line=dict(
                        color='black',
                        width=1
                    )
                )
            ]
        )
        city_end_soc_avg_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=city_end_soc_avg,
            title={'text': f"End Avg - {city}", 'font': {'size': 15}},
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={'axis': {'range': gauge_range}}
        ))
        city_end_soc_avg_gauge.update_layout(width=150, height=250)
        city_end_soc_median_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=city_end_soc_median,
            title={'text': f"End Median - {city}", 'font': {'size': 15}},
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={'axis': {'range': gauge_range}}
        ))
        city_end_soc_median_gauge.update_layout(width=150, height=250)

        st.subheader(city)
        col1, col2, col3, col4, col5, col6, col7, col8 = st.columns(8)
        with col1:
            st.plotly_chart(city_start_soc_min_gauge)

        with col2:
            st.plotly_chart(city_start_soc_max_gauge)

        with col3:
            st.plotly_chart(city_start_soc_avg_gauge)

        with col4:
            st.plotly_chart(city_start_soc_median_gauge)

        with col5:
            st.plotly_chart(city_end_soc_min_gauge)

        with col6:
            st.plotly_chart(city_end_soc_max_gauge)
        with col7:
            st.plotly_chart(city_end_soc_avg_gauge)
        with col8:
            st.plotly_chart(city_end_soc_median_gauge)

with tab2:
    CustomerNames = df['Customer Name'].unique()
    SubscriptionNames = df['type'].unique()

    col1, col2, col3, col4, col5, col6, col7 = st.columns(7)

    with col1:
        df['Actual Date'] = pd.to_datetime(df['Actual Date'], errors='coerce')
        min_date = df['Actual Date'].min().date()
        max_date = df['Actual Date'].max().date()
        start_date = st.date_input(
            'Start Date', min_value=min_date, max_value=max_date, value=min_date, key="cpi-date-start")

    with col2:
        end_date = st.date_input(
            'End Date', min_value=min_date, max_value=max_date, value=max_date, key="cpi-date-end")

    with col4:
        Name = st.multiselect(label='Select The Customers',
                              options=['All'] + CustomerNames.tolist(),
                              default='All')

    with col3:
        Sub_filter = st.multiselect(label='Select Subscription',
                                    options=['All'] + SubscriptionNames.tolist(),
                                    default='All')

    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    filtered_data = df[(df['Actual Date'] >= start_date) & (df['Actual Date'] <= end_date)]
    if 'All' in Name:
        Name = CustomerNames
    if 'All' in Sub_filter:
        Sub_filter = SubscriptionNames
    filtered_data = filtered_data[
        (filtered_data['Customer Name'].isin(Name)) & (filtered_data['type'].isin(Sub_filter))]


    def generate_multiline_plot(data):
        fig = go.Figure()
        color_map = {0: 'yellow', 1: 'green', 2: 'red'}
        names = {0: "T-15 Not Fulfilled", 1: "T-15 Fulfilled", 2: "Delayed"}

        # Create a new DataFrame to store the counts for each day
        daily_counts = data.pivot_table(index='Day', columns='t-15_kpi', values='count', fill_value=0).reset_index()
        daily_counts['On-Time SLA'] = daily_counts[0] + daily_counts[1]
        daily_counts['Total Count'] = daily_counts[0] + daily_counts[1] + daily_counts[2]

        for kpi_flag in data['t-15_kpi'].unique():
            subset = data[data['t-15_kpi'] == kpi_flag]
            fig.add_trace(go.Scatter(x=subset['Day'], y=subset['count'], mode='lines+text',
                                     name=names[kpi_flag], line_color=color_map[kpi_flag],
                                     text=[
                                         f"{round(count / daily_counts[daily_counts['Day'] == day]['Total Count'].values[0] * 100, 0)}%"
                                         for day, count in zip(subset['Day'], subset['count'])],
                                     textposition='top center',
                                     showlegend=True))

        # Add the "On-Time SLA" line to the plot
        fig.add_trace(go.Scatter(x=daily_counts['Day'], y=daily_counts['On-Time SLA'], mode='lines+text',
                                 name='On-Time SLA', line_color='purple',
                                 text=[
                                     f"{round(count / daily_counts[daily_counts['Day'] == day]['Total Count'].values[0] * 100, 0)}%"
                                     for day, count in zip(daily_counts['Day'], daily_counts['On-Time SLA'])],
                                 textposition='top center',
                                 showlegend=True))

        fig.add_trace(go.Scatter(x=daily_counts['Day'], y=daily_counts['Total Count'], mode='lines+markers+text',
                                 name='Total Count', line_color='blue',
                                 text=daily_counts['Total Count'],
                                 textposition='top center',
                                 showlegend=True))

        fig.update_layout(
            xaxis_title='Day', yaxis_title='Count', legend=dict(x=0, y=1.2, orientation='h'))
        fig.update_yaxes(title='Count', range=[
            0, daily_counts['Total Count'].max() * 1.2])
        fig.update_layout(width=500, height=500)
        return fig


    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    filtered_data['Day'] = pd.Categorical(filtered_data['Day'], categories=day_order, ordered=True)
    daily_count = filtered_data.groupby(['Day', 't-15_kpi']).size().reset_index(name='count')
    maxday = filtered_data.groupby(['Day']).size().reset_index(name='count')
    maxday['count'] = maxday['count'].astype(int)
    max_count_index = maxday['count'].idxmax()
    max_count_day = maxday.loc[max_count_index, 'Day']
    minday = filtered_data.groupby(['Day']).size().reset_index(name='count')
    minday['count'] = minday['count'].astype(int)
    min_count_index = minday['count'].idxmin()
    min_count_day = minday.loc[min_count_index, 'Day']

    with col7:
        for i in range(1, 10):
            st.write("\n")
        st.markdown("Most Sessions on Day")
        st.markdown("<span style = 'font-size:25px;line-height: 0.8;'>" + max_count_day + "</span>",
                    unsafe_allow_html=True)

    with col7:
        st.markdown("Min Sessions on Day")
        st.markdown("<span style = 'font-size:25px;line-height: 0.8;'>" + min_count_day + "</span>",
                    unsafe_allow_html=True)

    multiline_plot = generate_multiline_plot(daily_count)
    with col4:
        st.plotly_chart(multiline_plot)


    def count_t15_kpi(df):
        try:
            return df.groupby(['t-15_kpi']).size()['1']
        except KeyError:
            return 0


    def count_sessions(df):
        return df.shape[0]


    def count_cancelled(df):
        try:
            return df[df['canceled'] == True].shape[0]
        except KeyError:
            return 0


    def count_cancelled_with_penalty(df):
        try:
            return df[df['cancelledPenalty'] == 1].shape[0]
        except KeyError:
            return 0


    total_sessions = count_sessions(filtered_data)
    cancelled_sessions = count_cancelled(filtered_data)
    cancelled_sessions_with_penalty = count_cancelled_with_penalty(filtered_data)

    # Calculate Cancelled Sessions without Penalty (cancelled but without penalty)
    cancelled_sessions_without_penalty = cancelled_sessions

    labels = ['Actual Sessions', 'Cancelled Sessions', 'Cancelled with Penalty']
    values = [total_sessions, cancelled_sessions_without_penalty, cancelled_sessions_with_penalty]
    colors = ['blue', 'orange', 'red']

    fig = go.Figure(
        data=[go.Pie(labels=labels, values=values, hole=0.7, textinfo='label+percent', marker=dict(colors=colors))])

    fig.update_layout(showlegend=True, width=500)

    fig.add_annotation(
        text=f"Overall Sessions: {total_sessions}", x=0.5, y=0.5, font_size=15, showarrow=False)

    fig.update_layout(width=500, height=400)

    with col1:
        st.plotly_chart(fig)


    def generate_multiline_plot(data):
        fig = go.Figure()
        color_map = {0: 'yellow', 1: 'green', 2: 'red'}
        names = {0: "T-15 Not Fulfilled", 1: "T-15 Fulfilled", 2: "Delayed"}

        time_counts = data.pivot_table(index='Booking Session time', columns='t-15_kpi', values='count',
                                       fill_value=0).reset_index()
        time_counts['On-Time SLA'] = time_counts[0] + time_counts[1]
        time_counts['Total Count'] = time_counts[0] + time_counts[1] + time_counts[2]

        fig.update_layout(xaxis_title='Booking Session Time',
                          yaxis_title='Count', legend=dict(x=0, y=1.2, orientation='h'))


        for kpi_flag in data['t-15_kpi'].unique():
            subset = data[data['t-15_kpi'] == kpi_flag]
            fig.add_trace(go.Scatter(x=subset['Booking Session time'], y=subset['count'], mode='lines+text',
                                     name=names[kpi_flag], line_color=color_map[kpi_flag],
                                     text=[
                                         f"{round(count / time_counts[time_counts['Booking Session time'] == hr]['Total Count'].values[0] * 100, 0)}%"
                                         for hr, count in zip(subset['Booking Session time'], subset['count'])],
                                     textposition='top center',
                                     showlegend=True))

        # Add the "On-Time SLA" line to the plot
        fig.add_trace(go.Scatter(x=time_counts['Booking Session time'], y=time_counts['On-Time SLA'], mode='lines+text',
                                 name='On-Time SLA', line_color='purple',
                                 text=[
                                     f"{round(count / time_counts[time_counts['Booking Session time'] == day]['Total Count'].values[0] * 100, 0)}%"
                                     for day, count in zip(time_counts['Booking Session time'], time_counts['On-Time SLA'])],
                                 textposition='top center',
                                 showlegend=True))

        fig.add_trace(go.Scatter(x=time_counts['Booking Session time'], y=time_counts['Total Count'], mode='lines+markers+text',
                                 name='Total Count', line_color='blue',
                                 text=time_counts['Total Count'],
                                 textposition='top center',
                                 showlegend=True))


        fig.update_yaxes(title='Count', range=[
            0, time_counts['Total Count'].max() * 1.2])
        fig.update_layout(xaxis=dict(tickmode='array', tickvals=list(
            range(24)), ticktext=list(range(24))))
        fig.update_layout(width=1100, height=530)
        return fig


    filtered_data['Booking Session time'] = pd.to_datetime(
        filtered_data['Booking Session time'], format='mixed').dt.hour
    daily_count = filtered_data.groupby(
        ['Booking Session time', 't-15_kpi']).size().reset_index(name='count')
    maxmindf = filtered_data.groupby(
        ['Booking Session time']).size().reset_index(name='count')
    max_count_index = maxmindf['count'].idxmax()
    max_count_time = maxmindf.loc[max_count_index, 'Booking Session time']
    min_count_index = maxmindf['count'].idxmin()
    min_count_time = maxmindf.loc[min_count_index, 'Booking Session time']
    with col7:
        for i in range(1, 18):
            st.write("\n")
        st.markdown("Max Sessions at Time")
        st.markdown("<span style = 'font-size:25px;line-height: 0.8;'>" +
                    str(max_count_time) + "</span>", unsafe_allow_html=True)
    with col7:
        st.markdown("Min Sessions at Time")
        st.markdown("<span style = 'font-size:25px;line-height: 0.8;'>" +
                    str(min_count_time) + "</span>", unsafe_allow_html=True)
    multiline_plot = generate_multiline_plot(daily_count)
    with col1:
        st.plotly_chart(multiline_plot)
    st.divider()






    HSZs = df['Customer Location City'].dropna().unique()
    for city in HSZs:
        st.subheader(city)
        CustomerNames = df['Customer Name'].unique()
        SubscriptionNames = df['type'].unique()

        col1, col2, col3, col4, col5, col6, col7 = st.columns(7)

        with col1:
            df['Actual Date'] = pd.to_datetime(
                df['Actual Date'], errors='coerce')
            min_date = df['Actual Date'].min().date()
            max_date = df['Actual Date'].max().date()
            start_date = st.date_input(
                'Start Date', min_value=min_date, max_value=max_date, value=min_date, key=f"{city}cpi-date-start")

        with col2:
            end_date = st.date_input(
                'End Date', min_value=min_date, max_value=max_date, value=max_date, key=f"{city}cpi-date-end")
        with col4:

            Name = st.multiselect(label='Select The Customers',
                                  options=['All'] + CustomerNames.tolist(),
                                  default='All', key=f"{city}names")

        with col3:
            Sub_filter = st.multiselect(label='Select Subscription',
                                        options=['All'] +
                                        SubscriptionNames.tolist(),
                                        default='All', key=f"{city}sub")
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        filtered_data = df[(df['Actual Date'] >= start_date)
                           & (df['Actual Date'] <= end_date)]
        if 'All' in Name:
            Name = CustomerNames

        if 'All' in Sub_filter:
            Sub_filter = SubscriptionNames

        filtered_data = filtered_data[
            (filtered_data['Customer Name'].isin(Name)) &
            (filtered_data['type'].isin(Sub_filter))
        ]
        filtered_data = filtered_data[
            (filtered_data['Customer Location City'] == city)]


        def generate_multiline_plot(data):
            fig = go.Figure()
            color_map = {0: 'yellow', 1: 'green', 2: 'red'}
            names = {0: "T-15 Not Fulfilled", 1: "T-15 Fulfilled", 2: "Delayed"}

            # Create a new DataFrame to store the counts for each day
            daily_counts = data.pivot_table(index='Day', columns='t-15_kpi', values='count', fill_value=0).reset_index()
            daily_counts['On-Time SLA'] = daily_counts[0] + daily_counts[1]
            daily_counts['Total Count'] = daily_counts[0] + daily_counts[1] + daily_counts[2]

            for kpi_flag in data['t-15_kpi'].unique():
                subset = data[data['t-15_kpi'] == kpi_flag]
                fig.add_trace(go.Scatter(x=subset['Day'], y=subset['count'], mode='lines+text',
                                         name=names[kpi_flag], line_color=color_map[kpi_flag],
                                         text=[
                                             f"{round(count / daily_counts[daily_counts['Day'] == day]['Total Count'].values[0] * 100, 0)}%"
                                             for day, count in zip(subset['Day'], subset['count'])],
                                         textposition='top center',
                                         showlegend=True))

            # Add the "On-Time SLA" line to the plot
            fig.add_trace(go.Scatter(x=daily_counts['Day'], y=daily_counts['On-Time SLA'], mode='lines+text',
                                     name='On-Time SLA', line_color='purple',
                                     text=[
                                         f"{round(count / daily_counts[daily_counts['Day'] == day]['Total Count'].values[0] * 100, 0)}%"
                                         for day, count in zip(daily_counts['Day'], daily_counts['On-Time SLA'])],
                                     textposition='top center',
                                     showlegend=True))

            fig.add_trace(go.Scatter(x=daily_counts['Day'], y=daily_counts['Total Count'], mode='lines+markers+text',
                                     name='Total Count', line_color='blue',
                                     text=daily_counts['Total Count'],
                                     textposition='top center',
                                     showlegend=True))

            fig.update_layout(
                xaxis_title='Day', yaxis_title='Count', legend=dict(x=0, y=1.2, orientation='h'))
            fig.update_yaxes(title='Count', range=[
                0, daily_counts['Total Count'].max() * 1.2])
            fig.update_layout(width=500, height=500)
            return fig
        day_order = ['Monday', 'Tuesday', 'Wednesday',
                     'Thursday', 'Friday', 'Saturday', 'Sunday']
        filtered_data['Day'] = pd.Categorical(
            filtered_data['Day'], categories=day_order, ordered=True)
        daily_count = filtered_data.groupby(
            ['Day', 't-15_kpi']).size().reset_index(name='count')
        maxday = filtered_data.groupby(
            ['Day']).size().reset_index(name='count')
        maxday['count'] = maxday['count'].astype(int)
        max_count_index = maxday['count'].idxmax()
        max_count_day = maxday.loc[max_count_index, 'Day']
        minday = filtered_data.groupby(
            ['Day']).size().reset_index(name='count')
        minday['count'] = minday['count'].astype(int)
        min_count_index = minday['count'].idxmin()
        min_count_day = minday.loc[min_count_index, 'Day']
        with col7:
            for i in range(1, 10):
                st.write("\n")
            st.markdown("Most Sessions on Day")
            st.markdown("<span style = 'font-size:25px;line-height: 0.8;'>" +
                        max_count_day+"</span>", unsafe_allow_html=True)
        with col7:
            st.markdown("Min Sessions on Day")
            st.markdown("<span style = 'font-size:25px;line-height: 0.8;'>" +
                        min_count_day+"</span>", unsafe_allow_html=True)
        multiline_plot = generate_multiline_plot(daily_count)

        with col4:

            st.plotly_chart(multiline_plot)

        def count_t15_kpi(df):
            try:
                return df.groupby(
                    ['t-15_kpi']).size()['1']
            except KeyError:
                return 0


        def count_sessions(df):
            return df.shape[0]


        def count_cancelled(df):
            try:
                return df[df['canceled'] == True].shape[0]
            except KeyError:
                return 0


        def count_cancelled_with_penalty(df):
            try:
                return df[df['cancelledPenalty'] == 1].shape[0]
            except KeyError:
                return 0


        total_sessions = count_sessions(filtered_data)
        cancelled_sessions = count_cancelled(filtered_data)
        cancelled_sessions_with_penalty = count_cancelled_with_penalty(filtered_data)

        # Calculate Cancelled Sessions without Penalty (cancelled but without penalty)
        cancelled_sessions_without_penalty = cancelled_sessions


        labels = ['Actual Sessions', 'Cancelled Sessions', 'Cancelled with Penalty']
        values = [total_sessions, cancelled_sessions_without_penalty, cancelled_sessions_with_penalty]
        colors = ['blue', 'orange', 'red']

        fig = go.Figure(
            data=[go.Pie(labels=labels, values=values, hole=0.7, textinfo='label+percent', marker=dict(colors=colors))])

        fig.update_layout(
            showlegend=True, width=500,
        )

        fig.add_annotation(
            text=f"Overall Sessions: {total_sessions}", x=0.5, y=0.5, font_size=15, showarrow=False)

        fig.update_layout(width=500, height=400)

        with col1:
            st.plotly_chart(fig)


        def generate_multiline_plot(data):
            fig = go.Figure()
            color_map = {0: 'yellow', 1: 'green', 2: 'red'}
            names = {0: "T-15 Not Fulfilled", 1: "T-15 Fulfilled", 2: "Delayed"}

            time_counts = data.pivot_table(index='Booking Session time', columns='t-15_kpi', values='count',
                                           fill_value=0).reset_index()
            time_counts['On-Time SLA'] = time_counts[0] + time_counts[1]
            time_counts['Total Count'] = time_counts[0] + time_counts[1] + time_counts[2]

            fig.update_layout(xaxis_title='Booking Session Time',
                              yaxis_title='Count', legend=dict(x=0, y=1.2, orientation='h'))

            for kpi_flag in data['t-15_kpi'].unique():
                subset = data[data['t-15_kpi'] == kpi_flag]
                fig.add_trace(go.Scatter(x=subset['Booking Session time'], y=subset['count'], mode='lines+text',
                                         name=names[kpi_flag], line_color=color_map[kpi_flag],
                                         text=[
                                             f"{round(count / time_counts[time_counts['Booking Session time'] == hr]['Total Count'].values[0] * 100, 0)}%"
                                             for hr, count in zip(subset['Booking Session time'], subset['count'])],
                                         textposition='top center',
                                         showlegend=True))

            # Add the "On-Time SLA" line to the plot
            fig.add_trace(
                go.Scatter(x=time_counts['Booking Session time'], y=time_counts['On-Time SLA'], mode='lines+text',
                           name='On-Time SLA', line_color='purple',
                           text=[
                               f"{round(count / time_counts[time_counts['Booking Session time'] == day]['Total Count'].values[0] * 100, 0)}%"
                               for day, count in zip(time_counts['Booking Session time'], time_counts['On-Time SLA'])],
                           textposition='top center',
                           showlegend=True))

            fig.add_trace(go.Scatter(x=time_counts['Booking Session time'], y=time_counts['Total Count'],
                                     mode='lines+markers+text',
                                     name='Total Count', line_color='blue',
                                     text=time_counts['Total Count'],
                                     textposition='top center',
                                     showlegend=True))

            fig.update_yaxes(title='Count', range=[
                0, time_counts['Total Count'].max() * 1.2])
            fig.update_layout(xaxis=dict(tickmode='array', tickvals=list(
                range(24)), ticktext=list(range(24))))
            fig.update_layout(width=1100, height=530)
            return fig
        filtered_data['Booking Session time'] = pd.to_datetime(
            filtered_data['Booking Session time'], format='mixed').dt.hour
        daily_count = filtered_data.groupby(
            ['Booking Session time', 't-15_kpi']).size().reset_index(name='count')
        maxmindf = filtered_data.groupby(
            ['Booking Session time']).size().reset_index(name='count')
        max_count_index = maxmindf['count'].idxmax()
        max_count_time = maxmindf.loc[max_count_index, 'Booking Session time']
        min_count_index = maxmindf['count'].idxmin()
        min_count_time = maxmindf.loc[min_count_index, 'Booking Session time']
        with col7:
            for i in range(1, 18):
                st.write("\n")
            st.markdown("Max Sessions at Time")
            st.markdown("<span style = 'font-size:25px;line-height: 0.8;'>" +
                        str(max_count_time)+"</span>", unsafe_allow_html=True)

        with col7:
            st.markdown("Min Sessions at Time")
            st.markdown("<span style = 'font-size:25px;line-height: 0.8;'>" +
                        str(min_count_time)+"</span>", unsafe_allow_html=True)

        multiline_plot = generate_multiline_plot(daily_count)

        with col1:
            st.plotly_chart(multiline_plot)

        st.divider()

with tab3:
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        df['Actual Date'] = pd.to_datetime(df['Actual Date'], errors='coerce')
        min_date = df['Actual Date'].min().date()
        max_date = df['Actual Date'].max().date()
        start_date = st.date_input(
            'Start Date', min_value=min_date, max_value=max_date, value=min_date, key="epod-date-start")
    with col2:
        end_date = st.date_input(
            'End Date', min_value=min_date, max_value=max_date, value=max_date, key="epod-date-end")
    df['EPOD Name'] = df['EPOD Name'].str.replace('-', '')

    epods = vehicle_df['EPOD Name'].unique()

    with col3:
        EPod = st.multiselect(label='Select The EPOD',
                              options=['All'] + epods.tolist(),
                              default='All')
    with col1:
        st.markdown(":large_green_square: T-15 fulfilled")

    with col2:
        st.markdown(":large_yellow_square: T-15 Not fulfilled")
    with col3:
        st.markdown(":large_red_square: Delay")

    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    filtered_data = df[(df['Actual Date'] >= start_date)
                       & (df['Actual Date'] <= end_date)]

    if 'All' in EPod:
        EPod = epods

    filtered_data = filtered_data[
        (filtered_data['EPOD Name'].isin(EPod))]

    filtered_data_vehicle = vehicle_df[(vehicle_df['Actual Date'] >= start_date)
                                       & (vehicle_df['Actual Date'] <= end_date)]

    filtered_data_vehicle = filtered_data_vehicle[filtered_data_vehicle['EPOD Name'].isin(
        EPod)]

    record_count_df = filtered_data.groupby(
        ['EPOD Name', 't-15_kpi']).size().reset_index(name='Record Count')
    color_mapping = {0: 'yellow', 1: 'green', 2: 'red'}
    record_count_df['Color'] = record_count_df['t-15_kpi'].map(color_mapping)

    record_count_df = record_count_df.sort_values('EPOD Name')
    y_axis_range = [0, record_count_df['Record Count'].max() * 1.2]

    # Calculate average duration per EPod per session
    average_duration = filtered_data.groupby('EPOD Name')['Duration'].mean().reset_index().round(1)

    # Calculate average duration per session across all EPods
    avgdur = average_duration['Duration'].mean().round(2)

    # Display Average Duration/Session
    with col5:
        st.markdown("Average Duration/EPod")
        st.markdown("<span style='font-size: 25px;line-height: 0.8;'>" +
                    str(avgdur) + "</span>", unsafe_allow_html=True)

    # Filter and process data for vehicle DataFrame
    filtered_data_vehicle['KM Travelled for Session'] = filtered_data_vehicle['KM Travelled for Session'].replace('',
                                                                                                                  np.nan)
    filtered_data_vehicle['KM Travelled for Session'] = filtered_data_vehicle['KM Travelled for Session'].astype(float)

    # Calculate average kilometers per EPod per session
    average_kms = filtered_data_vehicle.groupby('EPOD Name')['KM Travelled for Session'].mean().reset_index().round(1)

    # Calculate average kilometers per session across all EPods
    avgkm = average_kms['KM Travelled for Session'].mean().round(2)

    # Display Average Kms/EPod
    with col4:
        st.markdown("Average Kms/EPod")
        st.markdown("<span style='font-size: 25px;line-height: 0.8;'>" +
                    str(avgkm) + "</span>", unsafe_allow_html=True)

    # Filter and process data for 'filtered_data' DataFrame
    filtered_data['KWH Pumped Per Session'] = filtered_data['KWH Pumped Per Session'].replace('', np.nan)
    filtered_data = filtered_data[filtered_data['KWH Pumped Per Session'] != '#VALUE!']
    filtered_data['KWH Pumped Per Session'] = filtered_data['KWH Pumped Per Session'].astype(float)
    filtered_data['KWH Pumped Per Session'] = filtered_data['KWH Pumped Per Session'].abs()

    # Calculate average kWh per EPod per session
    average_kwh = filtered_data.groupby('EPOD Name')['KWH Pumped Per Session'].mean().reset_index().round(1)

    # Calculate average kWh per session across all EPods
    avgkwh = average_kwh['KWH Pumped Per Session'].mean().round(2)

    # Display Average kWh/EPod
    with col6:
        st.markdown("Average kWh/EPod")
        st.markdown("<span style='font-size: 25px;line-height: 0.8;'>" +
                    str(avgkwh) + "</span>", unsafe_allow_html=True)

    # Calculate the total count for each EPod
    total_count_per_epod = filtered_data.groupby('EPOD Name')['t-15_kpi'].count().reset_index(name='Total Count')

    # Merge the total count with record_count_df to get the denominator for percentage calculation
    record_count_df = record_count_df.merge(total_count_per_epod, on='EPOD Name')

    # Calculate the percentage for each EPod
    record_count_df['Percentage'] = (record_count_df['Record Count'] / record_count_df['Total Count']) * 100

    # Calculate the percentage of T-15 Fulfilled and T-15 Not Fulfilled for each EPod
    sla_data = record_count_df.pivot(index='EPOD Name', columns='t-15_kpi', values='Percentage').reset_index()
    sla_data['On-Time SLA'] = sla_data[0] + sla_data[1]

    max_value = max(record_count_df['Record Count'].max(), sla_data['On-Time SLA'].max()) * 1.2

    fig = go.Figure()
    # Add the bar traces for T-15 Fulfilled, T-15 Not Fulfilled, and Delay
    for color, kpi_group in record_count_df.groupby('Color'):
        fig.add_trace(go.Bar(
            x=kpi_group['EPOD Name'],
            y=kpi_group['Percentage'],
            text=kpi_group['Percentage'].round(2).astype(str) + '%',
            textposition='auto',
            name=color,
            marker=dict(color=color),
            width=0.38,
            showlegend=False
        ))

    # Add the line trace for On-Time SLA
    fig.add_trace(go.Scatter(
        x=sla_data['EPOD Name'],
        y=sla_data['On-Time SLA'],
        text=sla_data['On-Time SLA'].round(0).astype(str) + '%',  # Display On-Time SLA text
        textposition='top center',
        mode='lines+markers+text',  # Add 'text' to display the text values
        line=dict(color='purple', width=2),  # Set the line color to purple
        marker=dict(color='purple', size=8),
        name='On-Time SLA',
        yaxis='y2'  # Plot the line on the secondary y-axis
    ))

    fig.update_layout(
        xaxis={'categoryorder': 'array', 'categoryarray': record_count_df['EPOD Name']},
        yaxis={'range': [0, max_value]},
        xaxis_title='EPOD Name',
        yaxis_title='Sessions',
        yaxis2=dict(overlaying='y', side='right', showgrid=False, range=[0, max_value]),
        height=340,
        width=600,
        title="T-15 for each EPod with On-Time SLA",
        legend=dict(title_font=dict(size=14), font=dict(size=12), x=0, y=1.1, orientation='h'),
    )

    with col1:
        st.plotly_chart(fig)

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=average_duration['EPOD Name'],
        y=average_duration['Duration'],
        text=average_duration['Duration'].round(2),
        textposition='auto',
        name='Average Duration',

    ))

    fig.update_layout(
        xaxis_title='EPOD Name',
        yaxis_title='Average Duration',
        barmode='group',
        width=600,
        height=340,
        title="Avg Duration Per EPod",

    )
    with col4:
        for i in range(1, 4):
            st.write("\n")
        st.plotly_chart(fig)

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=average_kms['EPOD Name'],
        y=average_kms['KM Travelled for Session'],
        text=average_kms['KM Travelled for Session'].round(2),
        textposition='auto',
        name='Average KM Travelled for Session',

    ))

    fig.update_layout(
        xaxis_title='EPOD Name',
        yaxis_title='Average KM Travelled for Session',
        barmode='group',
        width=600,
        height=340,
        title="Avg KMs Per EPod",
    )
    with col1:
        st.plotly_chart(fig)

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=average_kwh['EPOD Name'],
        y=average_kwh['KWH Pumped Per Session'],  # Use 'KWH Pumped Per Session' column for y-values
        text=average_kwh['KWH Pumped Per Session'].round(2),
        textposition='auto',
        name='Average KWH Pumped Per Session',
    ))

    fig.update_layout(
        xaxis_title='EPOD Name',
        yaxis_title='Average KWH Pumped Per Session',
        barmode='group',
        width=600,
        height=340,
        title="Avg kWh Per EPod",
    )

    # Assuming you have defined 'col4' for the layout in Streamlit
    with col4:
        st.plotly_chart(fig)

    # Calculate the average sessions per EPod in the selected time period
    average_sessions_per_epod = filtered_data.groupby('EPOD Name')['t-15_kpi'].count().reset_index(
        name='Sessions Count')
    average_sessions_per_epod['Average Sessions'] = average_sessions_per_epod.groupby('EPOD Name')[
        'Sessions Count'].transform('mean')

    # Calculate the total number of sessions in the selected time period
    total_sessions = average_sessions_per_epod['Sessions Count'].sum()

    # Calculate the average sessions per EPod in percentage
    average_sessions_per_epod['Average Sessions (%)'] = (average_sessions_per_epod[
                                                             'Average Sessions'] / total_sessions) * 100

    # Sort the data based on 'EPOD Name'
    average_sessions_per_epod = average_sessions_per_epod.sort_values('EPOD Name')

    # Create the line graph
    fig_line = go.Figure()

    fig_line.add_trace(go.Scatter(
        x=average_sessions_per_epod['EPOD Name'],
        y=average_sessions_per_epod['Average Sessions (%)'],
        mode='lines+markers+text',  # Use 'markers+text' mode to display text on data points
        line=dict(color='orange', width=2),
        marker=dict(color='orange', size=8),
        name='Average Sessions per EPod (%)',
        text=average_sessions_per_epod['Average Sessions (%)'].round(1),
        textposition='top center',
    ))

    fig_line.update_layout(
        xaxis={'categoryorder': 'array', 'categoryarray': average_sessions_per_epod['EPOD Name']},
        yaxis={'title': 'Average Sessions (%)'},
        height=400,
        width=1200,
        title="Average Sessions per EPod (%)",
        legend=dict(title_font=dict(size=14), font=dict(size=12), x=0, y=1.1, orientation='h'),
    )
    with col1:
        st.plotly_chart(fig_line)


    st.divider()
    for city in df['Customer Location City'].dropna().unique():

        st.subheader(city)

        col1, col2, col3, col4, col5, col6 = st.columns(6)
        with col1:
            df['Actual Date'] = pd.to_datetime(
                df['Actual Date'], errors='coerce')
            min_date = df['Actual Date'].min().date()
            max_date = df['Actual Date'].max().date()
            start_date = st.date_input(
                'Start Date', min_value=min_date, max_value=max_date, value=min_date, key=f"{city}epod-date-start")
        with col2:
            end_date = st.date_input(
                'End Date', min_value=min_date, max_value=max_date, value=max_date, key=f"{city}epod-date-end")
        df['EPOD Name'] = df['EPOD Name'].str.replace('-', '')

        with col3:
            EPod = st.multiselect(label='Select The EPOD',
                                  options=['All'] + epods.tolist(),
                                  default='All', key=f"{city}selectepods")
        with col1:
            st.markdown(":large_green_square: T-15 fulfilled")

        with col2:
            st.markdown(":large_yellow_square: T-15 Not fulfilled")
        with col3:
            st.markdown(":large_red_square: Delay")

        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        filtered_data = df[(df['Actual Date'] >= start_date)
                           & (df['Actual Date'] <= end_date)]
        if 'All' in EPod:
            EPod = epods


        filtered_data = filtered_data[
            (filtered_data['EPOD Name'].isin(EPod))]

        filtered_data = filtered_data[
            (filtered_data['Customer Location City'] == city)]
        filtered_data_vehicle = vehicle_df[(vehicle_df['Actual Date'] >= start_date)
                                           & (vehicle_df['Actual Date'] <= end_date)]
        filtered_data_vehicle = filtered_data_vehicle[
            (filtered_data_vehicle['Customer Location City'] == city)]
        filtered_data_vehicle = filtered_data_vehicle[filtered_data_vehicle['EPOD Name'].isin(
            EPod)]
        record_count_df = filtered_data.groupby(
            ['EPOD Name', 't-15_kpi']).size().reset_index(name='Record Count')
        color_mapping = {0: 'yellow', 1: 'green', 2: 'red'}
        record_count_df['Color'] = record_count_df['t-15_kpi'].map(
            color_mapping)

        record_count_df = record_count_df.sort_values('EPOD Name')
        y_axis_range = [0, record_count_df['Record Count'].max() * 1.2]

        # Calculate average duration per EPod per session
        average_duration = filtered_data.groupby('EPOD Name')['Duration'].mean().reset_index().round(1)

        # Calculate average duration per session across all EPods
        avgdur = average_duration['Duration'].mean().round(2)


        with col5:
            st.markdown("Average Duration/EPod")
            st.markdown("<span style='font-size: 25px;line-height: 0.8;'>" +
                        str(average_duration['Duration'].mean().round(2)) + "</span>", unsafe_allow_html=True)

        filtered_data_vehicle['KM Travelled for Session'] = filtered_data_vehicle['KM Travelled for Session'].replace(
            '', np.nan)
        filtered_data_vehicle['KM Travelled for Session'] = filtered_data_vehicle['KM Travelled for Session'].astype(
            float)

        # Calculate average kilometers per EPod per session
        average_kms = filtered_data_vehicle.groupby('EPOD Name')['KM Travelled for Session'].mean().reset_index().round(
            1)

        # Calculate average kilometers per session across all EPods
        avgkm = average_kms['KM Travelled for Session'].mean().round(2)

        with col4:
            st.markdown("Average Kms/EPod per Session")
            st.markdown("<span style='font-size: 25px;line-height: 0.8;'>" +
                        str(average_kms['KM Travelled for Session'].mean().round(2)) + "</span>", unsafe_allow_html=True)

        filtered_data['KWH Pumped Per Session'] = filtered_data['KWH Pumped Per Session'].replace(
            '', np.nan)

        filtered_data = filtered_data[filtered_data['KWH Pumped Per Session'] != '#VALUE!']

        filtered_data['KWH Pumped Per Session'] = filtered_data['KWH Pumped Per Session'].astype(
            float)
        filtered_data['KWH Pumped Per Session'] = filtered_data['KWH Pumped Per Session'].abs()

        # Calculate average kWh per EPod per session
        average_kwh = filtered_data.groupby('EPOD Name')['KWH Pumped Per Session'].mean().reset_index().round(1)

        # Calculate average kWh per session across all EPods
        avgkwh = average_kwh['KWH Pumped Per Session'].mean().round(2)

        with col6:
            st.markdown("Average kWh/EPod per Session")
            st.markdown("<span style='font-size: 25px;line-height: 0.8;'>" +
                        str(average_kwh['KWH Pumped Per Session'].mean().round(2)) + "</span>", unsafe_allow_html=True)

            # Calculate the total count for each EPod
            total_count_per_epod = filtered_data.groupby('EPOD Name')['t-15_kpi'].count().reset_index(
                name='Total Count')

            # Merge the total count with record_count_df to get the denominator for percentage calculation
            record_count_df = record_count_df.merge(total_count_per_epod, on='EPOD Name')

            # Calculate the percentage for each EPod
            record_count_df['Percentage'] = (record_count_df['Record Count'] / record_count_df['Total Count']) * 100

            # Calculate the percentage of T-15 Fulfilled and T-15 Not Fulfilled for each EPod
            sla_data = record_count_df.pivot(index='EPOD Name', columns='t-15_kpi', values='Percentage').reset_index()
            sla_data['On-Time SLA'] = sla_data[0] + sla_data[1]

            max_value = max(record_count_df['Record Count'].max(), sla_data['On-Time SLA'].max()) * 1.2

            fig = go.Figure()
            # Add the bar traces for T-15 Fulfilled, T-15 Not Fulfilled, and Delay
            for color, kpi_group in record_count_df.groupby('Color'):
                fig.add_trace(go.Bar(
                    x=kpi_group['EPOD Name'],
                    y=kpi_group['Percentage'],
                    text=kpi_group['Percentage'].round(2).astype(str) + '%',
                    textposition='auto',
                    name=color,
                    marker=dict(color=color),
                    width=0.38,
                    showlegend=False
                ))

            # Add the line trace for On-Time SLA
            fig.add_trace(go.Scatter(
                x=sla_data['EPOD Name'],
                y=sla_data['On-Time SLA'],
                text=sla_data['On-Time SLA'].round(0).astype(str) + '%',  # Display On-Time SLA text
                textposition='top center',
                mode='lines+markers+text',  # Add 'text' to display the text values
                line=dict(color='purple', width=2),  # Set the line color to purple
                marker=dict(color='purple', size=8),
                name='On-Time SLA',
                yaxis='y2'  # Plot the line on the secondary y-axis
            ))

            fig.update_layout(
                xaxis={'categoryorder': 'array', 'categoryarray': record_count_df['EPOD Name']},
                yaxis={'range': [0, max_value]},
                xaxis_title='EPOD Name',
                yaxis_title='Sessions',
                yaxis2=dict(overlaying='y', side='right', showgrid=False, range=[0, max_value]),
                height=340,
                width=600,
                title="T-15 for each EPod with On-Time SLA",
                legend=dict(title_font=dict(size=14), font=dict(size=12), x=0, y=1.1, orientation='h'),
            )

            with col1:
                st.plotly_chart(fig)

        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=average_duration['EPOD Name'],
            y=average_duration['Duration'],
            text=average_duration['Duration'].round(2),
            textposition='auto',
            name='Average Duration',

        ))

        fig.update_layout(
            xaxis_title='EPOD Name',
            yaxis_title='Average Duration',
            barmode='group',
            width=600,
            height=340,
            title="Avg Duration Per EPod",

        )
        with col4:
            for i in range(1, 4):
                st.write("\n")
            st.plotly_chart(fig)

        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=average_kms['EPOD Name'],
            y=average_kms['KM Travelled for Session'],
            text=average_kms['KM Travelled for Session'].round(2),
            textposition='auto',
            name='Average KM Travelled for Session',

        ))

        fig.update_layout(
            xaxis_title='EPOD Name',
            yaxis_title='Average KM Travelled for Session',
            barmode='group',
            width=600,
            height=340,
            title="Avg KMs Per EPod",
        )
        with col1:
            st.plotly_chart(fig)

        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=average_kwh['EPOD Name'],
            y=average_kwh['KWH Pumped Per Session'],
            text=average_kwh['KWH Pumped Per Session'].round(2),
            textposition='auto',
            name='Average KWH Pumped Per Session',

        ))
        fig.update_layout(
            xaxis_title='EPOD Name',
            yaxis_title='Average KWH Pumped Per Session',
            barmode='group',
            width=600,
            height=340,
            title="Avg kWh Per EPod",
        )

        with col4:
            st.plotly_chart(fig)

            # Calculate the average sessions per EPod in the selected time period
        average_sessions_per_epod = filtered_data.groupby('EPOD Name')['t-15_kpi'].count().reset_index(
            name='Sessions Count')
        average_sessions_per_epod['Average Sessions'] = average_sessions_per_epod.groupby('EPOD Name')[
            'Sessions Count'].transform('mean')

        # Calculate the total number of sessions in the selected time period
        total_sessions = average_sessions_per_epod['Sessions Count'].sum()

        # Calculate the average sessions per EPod in percentage
        average_sessions_per_epod['Average Sessions (%)'] = (average_sessions_per_epod[
                                                                 'Average Sessions'] / total_sessions) * 100

        # Sort the data based on 'EPOD Name'
        average_sessions_per_epod = average_sessions_per_epod.sort_values('EPOD Name')

        # Create the line graph
        fig_line = go.Figure()

        fig_line.add_trace(go.Scatter(
            x=average_sessions_per_epod['EPOD Name'],
            y=average_sessions_per_epod['Average Sessions (%)'],
            mode='lines+markers+text',  # Use 'markers+text' mode to display text on data points
            line=dict(color='orange', width=2),
            marker=dict(color='orange', size=8),
            name='Average Sessions per EPod (%)',
            text=average_sessions_per_epod['Average Sessions (%)'].round(1),
            textposition='top center',
        ))

        fig_line.update_layout(
            xaxis={'categoryorder': 'array', 'categoryarray': average_sessions_per_epod['EPOD Name']},
            yaxis={'title': 'Average Sessions (%)'},
            height=400,
            width=1200,
            title="Average Sessions per EPod (%)",
            legend=dict(title_font=dict(size=14), font=dict(size=12), x=0, y=1.1, orientation='h'),
        )
        with col1:
            st.plotly_chart(fig_line)

        st.divider()
with tab4:

    min_date = df['Actual Date'].min().date()
    max_date = df['Actual Date'].max().date()
    col1, col2, col3, col4, col5, col6, col7, col8 = st.columns(8)
    with col1:
        start_date = st.date_input(
            'Start Date', min_value=min_date, max_value=max_date, value=min_date, key="ops_start_date")
    with col2:
        end_date = st.date_input(
            'End Date', min_value=min_date, max_value=max_date, value=max_date, key="ops_end_date")

    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    filtered_df = df[(df['Actual Date'] >= start_date)
                     & (df['Actual Date'] <= end_date)]

    max_sessions = filtered_df.groupby('Actual OPERATOR NAME')[
        'Actual Date'].count().reset_index()
    max_sessions.columns = ['Actual OPERATOR NAME', 'Max Sessions']



    working_days = filtered_df.groupby('Actual OPERATOR NAME')[
        'Actual Date'].nunique().reset_index()
    working_days.columns = ['Actual OPERATOR NAME', 'Working Days']

    ##########Rank Data##########
    # Clean the "Overall Score" column and convert to numeric (percentage) format
    df_rank["Overall Score"] = df_rank["Overall Score"].str.replace("%", "").astype(float)

    # Clean leading/trailing whitespaces in the "Full name" column
    df_rank["Full name"] = df_rank["Full name"].str.strip()

    # Drop rows with missing values in the "Overall Score" column
    df_rank.dropna(subset=["Overall Score"], inplace=True)

    # Find the operator with the maximum Overall Score
    max_score_row = df_rank.loc[df_rank['Overall Score'].idxmax()]
    max_score_operator = max_score_row['Full name']
    max_score_value = max_score_row['Overall Score']

    # Find the operator with the minimum Overall Score
    min_score_row = df_rank.loc[df_rank['Overall Score'].idxmin()]
    min_score_operator = min_score_row['Full name']
    min_score_value = min_score_row['Overall Score']

    with col7:
        # Display the lowest and highest scoring operators along with their scores
        st.markdown("Lowest Scoring:")
        st.markdown("<span style='font-size: 25px; line-height: 0.8;'>{}</span>".format(min_score_operator),
                    unsafe_allow_html=True)

    with col8:
        st.markdown("Highest Scoring:")
        st.markdown("<span style='font-size: 25px; line-height: 0.8;'>{}</span>".format(max_score_operator),
                    unsafe_allow_html=True)

    grouped_df = filtered_df.groupby(
        ['Actual OPERATOR NAME', 'Customer Location City']).size().reset_index()
    grouped_df.columns = ['Operator', 'City', 'Count']

    cities_to_include = df['Customer Location City'].dropna().unique()
    grouped_df = grouped_df[grouped_df['City'].isin(cities_to_include)]

    pivot_df = grouped_df.pivot(
        index='Operator', columns='City', values='Count').fillna(0)

    figure_width = 1.9
    figure_height = 6
    font_size_heatmap = 5
    font_size_labels = 4

    plt.figure(figsize=(figure_width, figure_height), facecolor='none')

    sns.heatmap(pivot_df, cmap='YlGnBu', annot=True, fmt='g', linewidths=0.5, cbar=False,
                annot_kws={'fontsize': font_size_heatmap})

    plt.title('Operator v/s Locations',
              fontsize=8, color='black')
    plt.xlabel('Customer Location City',
               fontsize=font_size_labels, color='black')
    plt.ylabel('Operator', fontsize=font_size_labels, color='black')

    plt.xticks(rotation=0, ha='center',
               fontsize=font_size_labels, color='black')
    plt.yticks(fontsize=font_size_labels, color='black')
    with col1:
        st.pyplot(plt, use_container_width=False)

    grouped_df = filtered_df.groupby(
        ['Actual OPERATOR NAME', 'Customer Location City']).size().reset_index()
    grouped_df.columns = ['Operator', 'City', 'Count']

    cities_to_include = df['Customer Location City'].dropna().unique()
    grouped_df = grouped_df[grouped_df['City'].isin(cities_to_include)]

    cities = np.append(grouped_df['City'].unique(), "All")

    with col3:
        selected_city = st.selectbox('Select City', cities)

    if selected_city == "All":
        city_df = grouped_df
    else:
        city_df = grouped_df[grouped_df['City'] == selected_city]

    grouped_df = filtered_df.groupby(
        ['Actual OPERATOR NAME', 'Customer Location City']).size().reset_index()
    grouped_df.columns = ['Operator', 'City', 'Count']
    total_sessions = city_df.groupby('Operator')['Count'].sum().reset_index()

    working_days = filtered_df.groupby('Actual OPERATOR NAME')[
        'Actual Date'].nunique().reset_index()
    working_days.columns = ['Operator', 'Working Days']

    merged_df = pd.merge(total_sessions, working_days, on='Operator')

    avg_sessions = pd.DataFrame()
    avg_sessions['Operator'] = merged_df['Operator']
    avg_sessions['Avg. Sessions'] = merged_df['Count'] / \
        merged_df['Working Days']
    avg_sessions['Avg. Sessions'] = avg_sessions['Avg. Sessions'].round(0)
    fig_sessions = go.Figure()
    fig_sessions.add_trace(go.Bar(
        x=total_sessions['Operator'],
        y=total_sessions['Count'],
        name='Total Sessions',
        text=total_sessions['Count'],
        textposition='auto',
        marker=dict(color='yellow'),
        width=0.5
    ))
    fig_sessions.add_trace(go.Bar(
        x=avg_sessions['Operator'],
        y=avg_sessions['Avg. Sessions'],
        name='Average Sessions',
        text=avg_sessions['Avg. Sessions'],
        textposition='auto',
        marker=dict(color='green'),
        width=0.38
    ))
    fig_sessions.update_layout(
        title='Total Sessions and Average Sessions per Operator',
        xaxis=dict(title='Operator'),
        yaxis=dict(title='Count / Average Sessions'),
        margin=dict(l=50, r=50, t=80, b=80),
        legend=dict(yanchor="top", y=1.1, xanchor="left",
                    x=0.01, orientation="h"),
        width=1050,
        height=500
    )
    with col4:
        for i in range(1, 10):
            st.write("\n")
        st.plotly_chart(fig_sessions)

    working_days = filtered_df.groupby('Actual OPERATOR NAME')[
        'Actual Date'].nunique().reset_index()
    working_days.columns = ['Operator', 'Working Days']

    if selected_city == "All":
        selected_working_days = working_days
    else:
        selected_working_days = working_days[working_days['Operator'].isin(
            city_df['Operator'])]

    fig_working_days = go.Figure(data=go.Bar(
        x=selected_working_days['Operator'],
        y=selected_working_days['Working Days'],
        marker=dict(color='lightgreen'),
        text=selected_working_days['Working Days']
    ))
    fig_working_days.update_layout(
        title='Working Days per Operator',
        xaxis=dict(title='Operator'),
        yaxis=dict(title='Working Days'),
        margin=dict(l=50, r=50, t=80, b=80),
        width=800,
        height=500
    )
    with col4:
        st.plotly_chart(fig_working_days)

with tab5:
    # Helper function to generate bar graph
    def generate_bar_graph(filtered_df):
        type_counts = filtered_df['type'].value_counts().reset_index()
        type_counts.columns = ['Type', 'Count']
        total_sessions = type_counts['Count'].sum()
        type_counts['Percentage'] = (type_counts['Count'] / total_sessions) * 100
        type_counts['Percentage'] = type_counts['Percentage'].round(2)
        fig = px.bar(type_counts, x='Type', y='Percentage', text='Percentage',
                     labels={'Type': 'Subscription', 'Percentage': 'Percentage'}, width=525, height=525,
                     title='Total Sessions by Subscription Type')
        fig.update_layout(xaxis=dict(tickangle=-45))
        fig.update_traces(textposition='outside')
        return fig


    # Data preprocessing
    df['type'] = df['type'].str.replace('-', '')
    min_date = df['Actual Date'].min().date()
    max_date = df['Actual Date'].max().date()

    # Streamlit UI
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Overall")
    with col2:
        start_date = st.date_input(
            'Start Date', min_value=min_date, max_value=max_date, value=min_date, key="sub_start_date")
    with col3:
        end_date = st.date_input(
            'End Date', min_value=min_date, max_value=max_date, value=max_date, key="sub_end_date")

    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    filtered_df = df[(df['Actual Date'] >= start_date) & (df['Actual Date'] <= end_date)]

    # Generate and display the bar graph
    bar_graph = generate_bar_graph(filtered_df)
    bar_graph.update_layout(width=400, height=490)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.plotly_chart(bar_graph)
        st.write("\n")

    # Data cleaning and analysis for KWH Pumped Per Session
    filtered_df['KWH Pumped Per Session'] = filtered_df['KWH Pumped Per Session'].replace('', np.nan)
    filtered_df = filtered_df[filtered_df['KWH Pumped Per Session'] != '#VALUE!']
    filtered_df['KWH Pumped Per Session'] = filtered_df['KWH Pumped Per Session'].astype(float)
    filtered_df['KWH Pumped Per Session'] = filtered_df['KWH Pumped Per Session'].abs()
    average_kwh = filtered_df.groupby('type')['KWH Pumped Per Session'].mean().reset_index().round(1)

    # Generate and display the bar graph for Average kWh Pumped Per Session
    fig = go.Figure(
        data=[go.Bar(x=average_kwh['type'], y=average_kwh['KWH Pumped Per Session'],
                     text=average_kwh['KWH Pumped Per Session'], textposition='outside')])
    fig.update_layout(xaxis_title='Subscription', yaxis_title='Average kWh Pumped',
                      title='Average kWh Pumped Per Session by Subscription Type', width=400, height=490,
                      xaxis=dict(tickangle=-45))
    with col2:
        st.plotly_chart(fig)
        st.write("\n")

    # Data cleaning and analysis for Average Duration Per Session
    average_duration = filtered_df.groupby('type')['Duration'].mean().reset_index().round(1)

    # Generate and display the bar graph for Average Duration Per Session
    fig = go.Figure(
        data=[go.Bar(x=average_duration['type'], y=average_duration['Duration'], text=average_duration['Duration'],
                     textposition='outside')])
    fig.update_layout(xaxis_title='Subscription', yaxis_title='Avg Duration per Session',
                      title='Average Duration Per Session by Subscription Type', width=400, height=490,
                      xaxis=dict(tickangle=-45))
    with col3:
        st.plotly_chart(fig)
        st.write("\n")



    for city in filtered_df['Customer Location City'].dropna().unique():
        col1, col2, col3 = st.columns(3)
        with col1:
            for i in range(1, 4):
                st.write("\n")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader(city)
        with col2:
            start_date = st.date_input(
                'Start Date', min_value=min_date, max_value=max_date, value=min_date, key=f"{city}sub_start_date")
        with col3:
            end_date = st.date_input(
                'End Date', min_value=min_date, max_value=max_date, value=max_date, key=f"{city}sub_end_date")



        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)

        filtered_df = df[(df['Actual Date'] >= start_date) & (df['Actual Date'] <= end_date)]
        filtered_df = filtered_df[filtered_df['Customer Location City'] == city]

        # Generate and display the bar graph for each city
        bar_graph = generate_bar_graph(filtered_df)
        bar_graph.update_layout(width=400, height=490)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.plotly_chart(bar_graph)
            st.write("\n")

        # Data cleaning and analysis for KWH Pumped Per Session for each city
        filtered_df['KWH Pumped Per Session'] = filtered_df['KWH Pumped Per Session'].replace('', np.nan)
        filtered_df = filtered_df[filtered_df['KWH Pumped Per Session'] != '#VALUE!']
        filtered_df['KWH Pumped Per Session'] = filtered_df['KWH Pumped Per Session'].astype(float)
        filtered_df['KWH Pumped Per Session'] = filtered_df['KWH Pumped Per Session'].abs()
        average_kwh = filtered_df.groupby('type')['KWH Pumped Per Session'].mean().reset_index().round(1)

        # Generate and display the bar graph for Average kWh Pumped Per Session for each city
        fig = go.Figure(
            data=[go.Bar(x=average_kwh['type'], y=average_kwh['KWH Pumped Per Session'],
                         text=average_kwh['KWH Pumped Per Session'], textposition='outside')])
        fig.update_layout(xaxis_title='Subscription', yaxis_title='Average kWh Pumped',
                          title='Average kWh Pumped Per Session by Subscription Type', width=400, height=490,
                          xaxis=dict(tickangle=-45))
        with col2:
            st.plotly_chart(fig)
            st.write("\n")

        # Data cleaning and analysis for Average Duration Per Session for each city
        average_duration = filtered_df.groupby('type')['Duration'].mean().reset_index().round(1)

        # Generate and display the bar graph for Average Duration Per Session for each city
        fig = go.Figure(
            data=[go.Bar(x=average_duration['type'], y=average_duration['Duration'], text=average_duration['Duration'],
                         textposition='outside')])
        fig.update_layout(xaxis_title='Subscription', yaxis_title='Avg Duration per Session',
                          title='Average Duration Per Session by Subscription Type', width=400, height=490,
                          xaxis=dict(tickangle=-45))
        with col3:
            st.plotly_chart(fig)
            st.write("\n")


with tab6:
    # UI for selecting date range, Customers, and Subscriptions
    col1, col2, col3, col4, col5, col6, col7 = st.columns(7)

    with col1:
        df['Actual Date'] = pd.to_datetime(df['Actual Date'], errors='coerce')
        min_date = df['Actual Date'].min().date()
        max_date = df['Actual Date'].max().date()
        start_date = st.date_input(
            'Start Date', min_value=min_date, max_value=max_date, value=min_date, key="cpi-date-start-input")

    with col2:
        end_date = st.date_input(
            'End Date', min_value=min_date, max_value=max_date, value=max_date, key="cpi-date-end-input")

    with col4:
        selected_customers = st.multiselect(
            label='Select Customers',
            options=['All'] + df['Customer Name'].unique().tolist(),
            default='All'
        )

    with col3:
        selected_subscriptions = st.multiselect(
            label='Select Subscription',
            options=['All'] + df['type'].unique().tolist(),
            default='All'
        )

    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    filtered_data = df[(df['Actual Date'] >= start_date) & (df['Actual Date'] <= end_date)]

    # Filter the data based on Customer and Subscription selection
    if 'All' in selected_customers:
        selected_customers = df['Customer Name'].unique()
    if 'All' in selected_subscriptions:
        selected_subscriptions = df['type'].unique()

    filtered_data = filtered_data[
        (filtered_data['Customer Name'].isin(selected_customers)) & (filtered_data['type'].isin(selected_subscriptions))
        ]

    unique_types = filtered_data["type"].unique()
    type_colors = {type_: f"#{hash(type_) % 16777215:06x}" for type_ in unique_types}

    # Create a Streamlit map using folium
    st.write("### Subscription Wise Geographical Insights")
    m = folium.Map(location=[filtered_data['location.lat'].mean(), filtered_data['location.long'].mean()],
                   zoom_start=10)


    # Define a function to generate the HTML table for the popup
    def generate_popup_table(row):
        columns_to_show = [
            "Actual Date", "Customer Name", "EPOD Name", "Actual OPERATOR NAME", "Duration",
            "Day", "E-pod Arrival Time @ Session location", "Actual SoC_Start", "Actual Soc_End",
            "Booking Session time", "Customer Location City", "canceled", "cancelledPenalty",
            "t-15_kpi", "KWH Pumped Per Session"
        ]
        table_html = "<table style='border-collapse: collapse;'>"
        for col in columns_to_show:
            table_html += f"<tr><td style='border: 1px solid black; padding: 5px;'><strong>{col}</strong></td><td style='border: 1px solid black; padding: 5px;'>{row[col]}</td></tr>"
        table_html += "</table>"
        return table_html


    # Add circle markers for each location with different colors based on the type
    for index, row in filtered_data.iterrows():
        location_name = row["type"]
        longitude = row["location.long"]
        latitude = row["location.lat"]
        color = type_colors[location_name]

        # Creating the popup content with a table
        popup_html = f"""
                <strong>{location_name}</strong><br>
                Latitude: {latitude}<br>
                Longitude: {longitude}<br>
                {generate_popup_table(row)}
                """

        folium.CircleMarker(
            location=[latitude, longitude],
            radius=5,
            popup=folium.Popup(popup_html, max_width=400),  # Use the created popup content
            color=color,
            fill=True,
            fill_color=color,
        ).add_to(m)

    # Calculate and display the most and least subscription types ("type") in columns
    with col6:
        most_subscribed_type = filtered_data['type'].value_counts().idxmax()
        st.markdown("Most Subscribed Type")
        st.markdown("<span style='font-size: 25px; line-height: 0.7;'>{}</span>".format(most_subscribed_type),
                    unsafe_allow_html=True)

    with col7:
        least_subscribed_type = filtered_data['type'].value_counts().idxmin()
        st.markdown("Least Subscribed Type")
        st.markdown("<span style='font-size: 25px; line-height: 0.7;'>{}</span>".format(least_subscribed_type),
                    unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        # Display the map using folium_static
        folium_static(m)

    with col3:
        # Display the custom legend with the color scale
        legend_items = [(type_, color) for type_, color in type_colors.items()]

        # Calculate the split point for two columns
        split_point = len(legend_items) // 2

        # Split the legend items into two lists for two columns
        column1 = legend_items[:split_point]
        column2 = legend_items[split_point:]

        # Display the legend items in two columns
        col1, col2 = st.columns(2)

        with col1:
            for type_, color in column1:
                st.markdown(
                    f'<i style="background:{color}; width: 8px; height: 8px; display:inline-block;"></i> {type_}',
                    unsafe_allow_html=True)

        with col2:
            for type_, color in column2:
                st.markdown(
                    f'<i style="background:{color}; width: 8px; height: 8px; display:inline-block;"></i> {type_}',
                    unsafe_allow_html=True)
