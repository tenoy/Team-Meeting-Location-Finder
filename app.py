import streamlit as st
import pandas as pd
import numpy as np
import geopandas
from optimization import LocationModel
import pydeck as pdk
import datetime

@st.cache(suppress_st_warning=True)
def load_target_trainstations():
    #time.sleep(3)
    #st.write("No cached data!")
    df = pd.read_csv('D_Bahnhof_2020_target_prep.csv', sep=';')
    return df

@st.cache(suppress_st_warning=True)
def load_start_trainstations():
    #time.sleep(3)
    #st.write("No cached data!")
    df = pd.read_csv('D_Bahnhof_2020_start_prep.csv', sep=';')
    return df

def calculate(start_trainstations, target_trainstations):
    print("calculate")
    print(st.session_state.optimization_mode)
    if st.session_state.member_location_selectbox is None or len(st.session_state.member_location_selectbox) < 2:
        st.error("Please provide at least 2 starting locations!")
    elif len(st.session_state.member_location_selectbox) > 10:
        st.error("This demo only allows up to 10 team member locations as input")
    else:
        locationmodel = LocationModel(start_trainstations, 
                                        target_trainstations, 
                                        st.session_state.member_location_selectbox, 
                                        st.session_state.blacklist_location_selectbox, 
                                        st.session_state.date_picker, 
                                        st.session_state.time_picker,
                                        st.session_state.optimization_mode,
                                        st.secrets["google_api_key"])
        locationmodel.solve_and_print()
        st.session_state.member_location_df = locationmodel.member_location_df
        st.session_state.best_location_name = locationmodel.best_location_name
        st.session_state.best_location_df = locationmodel.best_location_df
        st.session_state.travel_times_to_best_location_df = locationmodel.travel_times_to_best_location_df
        print(locationmodel.best_location_df)

        st.session_state.best_location_layer = pdk.Layer(
            'ScatterplotLayer',
            data=locationmodel.best_location_df,
            get_position='[lng, lat]',
            get_color='[1, 1, 255]',
            get_radius=15000
        )
        st.session_state.member_location_layer = pdk.Layer(
            'ScatterplotLayer',
            data=locationmodel.member_location_df,
            get_position='[lng, lat]',
            get_fill_color='[0, 0, 0]',
            get_radius=15000
        )

def extract_geom_lines(input_geom):
        if (input_geom is None) or (input_geom is np.nan):
            return []
        else:
            if input_geom.type[:len('multi')].lower() == 'multi':
                full_coord_list = []
                for geom_part in input_geom.geoms:
                    geom_part_2d_coords = [[coord[0],coord[1]] for coord in list(geom_part.coords)]
                    full_coord_list.append(geom_part_2d_coords)
            else:
                full_coord_list = [[coord[0],coord[1]] for coord in list(input_geom.coords)]
            return full_coord_list

def main():

    if 'best_location_name' not in st.session_state:
        st.session_state.best_location_name = "None"

    if 'best_location_df' not in st.session_state:
        st.session_state.best_location_df = None

    if 'member_location_df' not in st.session_state:
        st.session_state.member_location_df = None
    
    if 'best_location_layer' not in st.session_state:
        st.session_state.best_location_layer = None

    if 'member_location_layer' not in st.session_state:
        st.session_state.member_location_layer = None

    if 'travel_times_to_best_location_df' not in st.session_state:
        st.session_state.travel_times_to_best_location_df = pd.DataFrame({'Starting location' : [], 'Travel duration (hh:mm:ss)' : []})
    

    st.title("Team meeting location finder")
    st.write("This tool helps to find the 'best' location for a team meeting in Germany, to which each team member travels via train. Simply input the starting train stations of each team member, select a date and time for the meeting and click 'Calculate best location for team meeting!'.")
    st.write("By default, the 'best' location is defined to be the location where the overall travel time is minimized. You can change to minimize the maximum travel for each member time via the radio buttons in the advanced settings tab. You can also exclude locations you don't like there.")
    st.write("The red dots on the map represent the set of possible meeting locations. After calculation, a blue dot appears that represents the 'best' team meeting location. The black dots represent the starting locations of each team member. Below the map you can find the name of the best meeting location and the travel time duration of each member is also given.")

    start_trainstations = load_start_trainstations()
    target_trainstations = load_target_trainstations()

    with st.form("calc_location_form"):
        st.write("Add team member starting location(s)")
        st.multiselect('Select a city / trainstation', start_trainstations , key="member_location_selectbox")
        col1, col2 = st.columns(2)
        with col1:
            min_date = datetime.date.today() + datetime.timedelta(days=1)
            st.date_input(label = "Date of meeting", value=min_date, min_value=min_date, key="date_picker")
        with col2:
            st.time_input(label = "Time of meeting", value=datetime.time(12, 0), key="time_picker")
        with st.expander("Advanced Settings"):
            st.radio(label="Select minimization objective", options=('Total travel time', 'Maximum travel time'), index=0, key="optimization_mode", horizontal=True)
            st.write("Exclude location(s)")
            st.multiselect('Select a city / trainstation', target_trainstations , key="blacklist_location_selectbox")
        st.form_submit_button("Calculate best location for team meeting!", on_click=calculate, args=(start_trainstations, target_trainstations, ))

    #map visualization of railroad network
    #db_geo = geopandas.read_file('strecken_polyline.shp')
    #print(db_geo)

    #Extract lines from shp
    #db_geo['coord_list'] = db_geo['geometry'].apply(extract_geom_lines)
    #print(db_geo)

    #use pydeck for map viz
    deck=pdk.Deck(
     map_style='mapbox://styles/mapbox/light-v9',
     initial_view_state=pdk.ViewState(
         latitude=51.25,
         longitude=10.25,
         zoom=4.9,
         pitch=0,
     ),
     layers=[
         
         pdk.Layer(
             'ScatterplotLayer',
             data=target_trainstations,
             get_position='[lng, lat]',
             get_color='[200, 30, 0, 160]',
             get_radius=7500
         ),
        #  pdk.Layer(
        #      'PathLayer',
        #      data=db_geo,
        #      pickable=True,
        #      get_color='[200, 30, 0, 160]',
        #      width_scale=20,
        #      width_min_pixels=2,
        #      get_path="coord_list",
        #      get_width=5,
        #  ),
     ]
    )
    deck.layers.append(st.session_state.member_location_layer)
    deck.layers.append(st.session_state.best_location_layer)

    st.pydeck_chart(pydeck_obj=deck, use_container_width=False)
    st.write('Best location: ', st.session_state.best_location_name)
    st.dataframe(data=st.session_state.travel_times_to_best_location_df)
    st.caption('The travel times used for optimization and display are real time values from the Google Distance Matrix API. Due to the restricted monthly query budget of this demo, a calculation may not always be possible.')
    


    

if __name__ == "__main__":
    main()
