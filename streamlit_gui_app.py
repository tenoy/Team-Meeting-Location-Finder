import streamlit as st
import pandas as pd
import pydeck as pdk
import datetime
from location_model_optimization import LocationModelOptimization


@st.cache(suppress_st_warning=True)
def load_target_trainstations():
    df = pd.read_csv('D_Bahnhof_2020_target_prep.csv', sep=';')
    return df


@st.cache(suppress_st_warning=True)
def load_start_trainstations():
    df = pd.read_csv('D_Bahnhof_2020_start_prep.csv', sep=';')
    return df


def calculate(start_trainstations, target_trainstations):
    print(st.session_state.optimization_mode)
    if st.session_state.member_location_selectbox is None or len(st.session_state.member_location_selectbox) < 2:
        st.error('Please provide at least 2 starting locations!')
    elif len(st.session_state.member_location_selectbox) > 10:
        st.error('This demo only allows up to 10 team member locations as input')
    elif len(st.session_state.target_locations_selectbox) == 1 or len(st.session_state.target_locations_selectbox) > 10:
            st.error('Please provide either no target location candidates or 2 to 10 target location candidates')
    else:
        locationmodel = LocationModelOptimization(start_trainstations, 
                                                    target_trainstations, 
                                                    st.session_state.member_location_selectbox,
                                                    st.session_state.target_locations_selectbox, 
                                                    st.session_state.blocklist_location_selectbox, 
                                                    st.session_state.date_picker, 
                                                    st.session_state.time_picker,
                                                    st.session_state.optimization_mode,
                                                    st.secrets['google_api_key'])
        locationmodel.solve_and_print()
        st.session_state.member_location_df = locationmodel.member_location_df
        st.session_state.best_location_name = locationmodel.best_location_name
        st.session_state.best_location_df = locationmodel.best_location_df
        st.session_state.travel_times_to_best_location_df = locationmodel.travel_times_to_best_location_df

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


def main():
    # Initialize session state variables
    if 'best_location_name' not in st.session_state:
        st.session_state.best_location_name = 'None'

    if 'best_location_df' not in st.session_state:
        st.session_state.best_location_df = None

    if 'best_location_layer' not in st.session_state:
        st.session_state.best_location_layer = None

    if 'member_location_df' not in st.session_state:
        st.session_state.member_location_df = None
    
    if 'target_location_layer' not in st.session_state:
        st.session_state.target_location_layer = None

    if 'member_location_layer' not in st.session_state:
        st.session_state.member_location_layer = None

    if 'travel_times_to_best_location_df' not in st.session_state:
        st.session_state.travel_times_to_best_location_df = pd.DataFrame({'Starting location' : [], 'Travel duration (hh:mm:ss)' : []})
    
    # Description of the tool
    st.title('Team meeting location finder')
    st.write('This tool helps to find the \'best\' location for a team meeting in Germany, to which each team member travels via train. Simply select the starting locations in the \'Add team member starting locations\' dropdown menu, select a date and time for the meeting and click \'Calculate best location for team meeting!\'. Optionally, you can define your own meeting location candidates via \'Add meeting location candidates\', otherwise a default set of candidates is used.')
    st.write('By default, the \'best\' location is defined to be the location where the overall travel time is minimized. You can change to minimize the maximum travel for each member time via the radio buttons in the advanced settings tab. You can also exclude locations from the default set you don\'t like there.')
    st.write('The red dots on the map represent the meeting locations candidates. After calculation, a blue dot appears that represents the \'best\' team meeting location. The black dots represent the starting locations of each team member. Below the map you can find the name of the best meeting location and the travel time duration for each starting location is also given.')

    # Load all possible start and target trainstations
    start_trainstations = load_start_trainstations()
    target_trainstations = load_target_trainstations()

    # Form for user input
    with st.form('calc_location_form'):
        st.multiselect('Select starting locations', start_trainstations , key='member_location_selectbox')
        st.multiselect('Select meeting location candidates (optional)', start_trainstations, key='target_locations_selectbox')
        st.caption('If left empty, default meeting location candidates are used.')
        col1, col2 = st.columns(2)
        with col1:
            min_date = datetime.date.today()
            st.date_input(label = 'Date of meeting', value=min_date, min_value=min_date, key='date_picker')
        with col2:
            st.time_input(label = 'Time of meeting', value=datetime.time(12, 0), key='time_picker')
        with st.expander('Advanced Settings'):
            st.radio(label='Select objective', options=('Minimize total travel time', 'Minimize maximum travel time'), index=0, key='optimization_mode', horizontal=True)
            st.write('Exclude locations from default meeting locations')
            st.multiselect('Select a trainstation', target_trainstations , key='blocklist_location_selectbox')
            st.caption('The exclusions are ignored if meeting location candidates are selected')
        st.form_submit_button('Calculate best location for team meeting!', on_click=calculate, args=(start_trainstations, target_trainstations, ))

    # Initialize pydeck object for map visualization
    deck=pdk.Deck(
     map_style='mapbox://styles/mapbox/light-v9',
     initial_view_state=pdk.ViewState(
         latitude=51.25,
         longitude=10.25,
         zoom=4.9,
         pitch=0,
     ),
     layers=[
     ]
    )
    # Update target trainstations if a target stations are blocklisted
    if len(st.session_state.blocklist_location_selectbox) > 0:
        target_trainstations = target_trainstations[~target_trainstations['NAME'].isin(st.session_state.blocklist_location_selectbox)]

    # Update target trainstations if a custom target candidate set is specified
    if len(st.session_state.target_locations_selectbox) > 1:
        target_trainstations = start_trainstations[start_trainstations['NAME'].isin(st.session_state.target_locations_selectbox)]

    # Use pydeck ScatterplotLayer to visualize target trainstation candidates
    st.session_state.target_location_layer = pdk.Layer(
            'ScatterplotLayer',
            data=target_trainstations,
            get_position='[lng, lat]',
            get_color='[200, 30, 0, 160]',
            get_radius=7500
        )

    # Append layers to the pydeck object
    deck.layers.append(st.session_state.target_location_layer)
    # Member locations and best locations are added after calculation to deck
    deck.layers.append(st.session_state.member_location_layer)
    deck.layers.append(st.session_state.best_location_layer)

    # Visualize via pydeck_chart method of streamlit
    st.pydeck_chart(pydeck_obj=deck, use_container_width=False)
    st.write('Best location: ', st.session_state.best_location_name)
    st.dataframe(data=st.session_state.travel_times_to_best_location_df)
    st.caption('The travel times used for optimization and display are real-time values from the Google Distance Matrix API. Due to the restricted monthly query budget of this demo, a calculation may not always be possible.')
    

if __name__ == '__main__':
    main()
