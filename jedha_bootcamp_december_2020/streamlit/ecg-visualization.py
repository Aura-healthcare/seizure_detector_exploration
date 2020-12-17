import streamlit as st
import pandas as pd
import numpy as np
import wfdb
import scaleogram as scg 
import re

st.title('ECG Visualization from https://physionet.org/')

record_list = wfdb.get_record_list('butqdb')
record_final_list=[]
for rcd in record_list:
    if re.search(r'ECG$', rcd):
        record_final_list.append(rcd.split('/')[0])

chosen_record = st.sidebar.selectbox(
   'Which record ?',
   record_final_list
)

time_lap =  list(range(0, 1500))
chosen_time = st.sidebar.selectbox(
   'Which start time (min) ?',
   time_lap
)
frequency = [1000,500,250]
#chosen_frequency = st.sidebar.selectbox(
#   'Which frequency (Hz) ?',
#   frequency
#)
secondstart = chosen_time * 10000
sampfromvalue,  samptovalue = st.slider('Select a range of values', secondstart, secondstart + 10000, (secondstart, secondstart + 2000))
#pn_dir='butqdb'.
@st.cache
def load_data():
    record = wfdb.rdrecord(f"{chosen_record}_ECG", pn_dir=f"butqdb/{chosen_record}/", sampfrom=sampfromvalue, sampto=samptovalue)
    return record.adc()[:,0]

data_load_state = st.text('Loading data...')
data = load_data()
data_load_state.text("Done! (using st.cache)")

if st.checkbox('Show raw data'):
    st.subheader('Raw data')
    st.write(data)

st.subheader(f"ECG record : {chosen_record}")

st.line_chart(data)

if samptovalue-sampfromvalue > 2000:
    st.text('No scaleogram range to long')
else : 
    # choose default wavelet function 
    scg.set_default_wavelet('morl')

    signal_length = samptovalue-sampfromvalue
    # range of scales to perform the transform
    scales = scg.periods2scales( np.arange(1, signal_length+1) )
    x_values_wvt_arr = range(0,len(data),1)

    # the scaleogram
    fig = scg.cws(data, scales=scales, figsize=(10, 4.0), coi = False, ylabel="Period", xlabel="Time")
    #st.plotly_chart(scal)
    #coeff, freq = pywt.cwt(data, 500 , 'morl', 1)
    st.pyplot(fig.figure)


