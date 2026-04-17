import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from scipy.signal import savgol_filter

import lumin.Z_deconv_oasis as oasis
from lumin.Z_deconv_oasis import preprocess



def spike(cell_properties_df: pd.DataFrame = None, prominence: float = None, amplitude_width_ratio: float = 0, imaging_interval: float = None, start_frame: int=None, end_frame: int=None, smoothing: bool = False,
    method: str = "find_peaks", tau: float = 1.0, fs: float = 1.0): #spike deconvonvolution parameter 
    smoothed_traces, filtered_peak_list,  amplitude_list,  prominence_list, width_list, rise_time_list, decay_time_list, low_quality_peak_list = [],[],[], [], [], [],[], []
    
    if method == "find_peaks":
        for index, cell in cell_properties_df.iterrows():
            trace = cell['dff']

            if smoothing == True:
                trace = savgol_filter(trace, 3, 1)
                smoothed_traces.append(trace)

            if end_frame is not None:
                if end_frame < 0: end_frame = None

            if start_frame is None:
                start_frame = 0
            if end_frame is None:
                end_frame = len(trace)  
    

            window_trace = trace[start_frame:end_frame]
            #print('trace length spike calling:', len(window_trace))
            #print(start_frame, end_frame)

            peak_location, peak_properties = find_peaks(trace, prominence = prominence, width=0, height=0)
            peak_properties.update({'peak_location':peak_location})
            peak_properties_df = pd.DataFrame(peak_properties)
            peak_properties_df = peak_properties_df[(peak_properties_df.peak_location >= start_frame) & (peak_properties_df.peak_location < end_frame)]
            peak_properties_df['amplitude'] = [trace[i] for i in peak_properties_df['peak_location']]
            peak_properties_df['amplitude_width_ratio'] = peak_properties_df['amplitude'] / peak_properties_df['widths']
            peak_properties_df['pass_qc']  = peak_properties_df['amplitude_width_ratio'] > amplitude_width_ratio
            peak_properties_df['rise_time'] = peak_properties_df['peak_location'] - peak_properties_df['left_ips']
            peak_properties_df['decay_time'] = peak_properties_df['right_ips'] - peak_properties_df['peak_location']
            low_quality_peaks = peak_properties_df[peak_properties_df['pass_qc'] == False].peak_location.tolist()
            
            low_quality_peak_list.append(low_quality_peaks)

            indices = [peak_properties_df['peak_location'].tolist().index(x) for x in peak_properties_df['peak_location'].tolist() if x in low_quality_peaks]
            
            filtered_peak_list.append(list(np.delete(peak_properties_df['peak_location'].to_numpy(), indices)))
            amplitude_list.append(list(np.delete(peak_properties_df['amplitude'].to_numpy(), indices)))
            prominence_list.append(list(np.delete(peak_properties_df['prominences'].to_numpy(), indices)))
            width_list.append(list(np.delete(peak_properties_df['widths'].to_numpy(), indices)))
            rise_time_list.append(list(np.delete(peak_properties_df['rise_time'].to_numpy(), indices)))
            decay_time_list.append(list(np.delete(peak_properties_df['decay_time'].to_numpy(), indices)))
        
        total_time_min = len(window_trace) * imaging_interval / 60

        cell_properties_df['peak_location'] = filtered_peak_list
        cell_properties_df['frequency'] = [len(x) / total_time_min for x in filtered_peak_list]
        cell_properties_df['amplitude'] = [np.mean(x) if len(x) != 0 else 0 for x in amplitude_list ]
        cell_properties_df['prominence'] = [np.mean(x) if len(x) != 0 else 0 for x in prominence_list ]
        cell_properties_df['width'] = [np.mean(x) * imaging_interval if len(x) != 0 else 0 for x in width_list ]
        cell_properties_df['rise_time'] = [np.mean(x) * imaging_interval if len(x) != 0 else 0 for x in rise_time_list ]
        cell_properties_df['decay_time'] = [np.mean(x) * imaging_interval if len(x) != 0 else 0 for x in decay_time_list ]
        cell_properties_df['low_quality_peaks'] = low_quality_peak_list

        cell_properties_df["response"] = np.where(cell_properties_df["frequency"] > 0, "active", "inactive")
        
        if smoothing==True: cell_properties_df['dff_smoothed'] = smoothed_traces

        return cell_properties_df, start_frame, end_frame
        
    # OASIS deconvolution
    elif method == "oasis":

        filtered_peak_list = []
        amplitude_list = []

        for index, cell in cell_properties_df.iterrows():
            trace = cell['dff']

            if start_frame is None:
                start_frame = 0
            if end_frame is None:
                end_frame = len(trace)

            window_trace = trace[start_frame:end_frame]

            F = np.expand_dims(window_trace, axis=0)

            F = preprocess(
                F,
                baseline="maximin",
                win_baseline=10,
                sig_baseline=2,
                fs=fs
            )

            S = oasis(F, batch_size=1, tau=tau, fs=fs)
            spike_trace = S[0]

            peak_locations = np.where(spike_trace > 0)[0]

            # threshold
            peak_locations = peak_locations[spike_trace[peak_locations] > 0.1]

            # back to global indices
            peak_locations = peak_locations + start_frame

            filtered_peak_list.append(peak_locations.tolist())
            amplitude_list.append([trace[i] for i in peak_locations])
        
        cell_properties_df['peak_location'] = filtered_peak_list
        cell_properties_df['amplitude'] = [np.mean(x) if len(x)>0 else 0 for x in amplitude_list]
        cell_properties_df['frequency'] = [len(x) for x in filtered_peak_list]

    return cell_properties_df, start_frame, end_frame

def baseline_change(cell_properties_df: pd.DataFrame = None,  control_condition: str = None,  std_threshold: float = None ):

    std_dict, mean_dict, response_l = {},{}, []

    for replicate in cell_properties_df.biological_replicate.unique():
        mean_dict[replicate] = cell_properties_df[(cell_properties_df.biological_replicate == replicate) & (cell_properties_df.stimulation == control_condition)].AUC.mean()
        std_dict[replicate] = cell_properties_df[(cell_properties_df.biological_replicate == replicate) & (cell_properties_df.stimulation == control_condition)].AUC.std()

    for index, row in cell_properties_df.iterrows():
        if mean_dict[row.biological_replicate] + std_threshold * std_dict[row.biological_replicate] < row.AUC:
            response_l.append('above')

        elif mean_dict[row.biological_replicate] - std_threshold * std_dict[row.biological_replicate] > row.AUC:
            response_l.append('below')

        else: response_l.append('no response')

    cell_properties_df['response'] = response_l

    return cell_properties_df

def peak_calling(filename,  stimulation,biological_replicate, image_id, label_id, dff_traces, param_prominence_threshold, param_amplitude_width_ratio, output_folder):

    filtering = True
    peak_properties_dict = {}
    peak_properties_all_dict = {}

    
    # Call all peaks in the data and plot
    for cell_idx,ca_trace in enumerate(dff_traces.T):

        #print(ca_trace)
        #print(type(ca_trace))
        #print(ca_trace.shape)
        #print(type(param_prominence_threshold))

        amplitude_list, low_quality_peak_list = [],[]

        peak_locations, peak_properties = find_peaks(ca_trace, prominence = param_prominence_threshold,width=0, height=0)
            
        peak_properties.update({'peak_locations':peak_locations})

        rise_time = [peak_locations[i] - peak_properties['left_ips'][i] for i in range(0,len(peak_locations))]
        decay_time = [peak_properties['right_ips'][i] - peak_locations[i] for i in range(0,len(peak_locations))]
        
        for i in peak_locations:
            amplitude_list.append(ca_trace[i])
    
        peak_properties.update({'peak_amplitudes':amplitude_list})
        peak_properties.update({'peak_rise_time':rise_time})
        peak_properties.update({'peak_decay_time':decay_time})
    
        for i in range(0, len(amplitude_list)):
            if float(peak_properties['peak_amplitudes'][i] / peak_properties['widths'][i]) < param_amplitude_width_ratio:
                low_quality_peak_list.append(peak_locations[i])

        peak_properties_all = peak_properties.copy()
        peak_properties_all.update({'low_quality_peaks':low_quality_peak_list})
        peak_properties_all_dict.update({cell_idx:peak_properties_all}) 
    
        # Filter low quality peaks
        if filtering == True:
            properties_to_remove_idx = np.intersect1d(peak_locations,np.array(low_quality_peak_list),return_indices=True)[1]
            if len(properties_to_remove_idx) > 0:

                for key in peak_properties.keys():
                    peak_properties[key] = np.delete(peak_properties[key],properties_to_remove_idx)
    
        peak_properties.update({'cell_id':label_id[cell_idx]})

        peak_properties_dict.update({cell_idx:peak_properties}) 

