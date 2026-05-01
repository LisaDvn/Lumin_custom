compute_connectivity(spike_trains)
compute_coactivity_matrix(spike_trains)
build_network_graph(connectivity_matrix)

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

