import numpy as np

def comp_costs(b_accepted, b_accepted_events, b_ht_count, b_both_count, b_as_count, bnjets, ht_weight=1, as_weight=4):

    bnjets_reshaped = bnjets[:, None, None] 

    # Trigger path cost: average nJets over accepted events
    b_Ecomp_cost = ((b_accepted * bnjets_reshaped).sum(axis=0)) / (b_accepted_events + 1e-10)

    # Event-level cost: weighted counts per accepted event
    b_Tcomp_cost = (ht_weight * (b_ht_count - b_both_count) + as_weight * b_as_count) / (b_accepted_events + 1e-10)

    return b_Ecomp_cost, b_Tcomp_cost

def V1(bht, sht1, sht2, bas, sas1, sas2):

    MAX = np.percentile(bht,99.99)
    
    ht_vals = np.linspace(np.percentile(bht,0.01), MAX, 100)
    #print('bht min and max: ', np.percentile(bht,0.01), MAX)
    

    MAX = np.percentile(bas,99.999)
    
    #print('bas min and max: ', np.percentile(bas,0.01), MAX)
    
    as_vals = np.linspace(np.percentile(bas,0.01), MAX, 100)

    HT, AS = np.meshgrid(ht_vals, as_vals, indexing='ij') 
    
    
    s1_accepted_ht = (sht1[:, None, None] >= HT[None, :, :])   # shape: (N_signal, n_ht, n_as)
    s1_accepted_as = (sas1[:, None, None] >= AS[None, :, :])     # shape: (N_signal, n_ht, n_as)

    s2_accepted_ht = (sht2[:, None, None] >= HT[None, :, :])   # shape: (N_signal, n_ht, n_as)
    s2_accepted_as = (sas2[:, None, None] >= AS[None, :, :])     # shape: (N_signal, n_ht, n_as)
    

    s1_ht_count = s1_accepted_ht.sum(axis=0)      # shape: (n_ht, n_as)
    s1_as_count = s1_accepted_as.sum(axis=0)        # shape: (n_ht, n_as)
    s1_both_count = (s1_accepted_ht & s1_accepted_as).sum(axis=0)  # events passing both

    s2_ht_count = s2_accepted_ht.sum(axis=0)      # shape: (n_ht, n_as)
    s2_as_count = s2_accepted_as.sum(axis=0)        # shape: (n_ht, n_as)
    s2_both_count = (s2_accepted_ht & s2_accepted_as).sum(axis=0)  # events passing both
    
    # Total signal accepted events = ht + as - both 
    s1_accepted_events = s1_ht_count + s1_as_count - s1_both_count
    r1_s = 100 * s1_accepted_events / sht1.shape[0]
    
    s2_accepted_events = s2_ht_count + s2_as_count - s2_both_count
    r2_s = 100 * s2_accepted_events / sht2.shape[0]

    total_s_accepted_events = s2_accepted_events + s1_accepted_events
    total_s_rate = 100 * (total_s_accepted_events)/ (sht1.shape[0] + sht2.shape[0])
    
    # -----------------------------
    # Background computations
    b_accepted_ht = (bht[:, None, None] >= HT[None, :, :])
    b_accepted_as = (bas[:, None, None] >= AS[None, :, :])
    
    b_ht_count = b_accepted_ht.sum(axis=0)
    b_as_count = b_accepted_as.sum(axis=0)
    b_both_count = (b_accepted_ht & b_accepted_as).sum(axis=0)
    
    b_accepted_events = b_ht_count + b_as_count - b_both_count
    r_b = 100 * b_accepted_events / bht.shape[0]
    
    # Overlap 
    b_overlap = 100 * (b_both_count+1e-10) / (b_accepted_events+1e-10)
    
    s1_overlap = 100 * (s1_both_count+1e-10) / (s1_accepted_events+1e-10)
    s2_overlap = 100 * (s2_both_count+1e-10) / (s2_accepted_events+1e-10)
    
    # Additional rates if needed
    r_bht = 100 * b_ht_count / bht.shape[0]
    r_bas = 100 * b_as_count / bht.shape[0]
    r1_sht = 100 * s1_ht_count / sht1.shape[0]
    r1_sas = 100 * s1_as_count / sas1.shape[0]

    r2_sht = 100 * s2_ht_count / sht2.shape[0]
    r2_sas = 100 * s2_as_count / sas2.shape[0]

    
    t_b = 0.25
    a = [100, .2]
    
    
    #cost = (a[0]*np.abs(r_b - t_b))**(4) + (a[1] *np.abs(total_s_rate - 100))**1 #+ a[2]*b_overlap**2 + a[2]*s_overlap**2
    cost = (a[0]*np.abs(r_b - t_b)) + (a[1] *np.abs(total_s_rate - 100))
    log_Cost = np.log10(cost.clip(min=1e-10))
    
    return log_Cost, r_b, r1_s, r2_s, r_bht, r_bas, r1_sht, r2_sht, r1_sas, r2_sas, HT, AS


def V2(bht, sht1, sht2, bas, sas1, sas2):
    #ttbar is picked to be 1
    
    

    MAX = np.percentile(bht,99.99)
    
    ht_vals = np.linspace(np.percentile(bht,0.01), MAX, 100)

    MAX = np.percentile(bas,99.99)
    
    as_vals = np.linspace(np.percentile(bas,0.01), MAX, 100)

    HT, AS = np.meshgrid(ht_vals, as_vals, indexing='ij') 
    
    
    s1_accepted_ht = (sht1[:, None, None] >= HT[None, :, :])   # shape: (N_signal, n_ht, n_as)
    s1_accepted_as = (sas1[:, None, None] >= AS[None, :, :])     # shape: (N_signal, n_ht, n_as)

    s2_accepted_ht = (sht2[:, None, None] >= HT[None, :, :])   # shape: (N_signal, n_ht, n_as)
    s2_accepted_as = (sas2[:, None, None] >= AS[None, :, :])     # shape: (N_signal, n_ht, n_as)
    

    s1_ht_count = s1_accepted_ht.sum(axis=0)      # shape: (n_ht, n_as)
    s1_as_count = s1_accepted_as.sum(axis=0)        # shape: (n_ht, n_as)
    s1_both_count = (s1_accepted_ht & s1_accepted_as).sum(axis=0)  # events passing both

    s2_ht_count = s2_accepted_ht.sum(axis=0)      # shape: (n_ht, n_as)
    s2_as_count = s2_accepted_as.sum(axis=0)        # shape: (n_ht, n_as)
    s2_both_count = (s2_accepted_ht & s2_accepted_as).sum(axis=0)  # events passing both
    
    # Total signal accepted events = ht + as - both 
    s1_accepted_events = s1_ht_count + s1_as_count - s1_both_count
    r1_s = 100 * s1_accepted_events / sht1.shape[0]
    
    s2_accepted_events = s2_ht_count + s2_as_count - s2_both_count
    r2_s = 100 * s2_accepted_events / sht2.shape[0]
    
    
    total_s_accepted_events = s2_accepted_events + s1_accepted_events
    total_s_rate = 100 * (total_s_accepted_events)/ (sht1.shape[0] + sht2.shape[0])

    
    # -----------------------------
    # Background computations
    b_accepted_ht = (bht[:, None, None] >= HT[None, :, :])
    b_accepted_as = (bas[:, None, None] >= AS[None, :, :])
    
    b_ht_count = b_accepted_ht.sum(axis=0)
    b_as_count = b_accepted_as.sum(axis=0)
    b_both_count = (b_accepted_ht & b_accepted_as).sum(axis=0)
    
    
    
    b_accepted_events = b_ht_count + b_as_count - b_both_count
    r_b = 100 * b_accepted_events / bht.shape[0]

    
    
    r_as_ex = 100 * (b_as_count - b_both_count)/(bht.shape[0])

    
    
    # Overlap 
    b_overlap = 100 * (b_both_count+1e-10) / (b_accepted_events+1e-10)
    
    s1_overlap = 100 * (s1_both_count+1e-10) / (s1_accepted_events+1e-10)
    s2_overlap = 100 * (s2_both_count+1e-10) / (s2_accepted_events+1e-10)
    
    # Additional rates if needed
    r_bht = 100 * b_ht_count / bht.shape[0]
    r_bas = 100 * b_as_count / bht.shape[0]
    r1_sht = 100 * s1_ht_count / sht1.shape[0]
    r1_sas = 100 * s1_as_count / sas1.shape[0]

    r2_sht = 100 * s2_ht_count / sht2.shape[0]
    r2_sas = 100 * s2_as_count / sas2.shape[0]

    
    a = [100, .2, 25]
    t_b = 0.25
    percentage = .5
    #(a[0] *np.abs(r_b - t_b))**(4) + (a[1]*np.abs(r1_s - 90))**1 + 
    cost =  (a[0] *np.abs(r_b - t_b)) + (a[1] *np.abs(r1_s - 90)) + (a[2] * np.abs(r_as_ex - percentage*t_b))

    log_Cost = np.log10(cost.clip(min=1e-10))
    

    return log_Cost, r_b, r1_s, r2_s, r_bht, r_bas, r1_sht, r2_sht, r1_sas, r2_sas, HT, AS


def V3(bht, sht1, sht2, bas, sas1, sas2, bnjets, ref=[5.6,2.7]):
    
    MAX = np.percentile(bht,99.99)
    
    ht_vals = np.linspace(np.percentile(bht,0.01), MAX, 100)
    #print('bht min and max: ', np.percentile(bht,0.01), MAX)
    
    MAX = np.percentile(bas,99.99)
    
    #print('bas min and max: ', np.percentile(bas,0.01), MAX)
    
    as_vals = np.linspace(np.percentile(bas,0.01), MAX, 100)

    HT, AS = np.meshgrid(ht_vals, as_vals, indexing='ij') 
    
    
    s1_accepted_ht = (sht1[:, None, None] >= HT[None, :, :])   # shape: (N_signal, n_ht, n_as)
    s1_accepted_as = (sas1[:, None, None] >= AS[None, :, :])     # shape: (N_signal, n_ht, n_as)

    s2_accepted_ht = (sht2[:, None, None] >= HT[None, :, :])   # shape: (N_signal, n_ht, n_as)
    s2_accepted_as = (sas2[:, None, None] >= AS[None, :, :])     # shape: (N_signal, n_ht, n_as)
    

    s1_ht_count = s1_accepted_ht.sum(axis=0)      # shape: (n_ht, n_as)
    s1_as_count = s1_accepted_as.sum(axis=0)        # shape: (n_ht, n_as)
    s1_both_count = (s1_accepted_ht & s1_accepted_as).sum(axis=0)  # events passing both

    s2_ht_count = s2_accepted_ht.sum(axis=0)      # shape: (n_ht, n_as)
    s2_as_count = s2_accepted_as.sum(axis=0)        # shape: (n_ht, n_as)
    s2_both_count = (s2_accepted_ht & s2_accepted_as).sum(axis=0)  # events passing both
    
    # Total signal accepted events = ht + as - both 
    s1_accepted_events = s1_ht_count + s1_as_count - s1_both_count
    r1_s = 100 * s1_accepted_events / sht1.shape[0]
    
    s2_accepted_events = s2_ht_count + s2_as_count - s2_both_count
    r2_s = 100 * s2_accepted_events / sht2.shape[0]
    
    total_s_accepted_events = s2_accepted_events + s1_accepted_events
    total_s_rate = 100 * (total_s_accepted_events)/ (sht1.shape[0] + sht2.shape[0])

    
    # -----------------------------
    # Background computations
    b_accepted_ht = (bht[:, None, None] >= HT[None, :, :])
    b_accepted_as = (bas[:, None, None] >= AS[None, :, :])
    b_accepted = (b_accepted_ht | b_accepted_as)
    
    b_ht_count = b_accepted_ht.sum(axis=0)
    b_as_count = b_accepted_as.sum(axis=0)
    b_both_count = (b_accepted_ht & b_accepted_as).sum(axis=0)
    
    b_accepted_events = b_ht_count + b_as_count - b_both_count
    r_b = 100 * b_accepted_events / bht.shape[0]
    
    # Overlap 
    b_overlap = 100 * (b_both_count+1e-10) / (b_accepted_events+1e-10)
    
    s1_overlap = 100 * (s1_both_count+1e-10) / (s1_accepted_events+1e-10)
    s2_overlap = 100 * (s2_both_count+1e-10) / (s2_accepted_events+1e-10)
    
    # Additional rates if needed
    r_bht = 100 * b_ht_count / bht.shape[0]
    r_bas = 100 * b_as_count / bht.shape[0]
    r1_sht = 100 * s1_ht_count / sht1.shape[0]
    r1_sas = 100 * s1_as_count / sas1.shape[0]

    r2_sht = 100 * s2_ht_count / sht2.shape[0]
    r2_sas = 100 * s2_as_count / sas2.shape[0]

    
    # -----------------------------
    # Compute the cost based on the selected index.
    a = [100, .2, 1/0.5, 1/0.5]
    #a = [100, .2, 1/3.5, 1/2.5]
    t_b = 0.25
    
    #reshape jets per event for broadcasting
    bnjets_reshaped = bnjets[:, None, None]  # shape (N_events, 1, 1)
    
    Ht_cost = 1
    AS_cost = 4
    
    # Trigger path cost
    b_Ecomp_cost = ((b_accepted*bnjets_reshaped).sum(axis=0))/(b_accepted_events)
        
    # Event level Cost 
    b_Tcomp_cost = (Ht_cost * (b_ht_count - b_both_count) + AS_cost * (b_as_count))/(b_accepted_events)
    
    
    
    cost = (
    a[0] * np.abs(r_b - t_b) +
    a[1] * np.abs(total_s_rate - 100) +
    a[2] * np.maximum(b_Ecomp_cost - ref[0], 0) +
    a[3] * np.maximum(b_Tcomp_cost - ref[1], 0)
    )

    
    log_Cost = np.log10(cost.clip(min=1e-10))

    return log_Cost, r_b, r1_s, r2_s, r_bht, r_bas, r1_sht, r2_sht, r1_sas, r2_sas, HT, AS



def local_V1(bht, sht1, sht2, bas, sas1, sas2, ht_value, as_value, ht_window=20, as_window=20, num_points=10):
    
    # Define the local range around the given ht_value and as_value
    ht_min, ht_max = ht_value - ht_window, ht_value + ht_window
    as_min, as_max = as_value - as_window, as_value + as_window

    # Generate local ht and as values
    MAX = np.percentile(bht,99.99)

    ht_vals = np.linspace(max(np.min(bht), ht_min), min(MAX, ht_max), num_points)
    
    MAX = np.percentile(bas,99.99)

    as_vals = np.linspace(max(np.min(bas), as_min), min(MAX, as_max), num_points)

    # Create a grid in the local window
    HT, AS = np.meshgrid(ht_vals, as_vals, indexing='ij')
    

    # Signal computations
    s1_accepted_ht = (sht1[:, None, None] >= HT[None, :, :])
    s1_accepted_as = (sas1[:, None, None] >= AS[None, :, :])
    
    s1_ht_count = s1_accepted_ht.sum(axis=0)
    s1_as_count = s1_accepted_as.sum(axis=0)
    s1_both_count = (s1_accepted_ht & s1_accepted_as).sum(axis=0)

    s1_accepted_events = s1_ht_count + s1_as_count - s1_both_count
    #r_s1 = 100 * s1_accepted_events / sht1.shape[0]
    
    
    s2_accepted_ht = (sht2[:, None, None] >= HT[None, :, :])
    s2_accepted_as = (sas2[:, None, None] >= AS[None, :, :])
    
    s2_ht_count = s2_accepted_ht.sum(axis=0)
    s2_as_count = s2_accepted_as.sum(axis=0)
    s2_both_count = (s2_accepted_ht & s2_accepted_as).sum(axis=0)

    s2_accepted_events = s2_ht_count + s2_as_count - s2_both_count
    #r_s2 = 100 * s2_accepted_events / sht2.shape[0]
    
    r_s = 100 * (s1_accepted_events+s2_accepted_events) / (sht1.shape[0]+sht2.shape[0])


    # Background computations
    b_accepted_ht = (bht[:, None, None] >= HT[None, :, :])
    b_accepted_as = (bas[:, None, None] >= AS[None, :, :])

    b_ht_count = b_accepted_ht.sum(axis=0)
    b_as_count = b_accepted_as.sum(axis=0)
    b_both_count = (b_accepted_ht & b_accepted_as).sum(axis=0)

    b_accepted_events = b_ht_count + b_as_count - b_both_count
    r_b = 100 * b_accepted_events / bht.shape[0]

    # Overlap calculations
    #b_overlap = 100 * b_both_count / bht.shape[0]
    #s_overlap = 100 * s_both_count / sht.shape[0]

    # Additional rates
    r_bht = 100 * b_ht_count / bht.shape[0]
    r_bas = 100 * b_as_count / bht.shape[0]
    r_sht = 100 * (s1_ht_count + s2_ht_count)/ (sht1.shape[0]+sht2.shape[0])
    r_sas = 100 * (s1_as_count + s2_as_count)/ (sas1.shape[0]+sas2.shape[0])
    
    
    
    # Cost function
    a = [100, .2]
    t_b = 0.25
    cost = (a[0]*np.abs(r_b - t_b)) + (a[1] *np.abs(r_s - 100))
    log_Cost = np.log10(cost.clip(min=1e-10))

    return log_Cost, r_b, r_s, HT, AS


def local_V2(bht, sht1, sht2, bas, sas1, sas2, ht_value, as_value, ht_window=20, as_window=20, num_points=10):
    

    # Define the local range around the given ht_value and as_value
    ht_min, ht_max = ht_value - ht_window, ht_value + ht_window
    as_min, as_max = as_value - as_window, as_value + as_window

    # Generate local ht and as values
    MAX = np.percentile(bht,99.99)

    ht_vals = np.linspace(max(np.min(bht), ht_min), min(MAX, ht_max), num_points)
    
    MAX = np.percentile(bht,99.99)

    as_vals = np.linspace(max(np.min(bas), as_min), min(MAX, as_max), num_points)

    # Create a grid in the local window
    HT, AS = np.meshgrid(ht_vals, as_vals, indexing='ij')
    

    # Signal computations
    s1_accepted_ht = (sht1[:, None, None] >= HT[None, :, :])   # shape: (N_signal, n_ht, n_as)
    s1_accepted_as = (sas1[:, None, None] >= AS[None, :, :])     # shape: (N_signal, n_ht, n_as)

    s2_accepted_ht = (sht2[:, None, None] >= HT[None, :, :])   # shape: (N_signal, n_ht, n_as)
    s2_accepted_as = (sas2[:, None, None] >= AS[None, :, :])     # shape: (N_signal, n_ht, n_as)
    

    s1_ht_count = s1_accepted_ht.sum(axis=0)      # shape: (n_ht, n_as)
    s1_as_count = s1_accepted_as.sum(axis=0)        # shape: (n_ht, n_as)
    s1_both_count = (s1_accepted_ht & s1_accepted_as).sum(axis=0)  # events passing both

    s2_ht_count = s2_accepted_ht.sum(axis=0)      # shape: (n_ht, n_as)
    s2_as_count = s2_accepted_as.sum(axis=0)        # shape: (n_ht, n_as)
    s2_both_count = (s2_accepted_ht & s2_accepted_as).sum(axis=0)  # events passing both
    
    # Total signal accepted events = ht + as - both 
    s1_accepted_events = s1_ht_count + s1_as_count - s1_both_count
    r1_s = 100 * s1_accepted_events / sht1.shape[0]
    
    s2_accepted_events = s2_ht_count + s2_as_count - s2_both_count
    #r2_s = 100 * s2_accepted_events / sht2.shape[0]
    
    
    total_s_accepted_events = s2_accepted_events + s1_accepted_events
    r_s = 100 * (total_s_accepted_events)/ (sht1.shape[0] + sht2.shape[0])

    
    # -----------------------------
    # Background computations
    b_accepted_ht = (bht[:, None, None] >= HT[None, :, :])
    b_accepted_as = (bas[:, None, None] >= AS[None, :, :])
    
    b_ht_count = b_accepted_ht.sum(axis=0)
    b_as_count = b_accepted_as.sum(axis=0)
    b_both_count = (b_accepted_ht & b_accepted_as).sum(axis=0)
    
    
    
    b_accepted_events = b_ht_count + b_as_count - b_both_count
    r_b = 100 * b_accepted_events / bht.shape[0]

    
    
    r_as_ex = 100 * (b_as_count - b_both_count)/(bht.shape[0])

    
    
    # Overlap 
    #b_overlap = 100 * (b_both_count+1e-10) / (b_accepted_events+1e-10)
    
    #s1_overlap = 100 * (s1_both_count+1e-10) / (s1_accepted_events+1e-10)
    #s2_overlap = 100 * (s2_both_count+1e-10) / (s2_accepted_events+1e-10)
    
    # Additional rates if needed
    r_bht = 100 * b_ht_count / bht.shape[0]
    r_bas = 100 * b_as_count / bht.shape[0]
    r_sht = 100 * (s1_ht_count + s2_ht_count)/ (sht1.shape[0]+sht2.shape[0])
    r_sas = 100 * (s1_as_count + s2_as_count)/ (sas1.shape[0]+sas2.shape[0])


    #r1_sht = 100 * s1_ht_count / sht1.shape[0]
    #r1_sas = 100 * s1_as_count / sas1.shape[0]

    #r2_sht = 100 * s2_ht_count / sht2.shape[0]
    #r2_sas = 100 * s2_as_count / sas2.shape[0]

    
    a = [100, .2, 25]
    t_b = 0.25
    percentage = .5
    #(a[0] *np.abs(r_b - t_b))**(4) + (a[1]*np.abs(r1_s - 90))**1 + 
    cost =  (a[0] *np.abs(r_b - t_b)) + (a[1] *np.abs(r1_s - 90)) + (a[2] * np.abs(r_as_ex - percentage*t_b))

    log_Cost = np.log10(cost.clip(min=1e-10))

    return log_Cost, r_b, r_s, HT, AS


def local_V3(bht, sht1, sht2, bas, sas1, sas2, bnjets, ref, ht_value, as_value, ht_window=20, as_window=20, num_points=10):
    
    # Define the local range around the given ht_value and as_value
    ht_min, ht_max = ht_value - ht_window, ht_value + ht_window
    as_min, as_max = as_value - as_window, as_value + as_window


    MAX = np.percentile(bht,99.99)
    
    ht_vals = np.linspace(max(np.min(bht), ht_min), min(MAX, ht_max), num_points)
    #print('bht min and max: ', np.percentile(bht,0.01), MAX)
    

    MAX = np.percentile(bas,99.99)
    
    #print('bas min and max: ', np.percentile(bas,0.01), MAX)
    
    as_vals = np.linspace(max(np.min(bas), as_min), min(MAX, as_max), num_points)

    HT, AS = np.meshgrid(ht_vals, as_vals, indexing='ij') 
    
    
    s1_accepted_ht = (sht1[:, None, None] >= HT[None, :, :])   # shape: (N_signal, n_ht, n_as)
    s1_accepted_as = (sas1[:, None, None] >= AS[None, :, :])     # shape: (N_signal, n_ht, n_as)

    s2_accepted_ht = (sht2[:, None, None] >= HT[None, :, :])   # shape: (N_signal, n_ht, n_as)
    s2_accepted_as = (sas2[:, None, None] >= AS[None, :, :])     # shape: (N_signal, n_ht, n_as)
    

    s1_ht_count = s1_accepted_ht.sum(axis=0)      # shape: (n_ht, n_as)
    s1_as_count = s1_accepted_as.sum(axis=0)        # shape: (n_ht, n_as)
    s1_both_count = (s1_accepted_ht & s1_accepted_as).sum(axis=0)  # events passing both

    s2_ht_count = s2_accepted_ht.sum(axis=0)      # shape: (n_ht, n_as)
    s2_as_count = s2_accepted_as.sum(axis=0)        # shape: (n_ht, n_as)
    s2_both_count = (s2_accepted_ht & s2_accepted_as).sum(axis=0)  # events passing both
    
    # Total signal accepted events = ht + as - both 
    s1_accepted_events = s1_ht_count + s1_as_count - s1_both_count
    #r1_s = 100 * s1_accepted_events / sht1.shape[0]
    
    s2_accepted_events = s2_ht_count + s2_as_count - s2_both_count
    #r2_s = 100 * s2_accepted_events / sht2.shape[0]
    
    total_s_accepted_events = s2_accepted_events + s1_accepted_events
    r_s = 100 * (total_s_accepted_events)/ (sht1.shape[0] + sht2.shape[0])

    
    # -----------------------------
    # Background computations
    b_accepted_ht = (bht[:, None, None] >= HT[None, :, :])
    b_accepted_as = (bas[:, None, None] >= AS[None, :, :])
    #b_accepted = (b_accepted_ht | b_accepted_as)
    
    b_ht_count = b_accepted_ht.sum(axis=0)
    b_as_count = b_accepted_as.sum(axis=0)
    b_both_count = (b_accepted_ht & b_accepted_as).sum(axis=0)
    
    b_accepted_events = b_ht_count + b_as_count - b_both_count
    r_b = 100 * b_accepted_events / bht.shape[0]
    
    # Overlap 
    #b_overlap = 100 * (b_both_count+1e-10) / (b_accepted_events+1e-10)
    
    #s1_overlap = 100 * (s1_both_count+1e-10) / (s1_accepted_events+1e-10)
    #s2_overlap = 100 * (s2_both_count+1e-10) / (s2_accepted_events+1e-10)
    
    # Additional rates if needed
    r_bht = 100 * b_ht_count / bht.shape[0]
    r_bas = 100 * b_as_count / bht.shape[0]
    r_sht = 100 * (s1_ht_count + s2_ht_count)/ (sht1.shape[0]+sht2.shape[0])
    r_sas = 100 * (s1_as_count + s2_as_count)/ (sas1.shape[0]+sas2.shape[0])

    #r1_sht = 100 * s1_ht_count / sht1.shape[0]
    #r1_sas = 100 * s1_as_count / sas1.shape[0]

    #r2_sht = 100 * s2_ht_count / sht2.shape[0]
    #r2_sas = 100 * s2_as_count / sas2.shape[0]

    
    # -----------------------------
    # Compute the cost based on the selected index.
    a = [100, .2, 1/0.5, 1/0.5]
    #a = [100, .2, 1/3.5, 1/2.5]
    t_b = 0.25
    
    b_Ecomp_cost, b_Tcomp_cost = comp_costs(
    (b_accepted_ht | b_accepted_as),
    b_accepted_events,
    b_ht_count,
    b_both_count,
    b_as_count,
    bnjets,
    )

    
    cost = (
    a[0] * np.abs(r_b - t_b) +
    a[1] * np.abs(r_s - 100) +
    a[2] * np.maximum(b_Ecomp_cost - ref[0], 0) +
    a[3] * np.maximum(b_Tcomp_cost - ref[1], 0)
    )
    
    log_Cost = np.log10(cost.clip(min=1e-10))
    

    return log_Cost, r_b, r_s, HT, AS


def Trigger(bht_, sht1_, sht2_, bas_, sas1_, sas2_, bnjets ,ht_cut, as_cut):
    #num_signal = sht_.shape[0]
    #num_background = bht_.shape[0]
    
    
    # Apply cuts to both signal and background
    s1_accepted_ht = sht1_ >= ht_cut  # Signal events accepted by Ht cut
    s2_accepted_ht = sht2_ >= ht_cut
    s1_accepted_as = sas1_ >= as_cut  # Signal events accepted by AS cut
    s2_accepted_as = sas2_ >= as_cut
    b_accepted_ht = bht_ >= ht_cut  # Background events accepted by Ht cut
    b_accepted_as = bas_ >= as_cut  # Background events accepted by AS cut
            
    # Calculate the number of accepted signal events
    s1_accepted_events = np.sum(s1_accepted_ht) + np.sum(s1_accepted_as) - np.sum(s1_accepted_ht & s1_accepted_as)
    s2_accepted_events = np.sum(s2_accepted_ht) + np.sum(s2_accepted_as) - np.sum(s2_accepted_ht & s2_accepted_as)
    
    # Calculate the number of accepted background events
    b_accepted_events = np.sum(b_accepted_ht) + np.sum(b_accepted_as) - np.sum(b_accepted_ht & b_accepted_as)
            
    # Calculate rates
    #r1_s = 100 * s1_accepted_events / sht1_.shape[0]
    #r2_s = 100 * s2_accepted_events / sht2_.shape[0]
    r_s = 100 *(s1_accepted_events+s2_accepted_events)/(sht1_.shape[0]+sht2_.shape[0]+1e-10)
    r_b = 100 * b_accepted_events / bht_.shape[0]
    
    r_sht = 100*(np.sum(s1_accepted_ht)+np.sum(s2_accepted_ht))/(sht1_.shape[0]+sht2_.shape[0]+1e-10)
    r_bht = 100*(np.sum(b_accepted_ht)/bht_.shape[0])
    r_sas = 100*(np.sum(s1_accepted_as)+np.sum(s2_accepted_as))/(sas1_.shape[0]+sas2_.shape[0]+1e-10)
    r_bas = 100*(np.sum(b_accepted_as)/bas_.shape[0])
    
    b_accepted = (b_accepted_ht | b_accepted_as)
    
    ht_weight = 1
    as_weight = 4
    b_ht_count = np.sum(b_accepted_ht)
    b_as_count = np.sum(b_accepted_as)
    b_both_count = np.sum(b_accepted_ht & b_accepted_as)
    
    
    # Trigger path cost: average nJets over accepted events
    b_Ecomp_cost = ((b_accepted * bnjets).sum(axis=0)) / (b_accepted_events + 1e-10)

    # Event-level cost: weighted counts per accepted event
    b_Tcomp_cost = (ht_weight * (b_ht_count - b_both_count) + as_weight * b_as_count) / (b_accepted_events + 1e-10)

            
    
                
    return r_b, r_s, r_bht, r_bas, r_sht, r_sas, b_Ecomp_cost, b_Tcomp_cost
