import numpy as np

def V1(bht, sht1, sht2, bas, sas1, sas2):

    max1 = np.percentile(sht1,99.99)
    max2 = np.percentile(sht2,99.99)
    MAX = max(max1,max2)
    MAX = np.percentile(bht,99.99)
    
    ht_vals = np.linspace(np.percentile(bht,0.01), MAX, 100)
    #print('bht min and max: ', np.percentile(bht,0.01), MAX)
    
    

    max1 = np.percentile(sas1,99.99)
    max2 = np.percentile(sas2,99.99)
    MAX = max(max1,max2)
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
    
    
    max1 = np.percentile(sht1,99.99)
    max2 = np.percentile(sht2,99.99)
    MAX = max(max1,max2)
    MAX = np.percentile(bht,99.99)
    
    ht_vals = np.linspace(np.percentile(bht,0.001), MAX, 100)

    max1 = np.percentile(sas1,99.99)
    max2 = np.percentile(sas2,99.99)
    MAX = max(max1,max2)
    MAX = np.percentile(bas,99.99)
    
    as_vals = np.linspace(np.percentile(bas,0.001), MAX, 100)

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
    percentage = .3
    #(a[0] *np.abs(r_b - t_b))**(4) + (a[1]*np.abs(r1_s - 90))**1 + 
    cost =  (a[0] *np.abs(r_b - t_b)) + (a[1] *np.abs(r1_s - 90)) + (a[2] * np.abs(r_as_ex - percentage*t_b))
    #cost =  (a[0] *np.abs(r_b - t_b)) + (a[1] *np.abs(r1_s - 90)) - (a[2] * r_as_ex)

    log_Cost = np.log10(cost.clip(min=1e-10))

    return cost, r_b, r1_s, r2_s, r_bht, r_bas, r1_sht, r2_sht, r1_sas, r2_sas, HT, AS


def V3(bht, sht1, sht2, bas, sas1, sas2, bnjets, snjets1, snjets2):
    
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
    
    #reshape jets per event for broadcasting
    bnjets_reshaped = bnjets[:, None, None]  # shape (N_events, 1, 1)
    #snjets1_reshaped = snjets1[:, None, None] 
    #snjets2_reshaped = snjets2[:, None, None] 

    Ht_cost = 1
    AS_cost = 4

    a = [100, .2, 1/0.5, 1/0.5]
    #a = [100, .2, 1/3.5, 1/2.5]
    t_b = 0.25
    
    
    
    # Trigger path cost
    b_Ecomp_cost = ((b_accepted*bnjets_reshaped).sum(axis=0))/(b_accepted_events)  
        
    # Event level Cost 
    b_Tcomp_cost = (Ht_cost*(b_ht_count - b_both_count) + AS_cost * (b_as_count))/(b_accepted_events)
    
    
    
    cost = (
    a[0] * np.abs(r_b - t_b) +
    a[1] * np.abs(total_s_rate - 100) +
    a[2] * np.maximum(b_Ecomp_cost - 5.5, 0) +
    a[3] * np.maximum(b_Tcomp_cost - 3.0, 0)
    )
    
    log_Cost = np.log10(cost.clip(min=1e-10))

    return log_Cost, r_b, r1_s, r2_s, r_bht, r_bas, r1_sht, r2_sht, r1_sas, r2_sas, HT, AS
