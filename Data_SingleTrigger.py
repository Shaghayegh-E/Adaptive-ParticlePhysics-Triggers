#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.lines as mlines

import numpy as np
import matplotlib.pyplot as plt
import re
import h5py
import hdf5plugin
#
import mplhep as hep #pip install mplhep
hep.style.use("CMS")




def PD_controller1(r_,pre_,cut_):
    #Kp = 2.55
    Kp = 30
    Kd = 5

    target = 0.25
    error = r_ - target
    delta = error - pre_
    newcut_ = cut_ + Kp*error + Kd*delta
    #print('error:',error)
    return newcut_, error

def PD_controller2(r_,pre_,cut_):

    #Kp = .05
    #Kp = 2.5
    Kp = 15
    Kd = 0

    target = 0.25
    error = r_ - target
    delta = error - pre_
    newcut_ = cut_ + Kp*error + Kd*delta
    #print('error:',error)
    return newcut_, error

def Sing_Trigger(bht_,ht_cut):
    num_ = bht_.shape[0]

    # Apply cuts
    accepted_ht_ = np.sum(bht_ >= ht_cut)
    r_ = 100 * accepted_ht_ / num_
    return r_


#def read_data(h5_file_path):
#    with h5py.File(h5_file_path, 'r') as h5_file:
#        # Read datasets for background
#        Bas01_tot = h5_file['mc_bkg_score01'][:]
#        Bas04_tot = h5_file['mc_bkg_score04'][:]
#        #Bas_tot = h5_file['mc_bkg_Hmets'][:]
#        Bht_tot = h5_file['mc_bkg_ht'][:]
#        B_npvs = h5_file['mc_bkg_Npv'][:]
#        
#        # Read datasets for signal
#        Sas01_tot1 = h5_file['mc_tt_score01'][:]
#        Sas04_tot1 = h5_file['mc_tt_score04'][:]
#        #Sas_tot1 = h5_file['mc_dijet_Hmets'][:]
#        Sht_tot1 = h5_file['mc_tt_ht'][:]
#        S_npvs1 = h5_file['tt_Npv'][:]
#        #sig_key1 = h5_file['dijet_key'][:]  # Read signal keys if needed
#
#        Sas01_tot2 = h5_file['mc_aa_score01'][:]
#        Sas04_tot2 = h5_file['mc_aa_score04'][:]
#        #Sas_tot2 = h5_file['mc_ttbar_Hmets'][:]
#        Sht_tot2 = h5_file['mc_aa_ht'][:]
#        S_npvs2 = h5_file['aa_Npv'][:]
#        #sig_key2 = h5_file['tt_key'][:]  # Read signal keys if needed
#        
#        # Read datasets for data
#        #data_ht = h5_file['data_ht'][:]
#        #data_score = h5_file['data_score'][:]
#        #data_npv = h5_file['data_Npv'][:]
#        
#    return Sas01_tot1, Sas04_tot1, Sht_tot1, S_npvs1, Sas01_tot2, Sas04_tot2, Sht_tot2, S_npvs2, Bas01_tot, Bas04_tot, Bht_tot, B_npvs #,data_ht, data_score, data_npv
#
#
#def PD_controller1(r_,pre_,cut_):
#    #Kp = 2.55
#    Kp = 50#30
#    Kd = 2
#
#    target = 0.25
#    error = r_ - target
#    delta = error - pre_
#    newcut_ = cut_ + Kp*error + Kd*delta
#    #print('error:',error)
#    return newcut_, error
#
#def PD_controller2(r_,pre_,cut_):
#
#    #Kp = .05
#    Kp = 100#2.5
#    #Kp = 15
#    Kd = 0
#
#    target = 0.25
#    error = r_ - target
#    delta = error - pre_
#    newcut_ = cut_ + Kp*error + Kd*delta
#    #print('error:',error)
#    return newcut_, error
#
#
#def PD_controller_with_bounds(r_, pre_, cut_):
#    lower_tol=0.22
#    upper_tol=0.28
#    Kp = 50
#    Kd = 2
#    target = 0.25
#
#    error = r_ - target
#    delta = error - pre_
#
#    newcut = cut_ + Kp * error + Kd * delta
#
#
#    return newcut, error
#
#
#def Sing_Trigger(bht_,ht_cut):
#    num_ = bht_.shape[0]
#
#    # Apply cuts
#    accepted_ht_ = np.sum(bht_ >= ht_cut)
#    r_ = 100 * accepted_ht_ / num_
#    return r_
#
#
#def find_cut_for_target_rate(data, target_rate):
#    sorted_scores = np.sort(data)
#    n = len(sorted_scores)
#    cut_index = int((1 - target_rate) * n)
#    return sorted_scores[cut_index]
#
def read_data(h5_file_path):
    with h5py.File(h5_file_path, 'r') as h5_file:
        # Background
        Bas01_tot = h5_file['data_scores01'][:]   # rinominato
        Bas04_tot = h5_file['data_scores04'][:]
        Bht_tot   = h5_file['data_ht'][:]
        B_npvs    = h5_file['data_Npv'][:]

        # Signal (ttbar)
        Sas01_tot1 = h5_file['matched_tt_scores01'][:]  # rinominato
        Sas04_tot1 = h5_file['matched_tt_scores04'][:]
        Sht_tot1   = h5_file['matched_tt_ht'][:]
        S_npvs1    = h5_file['matched_tt_npvs'][:]

        # Signal (AA→4b)
        Sas01_tot2 = h5_file['matched_aa_scores01'][:]  # rinominato
        Sas04_tot2 = h5_file['matched_aa_scores04'][:]
        Sht_tot2   = h5_file['matched_aa_ht'][:]
        S_npvs2    = h5_file['matched_aa_npvs'][:]

    return (
        Sas01_tot1, Sas04_tot1, Sht_tot1, S_npvs1,
        Sas01_tot2, Sas04_tot2, Sht_tot2, S_npvs2,
        Bas01_tot,  Bas04_tot, Bht_tot,  B_npvs
    )

#def read_data(h5_file_path):
#    with h5py.File(h5_file_path, 'r') as h5_file:
#        # Read datasets for background
#        Bas01_tot = h5_file['mc_bkg_score01'][:]
#        Bas04_tot = h5_file['mc_bkg_score04'][:]
#        #Bas_tot = h5_file['mc_bkg_Hmets'][:]
#        Bht_tot = h5_file['mc_bkg_ht'][:]
#        B_npvs = h5_file['mc_bkg_Npv'][:]
#        B_njets = h5_file['mc_bkg_njets'][:]
#        
#        # Read datasets for signal
#        Sas01_tot1 = h5_file['mc_tt_score01'][:]
#        Sas04_tot1 = h5_file['mc_tt_score04'][:]
#        #Sas_tot1 = h5_file['mc_dijet_Hmets'][:]
#        Sht_tot1 = h5_file['mc_tt_ht'][:]
#        S_npvs1 = h5_file['tt_Npv'][:]
#        #sig_key1 = h5_file['dijet_key'][:]  # Read signal keys if needed
#
#        Sas01_tot2 = h5_file['mc_aa_score01'][:]
#        Sas04_tot2 = h5_file['mc_aa_score04'][:]
#        #Sas_tot2 = h5_file['mc_ttbar_Hmets'][:]
#        Sht_tot2 = h5_file['mc_aa_ht'][:]
#        S_npvs2 = h5_file['aa_Npv'][:]
#        #sig_key2 = h5_file['tt_key'][:]  # Read signal keys if needed
#
## In[3]:
#
#    return Sas01_tot1, Sas04_tot1, Sht_tot1, S_npvs1, Sas01_tot2,  Sas04_tot2, Sht_tot2, S_npvs2, Bas01_tot, Bas04_tot, Bht_tot, B_npvs #,data_ht, data_score, data_npv
# path = "new_Data/Matched_data_2016_with04.h5" #Zixin: Trigger_food_Data.h5
path = "Data/Trigger_food_Data.h5" #Zixin: Trigger_food_Data.h5

# Load data from both files
#Sas_tot1, Sht_tot1, S_npvs1, Sas_tot2, Sht_tot2, S_npvs2, Bas_tot, Bht_tot, B_npvs, Dht_tot, Das_tot, D_npv = read_data(path)
#Sas_tot3, Sht_tot3, S_npvs3 = Sas_tot1, Sht_tot1/3, S_npvs1
#Sht_tot /= 5


Sas01_tot1, Sas04_tot1, Sht_tot1, S_npvs1, Sas01_tot2, Sas04_tot2, Sht_tot2, S_npvs2, Bas01_tot, Bas04_tot, Bht_tot, B_npvs = read_data(path)


Nb = len(B_npvs)
Ns = len(S_npvs1)
#Nd = len(D_npv)

print('hi')
#N = np.min([Nb,Ns,Nd])
N = Nb

#print(np.sum(S_npvs[:N]==B_npvs[:N]))


#print(Ns)
pre_r1 = 0
pre_r2_1 = 0
pre_r2_4 = 0


#Ht_cut = 200
#AS_cut = 7

fixed_Ht_cut = np.percentile(Bht_tot[200000:210000],99.75)
fixed_AS_cut1 = np.percentile(Bas01_tot[200000:210000],99.75)
fixed_AS_cut4 = np.percentile(Bas04_tot[200000:210000],99.75)
print('fixed_Ht_cut',fixed_Ht_cut)
print('fixed_AS_cut1',fixed_AS_cut1)
print('fixed_AS_cut4',fixed_AS_cut4)
print(np.percentile(Bht_tot[200000:],99.75))

percen_9975 = np.percentile(Bas04_tot, 99.75)
AA_passed = 100 * np.sum(Sas04_tot2 > percen_9975) / len(Sas04_tot2)
TT_passed = 100 * np.sum(Sas04_tot1 > percen_9975) / len(Sas04_tot1)
print('AA_passed',AA_passed)
print('TT_passed',TT_passed)


Ht_cut = fixed_Ht_cut
AS_cut1 = fixed_AS_cut1
AS_cut4 = fixed_AS_cut4
#AS_cut = 100

#print('passed rate  tt: ',100*np.sum(Sas_tot2>=fixed_AS_cut)/len(Sas_tot2))

bht = []
bas1 = []
bas4 = []
sht1 = []
sas1_1 = []
sas1_4 = []
sht2 = []
sas2_1 = []
sas2_4 = []
#sht3 = []
#sas3 = []
#das =[]
#dht = []

R1 = []
R2 = []
L_R3 = []
L_R4 = []
R3 = [0]
R4 = [0]
L_R5 = []
L_R6 = []
R5 = [0]
R6 = [0]
L_R7 = []
L_R8 = []
R7 = [0]
R8 = [0]


E1_1 = []
E2_1 = []
E1_4 = []
E2_4 = []
E3_1 = [0]
E4_1 = [0]
L_E3_1 = []
L_E4_1 = []

E3_4 = [0]
E4_4 = [0]
L_E3_4 = []
L_E4_4 = []

E5_1 = [0]
E6_1 = [0]
L_E5_1 = []
L_E6_1 = []
E5_4 = [0]
E6_4 = [0]
L_E5_4 = []
L_E6_4 = []



D1 = []
D2 = []
D3 = []
D4 = []


#chunk_size = 10000
chunk_size = 20000

for I in range(N):
    if I<200000: continue

    if I%chunk_size==0 : 
        start_idx = I
        end_idx = min(I + chunk_size, N)
        indices = list(range(start_idx, end_idx))

        bht = Bht_tot[indices]
        bas1 = Bas01_tot[indices]
        bas4 = Bas04_tot[indices]
        b_npvs = B_npvs[indices]  
        
        npv_min = np.min(b_npvs)
        npv_max = np.max(b_npvs)
        
        # Select signal events that fall within this Npv range
        signal_mask1 = (S_npvs1 >= npv_min) & (S_npvs1 <= npv_max)
        signal_mask2 = (S_npvs2 >= npv_min) & (S_npvs2 <= npv_max)
        #signal_mask3 = (S_npvs3 >= npv_min) & (S_npvs3 <= npv_max)
    
        # Extract matching signal events

        #tt
        sht1 = Sht_tot1[indices]
        sas1_1 = Sas01_tot1[indices]
        sas1_4 = Sas04_tot1[indices]

        #aa
        sht2 = Sht_tot2[indices]
        sas2_1 = Sas01_tot2[indices]
        sas2_4 = Sas04_tot2[indices]

        #scaled dijet
        #sht3 = Sht_tot3[indices]

        #data_mask = (D_npv >= npv_min) & (D_npv <= npv_max)
        #das = Das_tot[data_mask]
        #dht = Dht_tot[data_mask]

        #print(len(bas),len(sas),I)
        #if I==0 : continue

        
        #HT trigger

        rate1 = Sing_Trigger(np.array(bht),fixed_Ht_cut)
        rate2 = Sing_Trigger(np.array(bht),Ht_cut)
        #print(rate2)
        R1.append(rate1)
        R2.append(rate2)

        

        #signal1
        rate3 = Sing_Trigger(np.array(sht1),fixed_Ht_cut)
        rate4 = Sing_Trigger(np.array(sht1),Ht_cut)
        L_R3.append(rate3)
        L_R4.append(rate4)
        a = ((len(R3)-1)*R3[-1] + rate3)/len(R3)
        R3.append(a)
        b = ((len(R4)-1)*R4[-1] + rate4)/len(R4)
        R4.append(b)

        #signal2
        rate3 = Sing_Trigger(np.array(sht2),fixed_Ht_cut)
        rate4 = Sing_Trigger(np.array(sht2),Ht_cut)
        L_R5.append(rate3)
        L_R6.append(rate4)
        a = ((len(R5)-1)*R5[-1] + rate3)/len(R5)
        R5.append(a)
        b = ((len(R6)-1)*R6[-1] + rate4)/len(R6)
        R6.append(b)

        '''
        #signal3
        rate3 = Sing_Trigger(np.array(sht3),fixed_Ht_cut)
        rate4 = Sing_Trigger(np.array(sht3),Ht_cut)
        L_R7.append(rate3)
        L_R8.append(rate4)
        a = ((len(R7)-1)*R7[-1] + rate3)/len(R7)
        R7.append(a)
        b = ((len(R8)-1)*R8[-1] + rate4)/len(R8)
        R8.append(b)

        '''



        #rate5 = Sing_Trigger(dht,50)
        #rate6 = Sing_Trigger(dht,Ht_cut)
        #D1.append(rate5)
        #D2.append(rate6)

        Ht_cut, pre_r1 = PD_controller1(R2[-1],pre_r1,Ht_cut)
        

        #AS trigger
        rate1_1 = Sing_Trigger(np.array(bas1),fixed_AS_cut1)
        rate2_1 = Sing_Trigger(np.array(bas1),AS_cut1)
        #print(rate2)
        E1_1.append(rate1_1)
        E2_1.append(rate2_1)

        rate1_4 = Sing_Trigger(np.array(bas4),fixed_AS_cut4)
        rate2_4 = Sing_Trigger(np.array(bas4),AS_cut4)
        #print(rate2)
        E1_4.append(rate1_4)
        E2_4.append(rate2_4)

        #dim1_model
        rate3_1 = Sing_Trigger(np.array(sas1_1),fixed_AS_cut1)
        rate4_1 = Sing_Trigger(np.array(sas1_1),AS_cut1)
        L_E3_1.append(rate3_1)
        L_E4_1.append(rate4_1)
        a = ((len(E3_1)-1)*E3_1[-1] + rate3_1)/len(E3_1)
        E3_1.append(a)
        b = ((len(E4_1)-1)*E4_1[-1] + rate4_1)/len(E4_1)
        E4_1.append(b)

        #dim4_model
        rate3_4 = Sing_Trigger(np.array(sas1_4),fixed_AS_cut4)
        rate4_4 = Sing_Trigger(np.array(sas1_4),AS_cut4)
        L_E3_4.append(rate3_4)
        L_E4_4.append(rate4_4)
        a = ((len(E3_4)-1)*E3_4[-1] + rate3_4)/len(E3_4)
        E3_4.append(a)
        b = ((len(E4_4)-1)*E4_4[-1] + rate4_4)/len(E4_4)
        E4_4.append(b)

        #signal2
        rate3_1 = Sing_Trigger(np.array(sas2_1),fixed_AS_cut1)
        rate4_1 = Sing_Trigger(np.array(sas2_1),AS_cut1)
        L_E5_1.append(rate3_1)
        L_E6_1.append(rate4_1)
        a = ((len(E5_1)-1)*E5_1[-1] + rate3_1)/len(E5_1)
        E5_1.append(a)
        b = ((len(E6_1)-1)*E6_1[-1] + rate4_1)/len(E6_1)
        E6_1.append(b)

        rate3_4 = Sing_Trigger(np.array(sas2_4),fixed_AS_cut4)
        rate4_4 = Sing_Trigger(np.array(sas2_4),AS_cut4)
        L_E5_4.append(rate3_4)
        L_E6_4.append(rate4_4)
        a = ((len(E5_4)-1)*E5_4[-1] + rate3_4)/len(E5_4)
        E5_4.append(a)
        b = ((len(E6_4)-1)*E6_4[-1] + rate4_4)/len(E6_4)
        E6_4.append(b)


        #rate5 = Sing_Trigger(das,.5)
        #rate6 = Sing_Trigger(das,AS_cut)
        #D3.append(rate5)
        #D4.append(rate6)

        AS_cut1, pre_r2_1 = PD_controller2(E2_1[-1],pre_r2_1,AS_cut1)
        AS_cut4, pre_r2_4 = PD_controller2(E2_4[-1],pre_r2_4,AS_cut4)





E1_1 = np.array(E1_1)*400 #Convert into kHz
E1_4 = np.array(E1_4)*400
E2_1 = np.array(E2_1)*400
E2_4 = np.array(E2_4)*400

R1 = np.array(R1)*400
R2 = np.array(R2)*400

R3.pop(0)
R4.pop(0)
R5.pop(0)
R6.pop(0)
R7.pop(0)
R8.pop(0)





# In[74]:


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# Time axis normalized to [0, 1]
time = np.linspace(0, 1, len(R1))

plt.figure(figsize=(10, 6))

# --- Plots ---
plt.plot(time, R1,label='Constant Menu', color='tab:blue', linewidth=3, linestyle='dashed')
plt.plot(time, R2,label='PD Controller', color='mediumblue', linewidth=2.5, linestyle='solid')

# Tolerance lines (we'll legend them separately)
plt.axhline(y=0.28 * 400, color='gray', linestyle='--', linewidth=1.5)
plt.axhline(y=0.22 * 400, color='gray', linestyle='--', linewidth=1.5)

# --- Axes & labels ---
#plt.xlim(-0.01, 1.01)

#plt.ylim(15, 200)
plt.xlabel('Time (Fraction of Run)', loc='center')#, fontsize=18)
plt.ylabel('Background Rate [kHz]', loc='center')#, fontsize=18)
#plt.xticks(fontsize=18)
plt.ylim(0,200)
#plt.yticks(fontsize=18)
plt.grid(True, linestyle='--', alpha=0.6)

ax = plt.gca()

# ---- Main legend with column headers ----
header_const = mlines.Line2D([], [], color='none', linestyle='none')
header_pd    = mlines.Line2D([], [], color='none', linestyle='none')

const_rate = mlines.Line2D([], [], color='tab:blue',   linestyle='dashed', linewidth=3)
pd_rate    = mlines.Line2D([], [], color='mediumblue', linestyle='solid',  linewidth=2.5)

handles_main = [
    const_rate,   pd_rate  
]
labels_main = [
    "Constant Menu", "PD Controller"
]


# --- Prima legenda (sinistra) ---
leg_main = ax.legend(
    handles_main, labels_main,
    title="HT Trigger",
    ncol=1,
    loc='upper left',
    bbox_to_anchor=(0.02, 0.98),  # ← sposta leggermente dentro il grafico
    frameon=True,
    handlelength=2, columnspacing=1, labelspacing=0.8
)
ax.add_artist(leg_main)

# --- Seconda legenda (destra) ---
#
## ---- Second legend for tolerance lines ----
upper_tol = mlines.Line2D([], [], color='gray', linestyle='--', linewidth=1.5)
lower_tol = mlines.Line2D([], [], color='gray', linestyle='--', linewidth=1.5)
leg_tol = ax.legend(
    [upper_tol, lower_tol],
    ["Upper Tolerance (112)", "Lower Tolerance (88)"],
    title="Reference",
    loc='upper right',

    bbox_to_anchor=(0.98, 0.98),  # ← più interna verso sinistra
    frameon=True,
    fontsize=14,
    handlelength=2
)



# --- Save & show ---
plt.savefig('paper/bht_rate_pidData.pdf', bbox_inches='tight')
plt.show()
plt.close()







#AS
time = np.linspace(0, 1, len(E1_1))




import matplotlib.lines as mlines

# AS
time = np.linspace(0, 1, len(E1_1))

plt.figure(figsize=(10, 6))

# Plots (unchanged)
plt.plot(time, E1_1, label='Constant Menu, model dim=1', color='tab:blue', linewidth=3, linestyle='dotted')
#plt.plot(time, E1_4, label='Constant Menu, model dim=4', color='tab:blue', linewidth=3, linestyle='dashed')

plt.plot(time, E2_1, label='PD Controller, model dim=1', color='mediumblue', linewidth=2.5, linestyle='solid')
#plt.plot(time, E2_4, label='PD Controller, model dim=4', color='cyan', linewidth=2.5, linestyle='solid')

plt.axhline(y=0.28*400, color='gray', linestyle='--', linewidth=1.5, label='Upper Tolerance (112)')
plt.axhline(y=0.22*400, color='gray', linestyle='--', linewidth=1.5, label='Lower Tolerance (88)')

#plt.xlim(-0.01, 1.01)
#plt.ylim(60, 220)
plt.xlabel('Time (Fraction of Run)', loc='center')#, fontsize=20)
plt.ylabel('Background Rate [kHz]', loc='center')#, fontsize=20)
#plt.xticks(fontsize=18)
plt.ylim(60, 200)
#plt.yticks(fontsize=18)
plt.grid(True, linestyle='--', alpha=0.6)

# ---- Main legend with column headers ----
ax = plt.gca()

# Header placeholders
header_const = mlines.Line2D([], [], color='none', linestyle='none')
header_pd    = mlines.Line2D([], [], color='none', linestyle='none')

# Dummy handles that match the plotted styles
const_dim1 = mlines.Line2D([], [], color='tab:blue',    linestyle='dotted', linewidth=3)
pd_dim1    = mlines.Line2D([], [], color='mediumblue',  linestyle='solid',  linewidth=2.5)
const_dim4 = mlines.Line2D([], [], color='tab:blue',    linestyle='dashed', linewidth=3)
pd_dim4    = mlines.Line2D([], [], color='cyan',        linestyle='solid',  linewidth=2.5)

handles_main = [
    header_const,   
    const_dim1,   
    #const_dim4,  
    header_pd,
    pd_dim1,   
    #pd_dim4    
]
labels_main = [
    "Constant Menu", 
    "model dim=1",   
    #"model dim=4",
    "PD Controller",
    "model dim=1",  
    # "model dim=4"
]

# --- Prima legenda (sinistra) ---
leg_main = ax.legend(
    handles_main, labels_main,
    title="AD Trigger",
    fontsize=17,
    ncol=1,
    loc='upper left',
    bbox_to_anchor=(0.02, 0.98),  # ← sposta leggermente dentro il grafico
    frameon=True,
    handlelength=2, columnspacing=1, labelspacing=0.8
)
ax.add_artist(leg_main)

# --- Seconda legenda (destra) ---
leg_tol = ax.legend(
    [upper_tol, lower_tol],
    ["Upper Tolerance (112)", "Lower Tolerance (88)"],
    title="Reference",
    fontsize=18,
    loc='upper right',
    bbox_to_anchor=(0.98, 0.98),  # ← più interna verso sinistra
    frameon=True,
    handlelength=2
)


plt.savefig('paper/bas_rate_pidData.pdf', bbox_inches='tight')
plt.show()
plt.close()


# In[47]:


# Create a time axis (assuming equal time steps for all arrays)
# time = np.arange(len(R3))
time = np.linspace(0, 1, len(R3))




# In[49]:


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# Time axis normalized to [0, 1]
time = np.linspace(0, 1, len(R3))

# Consistent styles
styles = {
    "Constant": {"linestyle": "dashed", "linewidth": 2.5},
    "PD": {"linestyle": "solid",  "linewidth": 2.0},
}
colors = {
    "ttbar": "goldenrod",
    "HToAATo4B": "seagreen",
}

plt.figure(figsize=(10, 6))

# Constant Menu
plt.plot(time, R3 / R3[0], label="Constant Menu, ttbar",
         color=colors["ttbar"], **styles["Constant"])
plt.plot(time, R5 / R5[0], label="Constant Menu, HToAATo4B",
         color=colors["HToAATo4B"], **styles["Constant"])

# PD Controller
plt.plot(time, R4 / R4[0], label=f"PD Controller, ttbar, initial eff={R4[0]:.1f}%".format(R=R4),
         color=colors["ttbar"], **styles["PD"])
plt.plot(time, R6 / R6[0], label="PD Controller, HToAATo4B",
         color=colors["HToAATo4B"], **styles["PD"])

# Axes, labels
#plt.xlim(-0.01, 1.01)
#plt.ylim(0.85, 1.3)
plt.xlabel("Time (Fraction of Run)", loc='center')#, fontsize=20)
plt.ylabel("Relative Cumulative Efficiency", loc='center')#, fontsize=18)
#plt.xticks(fontsize=18)
#plt.yticks(fontsize=18)
plt.ylim(0.7,2)
plt.grid(True, linestyle="--", alpha=0.6)

# ---- Legend with per-column headers ----
# Create invisible header handles (no line shown)
header_const = mlines.Line2D([], [], color="none", linestyle="none")
header_pd    = mlines.Line2D([], [], color="none", linestyle="none")

# Real legend entries (match the plotted styles/colors)
const_ttbar   = mlines.Line2D([], [], color=colors["ttbar"],     **styles["Constant"])
pd_ttbar      = mlines.Line2D([], [], color=colors["ttbar"],     **styles["PD"])
const_higgs   = mlines.Line2D([], [], color=colors["HToAATo4B"], **styles["Constant"])
pd_higgs      = mlines.Line2D([], [], color=colors["HToAATo4B"], **styles["PD"])

# Order the handles row-by-row so ncol=2 makes column headers line up:
handles = [header_const, const_ttbar, const_higgs, header_pd, pd_ttbar, pd_higgs]
labels  = ["Constant Menu",  "ttbar", "HToAATo4B", "PD Controller", "ttbar",  "HToAATo4B"]

leg = plt.legend(
    handles, labels,
    title="HT Trigger", 
    #title_

    fontsize=18,
    ncol=2, loc="best", frameon=True,
    handlelength=2, columnspacing=0.5, labelspacing=0.8,
)

plt.ylim(0.8,1.4)
# Save & show
plt.savefig("paper/sht_rate_pidData.pdf", bbox_inches="tight")
plt.show()
plt.close()


# In[73]:


# Create a time axis (assuming equal time steps for all arrays)
# time = np.arange(len(R3))
time = np.linspace(0, 1, len(R3))




# Local S rate
# time = np.arange(len(L_R3))
time = np.linspace(0, 1, len(R3))



# In[75]:


# Time axis normalized to [0, 1]
time = np.linspace(0, 1, len(R3))

# Consistent styles
styles = {
    "Constant": {"linestyle": "dashed", "linewidth": 2.5},
    "PD": {"linestyle": "solid", "linewidth": 2.0},
}
colors = {
    "ttbar": "goldenrod",
    "HToAATo4B": "seagreen",
}

plt.figure(figsize=(10, 6))

# --- Plot Constant Menu ---
plt.plot(time, L_R3 / L_R3[0],
         label="Constant Menu, ttbar",
         color=colors["ttbar"], **styles["Constant"])
plt.plot(time, L_R5 / L_R5[0],
         label="Constant Menu, HToAATo4B",
         color=colors["HToAATo4B"], **styles["Constant"])

# --- Plot PD Controller ---
plt.plot(time, L_R4 / L_R4[0],
         label="PD Controller, HToBB",
         color=colors["ttbar"], **styles["PD"])
plt.plot(time, L_R6 / L_R6[0],
         label="PD Controller, HToAATo4B",
         color=colors["HToAATo4B"], **styles["PD"])

# --- Axis limits and labels ---
#plt.xlim(-0.01, 1.01)
#plt.ylim(0.8, 1.65)
plt.xlabel("Time (Fraction of Run)", loc='center')#, fontsize=20)
plt.ylabel("Relative Efficiency", loc='center')#, fontsize=20)
#plt.xticks(fontsize=18)
#plt.yticks(fontsize=18)
plt.grid(True, linestyle="--", alpha=0.6)

# ---- Legend with column headers ----
# Create invisible header handles
header_const = mlines.Line2D([], [], color="none", linestyle="none")
header_pd    = mlines.Line2D([], [], color="none", linestyle="none")

# Real legend entries (matching plotted lines)
const_ttbar  = mlines.Line2D([], [], color=colors["ttbar"],     **styles["Constant"])
pd_ttbar     = mlines.Line2D([], [], color=colors["ttbar"],     **styles["PD"])
const_higgs  = mlines.Line2D([], [], color=colors["HToAATo4B"], **styles["Constant"])
pd_higgs     = mlines.Line2D([], [], color=colors["HToAATo4B"], **styles["PD"])

# Order: row-by-row so that ncol=2 formats correctly
handles = [
    header_const, const_ttbar,       
    const_higgs, header_pd,         
    pd_ttbar ,pd_higgs          
]
labels = [
    "Constant Menu", 
    "ttbar", "HToBB",
    "PD Controller",
    "HToAATo4B", "HToAATo4B"
]

# Create legend
leg = plt.legend(
    handles, labels,
    title="HT Trigger",# title_fontsize=18,
    ncol=2, loc="best", frameon=True,
    handlelength=2, columnspacing=1, labelspacing=0.8,
    fontsize=18,

)
plt.ylim(0,2.50)
# --- Save and show ---
plt.savefig("paper/L_sht_rate_pidData.pdf", bbox_inches="tight")
plt.show()
plt.close()


# In[52]:





E3_1.pop(0)
E4_1.pop(0)
E3_4.pop(0)
E4_4.pop(0)
E5_1.pop(0)
E6_1.pop(0)
E5_4.pop(0)
E6_4.pop(0)



# In[57]:



E3_1.pop(0)
E4_1.pop(0)
E3_4.pop(0)
E4_4.pop(0)
E5_1.pop(0)
E6_1.pop(0)
E5_4.pop(0)
E6_4.pop(0)





import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# Remove first element from each list
for arr in [E3_1, E4_1, E3_4, E4_4, E5_1, E6_1, E5_4, E6_4]:
    arr.pop(0)

# Time axis normalized to [0, 1]
time = np.linspace(0, 1, len(E3_1))

# --- Consistent styles and colors ---
styles = {
    "Constant": {"linestyle": "dashed", "linewidth": 2.5},
    "PD": {"linestyle": "solid", "linewidth": 1.5},
}

# Color map for clarity
colors = {
    "ttbar_dim1": "goldenrod",
    "ttbar_dim4": "orangered",
    "higgs_dim1": "limegreen",
    "higgs_dim4": "seagreen",
}

plt.figure(figsize=(10, 6))

# --- Plot Constant Menu ---
plt.plot(time, E3_1 / E3_1[0], label="Constant Menu, ttbar (dim=1)",
         color=colors["ttbar_dim1"], **styles["Constant"])
#plt.plot(time, E3_4 / E3_4[0], label="Constant Menu, ttbar (dim=4)",
#         color=colors["ttbar_dim4"], **styles["Constant"])
#
plt.plot(time, E5_1 / E5_1[0], label="Constant Menu, HToAATo4B (dim=1)",
         color=colors["higgs_dim1"], **styles["Constant"])
#plt.plot(time, E5_4 / E5_4[0], label="Constant Menu, HToAATo4B (dim=4)",
#         color=colors["higgs_dim4"], **styles["Constant"])

# --- Plot PD Controller ---
plt.plot(time, E4_1 / E4_1[0], label="PD Controller, ttbar (dim=1)",
         color=colors["ttbar_dim1"], **styles["PD"])
#plt.plot(time, E4_4 / E4_4[0], label="PD Controller, ttbar (dim=4)",
#         color=colors["ttbar_dim4"], **styles["PD"])

plt.plot(time, E6_1 / E6_1[0], label="PD Controller, HToAATo4B (dim=1)",
         color=colors["higgs_dim1"], **styles["PD"])
#plt.plot(time, E6_4 / E6_4[0], label="PD Controller, HToAATo4B (dim=4)",
#         color=colors["higgs_dim4"], **styles["PD"])

# --- Axis limits and labels ---
#plt.xlim(-0.01, 1.01)
#plt.ylim(0.95, 1.4)  

plt.xlabel("Time (Fraction of Run)", loc='center')#, fontsize=20)
plt.ylabel("Relative Cumulative Efficiency", loc='center')#, fontsize=18)
#plt.xticks(fontsize=18)
#plt.yticks(fontsize=18)
plt.ylim(0.9,1.9)
plt.grid(True, linestyle="--", alpha=0.6)

# ---- Legend with headers and title ----
# Invisible header placeholders
header_const = mlines.Line2D([], [], color="none", linestyle="none")
header_pd    = mlines.Line2D([], [], color="none", linestyle="none")

# Create dummy lines for legend entries
# Constant Menu
const_ttbar_dim1  = mlines.Line2D([], [], color=colors["ttbar_dim1"],  **styles["Constant"])
#const_ttbar_dim4  = mlines.Line2D([], [], color=colors["ttbar_dim4"],  **styles["Constant"])
const_higgs_dim1  = mlines.Line2D([], [], color=colors["higgs_dim1"],  **styles["Constant"])
#const_higgs_dim4  = mlines.Line2D([], [], color=colors["higgs_dim4"],  **styles["Constant"])

# PD Controller
pd_ttbar_dim1     = mlines.Line2D([], [], color=colors["ttbar_dim1"],  **styles["PD"])
#pd_ttbar_dim4     = mlines.Line2D([], [], color=colors["ttbar_dim4"],  **styles["PD"])
pd_higgs_dim1     = mlines.Line2D([], [], color=colors["higgs_dim1"],  **styles["PD"])
#pd_higgs_dim4     = mlines.Line2D([], [], color=colors["higgs_dim4"],  **styles["PD"])

# Correct order: all Constant Menu (left column), then all PD Controller (right column)
handles = [
    header_const,           
    const_ttbar_dim1,
    # const_ttbar_dim4,   
    const_higgs_dim1, 
    #const_higgs_dim4,
    header_pd,
    pd_ttbar_dim1, 
    #pd_ttbar_dim4,
    pd_higgs_dim1, 
    #pd_higgs_dim4     
]

labels = [
    "Constant Menu", 
    "ttbar (dim=1)",
    # "ttbar (dim=4)",
    "HToAATo4B (dim=1)", 
    #"HToAATo4B (dim=4)",
    "PD Controller",
    "ttbar (dim=1)",
    # "ttbar (dim=4)",
    "HToAATo4B (dim=1)", 
    #"HToAATo4B (dim=4)",
]

# Create legend with title
leg = plt.legend(
    handles, labels,
    title="AD Trigger", # Legend title
    ncol=2, loc="best", frameon=True,
    handlelength=2, columnspacing=1, labelspacing=0.8,
    fontsize=18,
)

# Make legend title bold as well
#leg.get_title().set_fontweight("bold")
plt.ylim(0.98,1.3)
# --- Save and show ---
plt.savefig("paper/sas_rate_pidData.pdf", bbox_inches="tight")
plt.show()
plt.close()


# In[70]:


E3_1.pop(0)
E4_1.pop(0)
E3_4.pop(0)
E4_4.pop(0)
E5_1.pop(0)
E6_1.pop(0)
E5_4.pop(0)
E6_4.pop(0)




import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# Time axis normalized to [0, 1]
time = np.linspace(0, 1, len(L_E3_1))

# --- Consistent styles and colors ---
styles = {
    "Constant": {"linestyle": "dashed", "linewidth": 2.5},
    "PD": {"linestyle": "solid", "linewidth": 1.5},
}
colors = {
    "ttbar_dim1": "goldenrod",
    #"ttbar_dim4": "orangered",
    "higgs_dim1": "limegreen",
    #"higgs_dim4": "seagreen",
}

plt.figure(figsize=(10, 6))

# --- Constant Menu ---
plt.plot(time, L_E3_1 / L_E3_1[0],
         label="Constant Menu, ttbar (dim=1)",
         color=colors["ttbar_dim1"], **styles["Constant"])
#plt.plot(time, L_E3_4 / L_E3_4[0],
#         label="Constant Menu, ttbar (dim=4)",
#         color=colors["ttbar_dim4"], **styles["Constant"])

plt.plot(time, L_E5_1 / L_E5_1[0],
         label="Constant Menu, HToAATo4B (dim=1)",
         color=colors["higgs_dim1"], **styles["Constant"])
#plt.plot(time, L_E5_4 / L_E5_4[0],
#         label="Constant Menu, HToAATo4B (dim=4)",
#         color=colors["higgs_dim4"], **styles["Constant"])

# --- PD Controller ---
plt.plot(time, L_E4_1 / L_E4_1[0],
         label="PD Controller, ttbar (dim=1)",
         color=colors["ttbar_dim1"], **styles["PD"])
#plt.plot(time, L_E4_4 / L_E4_4[0],
#         label="PD Controller, ttbar (dim=4)",
#         color=colors["ttbar_dim4"], **styles["PD"])

plt.plot(time, L_E6_1 / L_E6_1[0],
         label="PD Controller, HToAATo4B (dim=1)",
         color=colors["higgs_dim1"], **styles["PD"])
#plt.plot(time, L_E6_4 / L_E6_4[0],
#         label="PD Controller, HToAATo4B (dim=4)",
#         color=colors["higgs_dim4"], **styles["PD"])

# --- Axes & labels ---
#plt.xlim(-0.01, 1.01)
#plt.ylim(0.95, 1.7)  # Optional fixed y-range
plt.xlabel("Time (Fraction of Run)", loc='center')#, fontsize=20)
plt.ylabel("Relative Efficiency", loc='center')#, fontsize=20)
#plt.xticks(fontsize=18)
#plt.yticks(fontsize=18)
plt.grid(True, linestyle="--", alpha=0.6)

# ---- Legend with column headers & title ----
header_const = mlines.Line2D([], [], color="none", linestyle="none")
header_pd    = mlines.Line2D([], [], color="none", linestyle="none")

# Dummy handles matching plot styles/colors
const_ttbar_dim1 = mlines.Line2D([], [], color=colors["ttbar_dim1"],  **styles["Constant"])
pd_ttbar_dim1    = mlines.Line2D([], [], color=colors["ttbar_dim1"],  **styles["PD"])

#const_ttbar_dim4 = mlines.Line2D([], [], color=colors["ttbar_dim4"],  **styles["Constant"])
#pd_ttbar_dim4    = mlines.Line2D([], [], color=colors["ttbar_dim4"],  **styles["PD"])

const_higgs_dim1 = mlines.Line2D([], [], color=colors["higgs_dim1"],  **styles["Constant"])
pd_higgs_dim1    = mlines.Line2D([], [], color=colors["higgs_dim1"],  **styles["PD"])

#const_higgs_dim4 = mlines.Line2D([], [], color=colors["higgs_dim4"],  **styles["Constant"])
#pd_higgs_dim4    = mlines.Line2D([], [], color=colors["higgs_dim4"],  **styles["PD"])

# Order row-by-row for ncol=2
handles = [
    header_const,          
    const_ttbar_dim1, 
    #const_ttbar_dim4,  
    const_higgs_dim1,
    # const_higgs_dim4,
    header_pd,
    pd_ttbar_dim1, 
    #pd_ttbar_dim4,
    pd_higgs_dim1, 
    #pd_higgs_dim4   
]
labels = [
    "Constant Menu", 
    "ttbar (dim=1)",
    # "ttbar (dim=4)",
    "HToAATo4B (dim=1)",
    # "HToAATo4B (dim=4)",
    "PD Controller",
    "ttbar (dim=1)",
    # "ttbar (dim=4)",
    "HToAATo4B (dim=1)", 
    #"HToAATo4B (dim=4)"
]

leg = plt.legend(
    handles, labels,
    title="AD Trigger",# title_
    fontsize=18,
    ncol=2, loc="best", frameon=True,
    handlelength=2, columnspacing=1, labelspacing=0.8
)
plt.ylim(0.9,1.6)

# --- Save & show ---
plt.savefig("paper/L_sas_rate_pidData.pdf", bbox_inches="tight")
plt.show()
plt.close()


# Local S rate
time = np.linspace(0, 1, len(L_E3_1))

# Create the plot
plt.figure(figsize=(10, 6))

# Plot R1 (constant cut) as a solid line
plt.plot(time, L_E3_1/L_E3_1[0], label=f'ttbar, model dim=1', color='goldenrod', linewidth=2.5, linestyle='dashed')
plt.plot(time, L_E3_4/L_E3_4[0], label=f'ttbar, model dim=4', color='orangered', linewidth=2.5, linestyle='dashed')

plt.plot(time, L_E5_1/L_E5_1[0], label=f'HToAATo4B, model dim=1', color='limegreen', linewidth=2.5, linestyle='dashed')
plt.plot(time, L_E5_4/L_E5_4[0], label=f'HToAATo4B, model dim=4', color='seagreen', linewidth=2.5, linestyle='dashed')


#plt.xlim(-0.1, 1.1)
# plt.ylim(0., 2.8)

# Add labels
plt.xlabel('Time (Fraction of Run)')
plt.ylabel('Relative Efficiency')


# Add a legend to differentiate the lines
plt.legend(title='AD Constant Menu',  loc='best', frameon=True)

# Add grid for better visibility
plt.grid(True, linestyle='--', alpha=0.6)


# Save the plot
plt.savefig('paper/L_sas_rate_fixed.pdf', bbox_inches='tight')

plt.show()
plt.close()

import numpy as np
import matplotlib.pyplot as plt

# --- HT Trigger: Background Rate ---
time = np.linspace(0, 1, len(R1))
plt.figure(figsize=(10, 6))
plt.plot(time, R1, label='Constant Menu', color='tab:blue', linewidth=3, linestyle='dashed')

plt.axhline(y=0.28*400, color='gray', linestyle='--', linewidth=1.5, label='Upper Tolerance (112)')
plt.axhline(y=0.22*400, color='gray', linestyle='--', linewidth=1.5, label='Lower Tolerance (88)')

plt.xlabel('Time (Fraction of Run)', loc='center')
plt.ylabel('Background Rate [kHz]', loc='center')
plt.legend(title='HT Trigger', loc='best', frameon=True)
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig('paper/bht_rate_fixed.pdf', bbox_inches='tight', dpi=300)
plt.show()
plt.close()


# --- AD Trigger: Background Rate ---
time = np.linspace(0, 1, len(E1_1))
plt.figure(figsize=(10, 6))
plt.plot(time, E1_1, label='Constant Menu, model dim=1', color='tab:blue', linewidth=3, linestyle='dotted')
plt.plot(time, E1_4, label='Constant Menu, model dim=4', color='tab:blue', linewidth=3, linestyle='dashed')

plt.axhline(y=0.28*400, color='gray', linestyle='--', linewidth=1.5, label='Upper Tolerance (112)')
plt.axhline(y=0.22*400, color='gray', linestyle='--', linewidth=1.5, label='Lower Tolerance (88)')

plt.xlabel('Time (Fraction of Run)', loc='center')
plt.ylabel('Background Rate [kHz]', loc='center')
plt.ylim(0,300)
plt.legend(title='AD Trigger', loc='best', frameon=True)
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig('paper/bas_rate_fixed.pdf', bbox_inches='tight', dpi=300)
plt.show()
plt.close()


# --- HT Trigger: Cumulative Signal Efficiency ---
time = np.linspace(0, 1, len(R3))
plt.figure(figsize=(10, 6))
plt.plot(time, R3/R3[0], label='ttbar', color='goldenrod', linewidth=2.5, linestyle='dashed')
plt.plot(time, R5/R5[0], label='HToAATo4B', color='seagreen', linewidth=2.5, linestyle='dashed')

plt.xlabel('Time (Fraction of Run)', loc='center')
plt.ylabel('Relative Cumulative Efficiency', loc='center')
plt.legend(title='HT Constant Menu', frameon=True)
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig('paper/sht_rate_fixed.pdf', bbox_inches='tight', dpi=300)
plt.show()
plt.close()


# --- HT Trigger: Local Signal Efficiency ---
time = np.linspace(0, 1, len(R3))
plt.figure(figsize=(10, 6))
plt.plot(time, L_R3/L_R3[0], label='ttbar', color='goldenrod', linewidth=2.5, linestyle='dashed')
plt.plot(time, L_R5/L_R5[0], label='HToAATo4B', color='seagreen', linewidth=2.5, linestyle='dashed')

plt.xlabel('Time (Fraction of Run)', loc='center')
plt.ylabel('Relative Efficiency', loc='center')
plt.legend(title='HT Constant Menu', frameon=True)
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig('paper/L_sht_rate_fixed.pdf', bbox_inches='tight', dpi=300)
plt.show()
plt.close()


# --- AD Trigger: Cumulative Signal Efficiency ---
time = np.linspace(0, 1, len(E3_1))
plt.figure(figsize=(10, 6))
plt.plot(time, E3_1/E3_1[0], label='ttbar, model dim=1', color='goldenrod', linewidth=2.5, linestyle='dashed')
plt.plot(time, E3_4/E3_4[0], label='ttbar, model dim=4', color='orangered', linewidth=2.5, linestyle='dashed')
plt.plot(time, E5_1/E5_1[0], label='HToAATo4B, model dim=1', color='limegreen', linewidth=2.5, linestyle='dashed')
plt.plot(time, E5_4/E5_4[0], label='HToAATo4B, model dim=4', color='seagreen', linewidth=2.5, linestyle='dashed')

plt.xlabel('Time (Fraction of Run)', loc='center')
plt.ylim(0.8,2)
plt.ylabel('Relative Cumulative Efficiency', loc='center')
plt.legend(title='AD Constant Menu', loc='best', frameon=True)
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig('paper/sas_rate_fixed.pdf', bbox_inches='tight', dpi=300)
plt.show()
plt.close()


# --- AD Trigger: Local Signal Efficiency ---



import numpy as np
import matplotlib.pyplot as plt

# ========== Common styles ==========
styles = {
    "Constant": {"linestyle": "dashed", "linewidth": 2.5},
    "PD": {"linestyle": "solid", "linewidth": 1.8},
}
colors = {
    "ttbar": "goldenrod",
    "HToAATo4B": "limegreen",
}

# =========================================================
# === 1. HT Trigger: Background Rate ======================
# =========================================================
time = np.linspace(0, 1, len(R1))
plt.figure(figsize=(10, 6))
plt.plot(time, R1, label="Constant Menu", color='tab:blue', **styles["Constant"])
plt.plot(time, R2, label="PD Controller", color='mediumblue', **styles["PD"])
plt.axhline(y=0.28 * 400, color='gray', linestyle='--', linewidth=1.5, label='Upper Tolerance (112)')
plt.axhline(y=0.22 * 400, color='gray', linestyle='--', linewidth=1.5, label='Lower Tolerance (88)')
plt.xlabel("Time (Fraction of Run)", loc='center')
plt.ylabel("Background Rate [kHz]", loc='center')
plt.legend(title="HT Trigger", frameon=True)
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig("paper/bht_rate_pidData2data.png", bbox_inches='tight', dpi=300)
plt.close()

# =========================================================
# === 2. AD Trigger: Background Rate ======================
# =========================================================
time = np.linspace(0, 1, len(E1_1))
plt.figure(figsize=(10, 6))
plt.plot(time, E1_1, label="Constant Menu", color='tab:blue', **styles["Constant"])
plt.plot(time, E2_1, label="PD Controller", color='mediumblue', **styles["PD"])
plt.axhline(y=0.28 * 400, color='gray', linestyle='--', linewidth=1.5, label='Upper Tolerance (112)')
plt.axhline(y=0.22 * 400, color='gray', linestyle='--', linewidth=1.5, label='Lower Tolerance (88)')
plt.xlabel("Time (Fraction of Run)", loc='center')
plt.ylabel("Background Rate [kHz]", loc='center')
plt.legend(title="AD Trigger", frameon=True)
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig("paper/bas_rate_pidData2data.png", bbox_inches='tight', dpi=300)
plt.close()

# =========================================================
# === 3. HT Trigger: Cumulative Efficiency ================
# =========================================================
colors = {
    "ttbar": "goldenrod",
    "HToAATo4B": "seagreen",  # changed here for better contrast
}
eff0_ttbar = R3[0]
eff0_AA = R5[0]
time = np.linspace(0, 1, len(R3))
plt.figure(figsize=(10, 6))
plt.plot(time, R3/eff0_ttbar,
         label=fr"Constant Menu, ttbar ($\epsilon[t_0]={eff0_ttbar:.2f}$)",
         color=colors["ttbar"], **styles["Constant"])
plt.plot(time, R5/eff0_AA,
         label=fr"Constant Menu, HToAATo4B ($\epsilon[t_0]={eff0_AA:.2f}$)",
         color=colors["HToAATo4B"], **styles["Constant"])
eff0_ttbar_PD = R4[0]
eff0_AA_PD = R6[0]
plt.plot(time, R4/eff0_ttbar_PD,
         label=fr"PD Controller, ttbar ($\epsilon[t_0]={eff0_ttbar_PD:.2f}$)",
         color=colors["ttbar"], **styles["PD"])
plt.plot(time, R6/eff0_AA_PD,
         label=fr"PD Controller, HToAATo4B ($\epsilon[t_0]={eff0_AA_PD:.2f}$)",
         color=colors["HToAATo4B"], **styles["PD"])
plt.xlabel("Time (Fraction of Run)", loc='center')
plt.ylabel("Relative Cumulative Efficiency", loc='center')
plt.grid(True, linestyle="--", alpha=0.6)
plt.ylim(0.85, 1.5)
plt.legend(title="HT Trigger", fontsize=16, frameon=True, loc="best")
plt.savefig("paper/sht_rate_pidData2data.pdf", bbox_inches="tight", dpi=300)
plt.close()

# =========================================================
# === 4. AD Trigger: Cumulative Efficiency ================
# =========================================================
colors = {"ttbar": "goldenrod",
"HToAATo4B": "limegreen"}
eff0_ttbar_const = E3_1[0]
eff0_AA_const = E5_1[0]
eff0_ttbar_pd = E4_1[0]
eff0_AA_pd = E6_1[0]
time = np.linspace(0, 1, len(E3_1))
plt.figure(figsize=(10, 6))
plt.plot(time, E3_1/eff0_ttbar_const,
         label=fr"Constant Menu, ttbar ($\epsilon[t_0]={eff0_ttbar_const:.2f}$)",
         color=colors["ttbar"], **styles["Constant"])
plt.plot(time, E5_1/eff0_AA_const,
         label=fr"Constant Menu, HToAATo4B ($\epsilon[t_0]={eff0_AA_const:.2f}$)",
         color=colors["HToAATo4B"], **styles["Constant"])
plt.plot(time, E4_1/eff0_ttbar_pd,
         label=fr"PD Controller, ttbar ($\epsilon[t_0]={eff0_ttbar_pd:.2f}$)",
         color=colors["ttbar"], **styles["PD"])
plt.plot(time, E6_1/eff0_AA_pd,
         label=fr"PD Controller, HToAATo4B ($\epsilon[t_0]={eff0_AA_pd:.2f}$)",
         color=colors["HToAATo4B"], **styles["PD"])
plt.xlabel("Time (Fraction of Run)", loc='center')
plt.ylabel("Relative Cumulative Efficiency", loc='center')
plt.grid(True, linestyle="--", alpha=0.6)
plt.ylim(0.98, 1.3)
plt.legend(title="AD Trigger", fontsize=16, frameon=True, loc="best")
plt.savefig("paper/sas_rate_pidData2data.pdf", bbox_inches="tight", dpi=300)
plt.close()

# =========================================================
# === 5. HT Trigger: Local Efficiency =====================
# =========================================================
colors = {
    "ttbar": "goldenrod",
    "HToAATo4B": "seagreen",  # changed here for better contrast
}

eff0_ttbar = L_R3[0]
eff0_AA = L_R5[0]
eff0_ttbar_PD = L_R4[0]
eff0_AA_PD = L_R6[0]
time = np.linspace(0, 1, len(L_R3))
plt.figure(figsize=(10, 6))
plt.plot(time, L_R3/eff0_ttbar,
         label=fr"Constant Menu, ttbar ($\epsilon[t_0]={eff0_ttbar:.2f}$)",
         color=colors["ttbar"], **styles["Constant"])
plt.plot(time, L_R5/eff0_AA,
         label=fr"Constant Menu, HToAATo4B ($\epsilon[t_0]={eff0_AA:.2f}$)",
         color=colors["HToAATo4B"], **styles["Constant"])
plt.plot(time, L_R4/eff0_ttbar_PD,
         label=fr"PD Controller, ttbar ($\epsilon[t_0]={eff0_ttbar_PD:.2f}$)",
         color=colors["ttbar"], **styles["PD"])
plt.plot(time, L_R6/eff0_AA_PD,
         label=fr"PD Controller, HToAATo4B ($\epsilon[t_0]={eff0_AA_PD:.2f}$)",
         color=colors["HToAATo4B"], **styles["PD"])
plt.xlabel("Time (Fraction of Run)", loc='center')
plt.ylabel("Relative Efficiency", loc='center')
plt.grid(True, linestyle="--", alpha=0.6)
plt.ylim(0, 1.6)
plt.legend(title="HT Trigger", fontsize=15, frameon=True, loc="best")
plt.savefig("paper/L_sht_rate_pidData2data.pdf", bbox_inches="tight", dpi=300)
plt.close()

# =========================================================
# === 6. AD Trigger: Local Efficiency =====================
# =========================================================
# =========================================================
colors = {
    "ttbar": "goldenrod",
    "HToAATo4B": "limegreen"}
eff0_ttbar_const = L_E3_1[0]
eff0_AA_const = L_E5_1[0]
eff0_ttbar_pd = L_E4_1[0]
eff0_AA_pd = L_E6_1[0]
time = np.linspace(0, 1, len(L_E3_1))
plt.figure(figsize=(10, 6))
plt.plot(time, L_E3_1/eff0_ttbar_const,
         label=fr"Constant Menu, ttbar ($\epsilon[t_0]={eff0_ttbar_const:.2f}$)",
         color=colors["ttbar"], **styles["Constant"])
plt.plot(time, L_E5_1/eff0_AA_const,
         label=fr"Constant Menu, HToAATo4B ($\epsilon[t_0]={eff0_AA_const:.2f}$)",
         color=colors["HToAATo4B"], **styles["Constant"])
plt.plot(time, L_E4_1/eff0_ttbar_pd,
         label=fr"PD Controller, ttbar ($\epsilon[t_0]={eff0_ttbar_pd:.2f}$)",
         color=colors["ttbar"], **styles["PD"])
plt.plot(time, L_E6_1/eff0_AA_pd,
         label=fr"PD Controller, HToAATo4B ($\epsilon[t_0]={eff0_AA_pd:.2f}$)",
         color=colors["HToAATo4B"], **styles["PD"])
plt.xlabel("Time (Fraction of Run)", loc='center')
plt.ylabel("Relative Efficiency", loc='center')
plt.grid(True, linestyle="--", alpha=0.6)
plt.ylim(0.9, 1.6)
plt.legend(title="AD Trigger", fontsize=15, frameon=True, loc="best")
plt.savefig("paper/L_sas_rate_pidData2data.pdf", bbox_inches="tight", dpi=300)
plt.close()
