import numpy as np
import matplotlib.pyplot as plt
import time
import os
import argparse

# write a parser to set up the running from command line
parser = argparse.ArgumentParser(description='Tranforms Trigger Primitives to images.')
parser.add_argument('-filename', type=str, default='fd-snb-mc-hits/snana_hits_apa0.txt', help='path to the file with TPs')
parser.add_argument('-chanMap', type=str, default='FDHDChannelMap_v1_wireends.txt', help='path to the file with Channel Map')
parser.add_argument('-show', action='store_true', help='show the image')
parser.add_argument('-save', action='store_true', help='save the image')
parser.add_argument('-save_path', type=str, default='TP_to_img/images/', help='path to save the image')
parser.add_argument('-n_events', type=int, default=1000, help='number of events to process')

args = parser.parse_args()
# unpack the arguments
filename = args.filename
chanMap = args.chanMap
show = args.show
save = args.save
save_path = args.save_path
n_events = args.n_events




'''
TP format:  
[0] time_start
[1] time_over_threshold
[2] time_peak
[3] channel
[4] adc_integral
[5] adc_peak
[6] detID
[7] type
'''


'''
Channel Map format:
[0] offlchan    in gdml and channel sorting convention
[1] upright     0 for inverted, 1 for upright
[2] wib         1, 2, 3, 4 or 5  (slot number +1?)
[3] link        link identifier: 0 or 1
[4] femb_on_link    which of two FEMBs in the WIB frame this FEMB is:  0 or 1
[5] cebchan     cold electronics channel on FEMB:  0 to 127
[6] plane       0: U,  1: V,  2: X
[7] chan_in_plane   which channel this is in the plane in the FEMB:  0:39 for U and V, 0:47 for X
[8] femb       which FEMB on an APA -- 1 to 20
[9] asic       ASIC:   1 to 8
[10] asicchan   ASIC channel:  0 to 15
[11] wibframechan   channel index in WIB frame (used with get_adc in detdataformats/WIB2Frame.hh).  0:255
'''




if __name__=='__main__':

    #read first n_events lines in the data from txt, all are ints
    all_TPs = np.loadtxt(filename, skiprows=0, max_rows=n_events, dtype=int)
    print(all_TPs.shape)

    #read channel map
    channel_map = np.loadtxt(chanMap, skiprows=0, dtype=int)
    print(channel_map.shape)
    




