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



#read first n_events lines in the data from txt, all are ints
all_TPs = np.loadtxt(filename, skiprows=0, max_rows=n_events, dtype=int)

#read channel map
channel_map = np.loadtxt(chanMap, skiprows=0, dtype=int)


def from_tp_to_img_by_time(all_tps, t_start, t_end, tp_type):
    '''
    :param all_tps: all tps in the data
    :param t_start: start time of the image
    :param t_end: end time of the image
    :return: image
    '''
    #select TPs in the time range
    tps = all_tps[np.where((all_tps[:, 0] > t_start) & (all_tps[:, 0] + all_tps[:, 1] < t_end ))]
    #select TPs in the type
    tps = tps[np.where(tps[:, 7] == tp_type)]
    #create image
    x_max = int(tps[:, 3].max()+1)
    x_min = int(tps[:, 3].min()-1)
    x_range = x_max - x_min
    y_range = int((t_end - t_start))

    img_width = 500
    img_height = 1000

    img = np.zeros((img_width, img_height))
    #fill image
    for tp in tps:
        x = (tp[3] - x_min)/x_range * img_width
        y_start = (tp[0] - t_start)/y_range * img_height
        y_end = (tp[0] + tp[1] - t_start)/y_range * img_height

        img[int(x), int(y_start):int(y_end)] += tp[4]/(y_end - y_start)

    return img

def from_tp_to_img(tps, tp_type):
    '''
    :param all_tps: all tps in the data
    :param t_start: start time of the image
    :param t_end: end time of the image
    :return: image
    '''

    t_start = tps[0, 0]
    t_end = tps[-1, 0] + tps[-1, 1]

    #select TPs in the type
    tps = tps[np.where(tps[:, 7] == tp_type)]
    #create image
    x_max = int(tps[:, 3].max()+1)
    x_min = int(tps[:, 3].min()-1)


    x_range = x_max - x_min 
    y_range = int((t_end - t_start))
    x_margin = int(x_range/2)

    img_width = np.min([500, x_range + 2*x_margin])
    img_height = np.min([1000, y_range ])

    img = np.zeros((img_width, img_height))
    #fill image
    for tp in tps:
        x = (tp[3] - x_min)/x_range * (img_width-2*x_margin) + x_margin
        y_start = (tp[0] - t_start)/y_range * img_height
        y_end = (tp[0] + tp[1] - t_start)/y_range * img_height 

        img[int(x), int(y_start):int(y_end)] += tp[4]/(y_end - y_start)

    return img

def create_channel_map_dict(channel_map):
    '''
    :param channel_map: channel map
    :return: dictionary with key: channel, value: [upright, plane]
    '''
    #pericolo, due volte lo stesso numero di canali per le due versioni di upright, qui si sovrascrivono

    channel_map_dict = {}
    for elem in channel_map:
        channel_map_dict[elem[0]] = [elem[1], elem[6]]
    
    return channel_map_dict

def add_channel_map_to_tps(all_tps, channel_map_dict):
    '''
    :param all_tps: all tps in the data
    :param channel_map_dict: dictionary with key: channel, value: [crate, upright, plane]
    :return: all_tps with crate, upright, plane added
    '''
    all_tps_with_map = np.zeros((all_tps.shape[0], all_tps.shape[1]+2), dtype=int)
    all_tps_with_map[:, :8] = all_tps
    for i, tp in enumerate(all_tps):
        all_tps_with_map[i, 8:] = channel_map_dict[tp[3]]

    return all_tps_with_map


def find_candidates(all_tps, lim_channel, lim_time, tp_type):
    '''
    :param all_tps: all tps in the data
    :param lim_channel: limit of channel difference
    :param lim_time: limit of time difference
    :return: candidates

    a candidate is a list of TPs which are close in time and space
    '''
    candidates = []
    buffer = []


    #select TPs in the type
    tps = all_tps[np.where(all_tps[:, 7] == tp_type)]
    #sort TPs by time
    tps = tps[tps[:, 2].argsort()]
    #find candidates
    for tp in tps:
        if len(buffer) == 0:
            buffer.append([tp])
        else:
            appended = False
            for i, elem in enumerate(buffer):
                if (tp[2] - elem[-1][2]) < lim_time:
                    if abs(tp[3] - elem[-1][3]) < lim_channel:
                        elem.append(tp)
                        appended = True
                elif len(elem) > 3:
                    candidates.append(elem)
                    buffer.pop(i)

                else:
                    buffer.pop(i)

            if not appended:
                buffer.append([tp])

    return candidates




if __name__=='__main__':
    print(all_TPs.shape)


    print(channel_map.shape)

    # save channel map in two different files
    np.savetxt('channel_map_upright.txt', channel_map[np.where(channel_map[:, 1] == 1)], fmt='%i')
    np.savetxt('channel_map_inverted.txt', channel_map[np.where(channel_map[:, 1] == 0)], fmt='%i')
    

    channel_map_dict = create_channel_map_dict(channel_map)

    all_TPs_with_map = add_channel_map_to_tps(all_TPs, channel_map_dict)

    # count the possible values of the upright
    print(all_TPs_with_map[:, 8])
    print(np.unique(all_TPs_with_map[:, 8]))

    # count the possible values of the plane
    print(np.unique(all_TPs_with_map[:, 9]))




    print(all_TPs_with_map.shape)
    print(all_TPs_with_map[0])

    U_tps = all_TPs_with_map[np.where((all_TPs_with_map[:, 7] == 1) & (all_TPs_with_map[:, 8] == 0) & (all_TPs_with_map[:, 9] == 0))]
    V_tps = all_TPs_with_map[np.where((all_TPs_with_map[:, 7] == 1) & (all_TPs_with_map[:, 8] == 1) & (all_TPs_with_map[:, 9] == 0))]
    X_tps = all_TPs_with_map[np.where((all_TPs_with_map[:, 7] == 1) & (all_TPs_with_map[:, 8] == 2) & (all_TPs_with_map[:, 9] == 0))]

    print(U_tps.shape)
    print(V_tps.shape)
    print(X_tps.shape)

    # candidates = find_candidates(all_TPs, 100, 1000)

    # # make candidates numpy array
    # for i, candidate in enumerate(candidates):
    #     candidates[i] = np.array(candidate)


    # print(len(candidates))

    # #plot candidates
    # for candidate in candidates[:9]:
    #     # scatter plot
    #     plt.scatter(candidate[:, 2], candidate[:, 3])
        

    #     plt.show()

    # print(all_TPs[39:46])


    # img = from_tp_to_img(all_TPs[39:46], 1)
    # print(img.shape)

    # plt.imshow(np.rot90(img))
    # plt.colorbar()
    # if save:    
    #     if not os.path.exists(save_path):
    #         os.makedirs(save_path)
    #     plt.savefig(save_path+'tp_image.png')
    
    # if show:
    #     plt.show()


    
    # candidates=find_candidates(all_TPs, 100, 1500, 1)

    # print(len(candidates))

    # for candidate in candidates[:9]:

    #     candidate = np.array(candidate)
    #     img = from_tp_to_img(candidate,1)

        
    #     fig= plt.figure(figsize=(20, 20))
    #     plt.imshow(np.rot90(img))
    #     plt.colorbar()
    #     if show:
    #         plt.show()
    
    #     if save:    
    #         if not os.path.exists(save_path):
    #             os.makedirs(save_path)
    #         plt.savefig(save_path+'tp_image.png')  