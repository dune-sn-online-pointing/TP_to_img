import numpy as np
import matplotlib.pyplot as plt
import time
import os
import argparse

# write a parser to set up the running from command line
parser = argparse.ArgumentParser(description='Tranforms Trigger Primitives to images.')
parser.add_argument('-filename', type=str, default='fd-snb-mc-hits/snana_hits_apa0.txt', help='path to the file with TPs')
parser.add_argument('-chanmap', type=str, default='FDHDChannelMap_v1_wireends.txt', help='path to the file with Channel Map')
parser.add_argument('-show', action='store_true', help='show the image')
parser.add_argument('-save', action='store_true', help='save the image')
parser.add_argument('-save_path', type=str, default='images/', help='path to save the image')
parser.add_argument('-n_events', type=int, default=0, help='number of events to process')

args = parser.parse_args()
# unpack the arguments
filename = args.filename
chanMap = args.chanmap
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
def create_channel_map_array(chanMap):
    '''
    :param chanMap: path to the channel map 
    :return: channel map array
    '''
    channel_map = np.loadtxt(chanMap, skiprows=0, dtype=int)
    # sort the channel map by the first column (offlchan)
    channel_map = channel_map[np.argsort(channel_map[:, 0])]
    return channel_map


def from_tp_to_imgs(tps, make_fixed_size=False, width=500, height=1000, x_margin=10, y_margin=500):
    '''
    :param tps: all trigger primitives to draw
    :return: image
    '''
    t_start = tps[0, 0] - y_margin
    t_end = tps[-1, 0] + tps[-1, 1] + y_margin
    
    x_max = (tps[:, 3].max()+x_margin)
    x_min = (tps[:, 3].min()-x_margin)

    x_range = x_max - x_min
    y_range = int((t_end - t_start))

    # create the image
    if make_fixed_size:
        img_width = width
        img_height = height
    else:
        img_width = np.min([width, x_range + 2*x_margin])
        img_height = np.min([height, y_range + 2*y_margin])
    img = np.zeros((img_height, img_width))
    # fill image
    for tp in tps:
        x = (tp[3] - x_min)/x_range * img_width
        y_start = (tp[0] - t_start)/y_range * img_height
        y_end = (tp[0] + tp[1] - t_start)/y_range * img_height
        img[int(y_start):int(y_end), int(x)] = tp[4]/(y_end - y_start)

    return img



def all_views_img_maker(tps, channel_map, min_tps=2):
    '''
    :param tps: all trigger primitives in the event
    :param channel_map: channel map
    :param min_tps: minimum number of TPs to create the image
    :return: images or -1 if there are not enough TPs
    '''
    # U plane, take only the tps where the corrisponding position in the channel map is 0      
    tps_u = tps[np.where(channel_map[tps[:, 3], 6] == 0)]

    # V plane, take only the tps where the corrisponding position in the channel map is 1
    tps_v = tps[np.where(channel_map[tps[:, 3], 6] == 1)]

    # X plane, take only the tps where the corrisponding position in the channel map is 2
    tps_x = tps[np.where(channel_map[tps[:, 3], 6] == 2)]

    img_u, img_v, img_x = np.array([[-1]]), np.array([[-1]]), np.array([[-1]])
    
    if tps_u.shape[0] > min_tps:
        img_u = from_tp_to_imgs(tps_u)

    if tps_v.shape[0] > min_tps: 
        img_v = from_tp_to_imgs(tps_v)

    if tps_x.shape[0] > min_tps:
        img_x = from_tp_to_imgs(tps_x)

    return img_u, img_v, img_x



def cluster_maker(all_tps, channel_map, ticks_limit=100, channel_limit=20, min_hits=2):
    '''
    :param all_tps: all trigger primitives in the event
    :param channel_map: channel map
    :param ticks_limit: maximum time window to consider
    :param channel_limit: maximum number of channels to consider
    :param min_hits: minimum number of hits to consider
    :return: list of clusters
    '''

    # create a list of clusters
    clusters = []
    buffer = []
    # loop over the TPs
    for tp in all_tps:
        if len(buffer) == 0:
            buffer.append([tp])
        else:
            appended = False
            for i, elem in enumerate(buffer):
                if (tp[2] - elem[-1][2]) < ticks_limit:
                    if abs(tp[3] - elem[-1][3]) < channel_limit: 
                         
                        elem.append(tp)
                        appended = True
                elif len(elem) > 3:
                    clusters.append(elem)
                    buffer.pop(i)

                else:
                    buffer.pop(i)

            if not appended:
                buffer.append([tp])

    return clusters


def show_or_save_img(all_TPs, channel_map, show=False, save=False, save_path='TP_to_img/images/', outname='test'):
    '''
    :param img: image to show or save
    :param show: show the image
    :param save: save the image
    :param save_path: path where to save the image
    :param filename: name of the file
    :return: None
    '''
    #create images
    img_u, img_v, img_x = all_views_img_maker(all_TPs, channel_map)

    #show images
    if show:
        if img_u[0, 0] != -1:
            plt.figure(figsize=(8, 20))
            plt.imshow(img_u)
            plt.show()
        if img_v[0, 0] != -1:
            plt.figure(figsize=(8, 20))
            plt.imshow(img_v)
            plt.show()
        if img_x[0, 0] != -1:
            plt.figure(figsize=(8, 20))
            plt.imshow(img_x)
            plt.show()

    #save images
    if save:    
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        if img_u[0, 0] != -1:
            plt.imsave(save_path + 'u_' + os.path.basename(outname) + '.png', img_u)
        if img_v[0, 0] != -1:
            plt.imsave(save_path + 'v_' + os.path.basename(outname)+ '.png', img_v)
        if img_x[0, 0] != -1:
            plt.imsave(save_path + 'x_' + os.path.basename(outname) + '.png', img_x)


if __name__=='__main__':

    #read first n_events lines in the data from txt, all are ints
    if n_events:
        all_TPs = np.loadtxt(filename, skiprows=0, max_rows=n_events, dtype=int)
    else:
        all_TPs = np.loadtxt(filename, skiprows=0, dtype=int)
    
    print(all_TPs.shape)
    print(np.unique(all_TPs[:, 3]).shape)
    #read channel map
    channel_map = create_channel_map_array(chanMap)
    print(channel_map.shape)
    print((channel_map[:, 0]))
    
    clusters = cluster_maker(all_TPs, channel_map)
    for i, cluster in enumerate(clusters):
        show_or_save_img(np.array(cluster), channel_map, show=show, save=save, save_path=save_path, outname='test'+str(i))


