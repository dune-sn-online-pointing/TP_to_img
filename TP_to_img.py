import numpy as np
import matplotlib.pyplot as plt
import time
import os
import argparse
import warnings
import sys
import gc

# write a parser to set up the running from command line
parser = argparse.ArgumentParser(description='Tranforms Trigger Primitives to images.')
parser.add_argument('-filename', type=str, default='fd-snb-mc-hits/snana_hits_apa0.txt', help='path to the file with TPs')
parser.add_argument('-chanmap', type=str, default='FDHDChannelMap_v1_wireends.txt', help='path to the file with Channel Map')
parser.add_argument('-show', action='store_true', help='show the image')
parser.add_argument('-save', action='store_true', help='save the image')
parser.add_argument('-write', action='store_true', help='write the clusters to a file')
parser.add_argument('-save_path', type=str, default='images/', help='path to save the image')
parser.add_argument('-n_events', type=int, default=0, help='number of events to process')

args = parser.parse_args()
# unpack the arguments
filename = args.filename
chanMap = args.chanmap
show = args.show
save = args.save
write = args.write
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


def from_tp_to_imgs(tps, make_fixed_size=False, width=500, height=1000, x_margin=10, y_margin=100):
    '''
    :param tps: all trigger primitives to draw
    :param make_fixed_size: if True, the image will have fixed size, otherwise it will be as big as the TPs
    :param width: width of the image
    :param height: height of the image
    :param x_margin: margin on the x axis
    :param y_margin: margin on the y axis
    :return: image
    '''
    t_start = tps[0, 0]
    t_end = tps[-1, 0] + tps[-1, 1]
    
    x_max = (tps[:, 3].max())
    x_min = (tps[:, 3].min())

    x_range = x_max - x_min 
    y_range = int((t_end - t_start))

    # create the image
    if make_fixed_size:
        img_width = width
        img_height = height
    else:
        img_width =  x_range + 2*x_margin
        img_height =  y_range + 2*y_margin
    img = np.zeros((img_height, img_width))
    # fill image
    if (not make_fixed_size):
        for tp in tps:
            x = (tp[3] - x_min) + x_margin
            y_start = (tp[0] - t_start) + y_margin
            y_end = (tp[0] + tp[1] - t_start) + y_margin
            img[int(y_start):int(y_end), int(x)] = tp[4]/(y_end - y_start)


    else:
    # We stretch the image inwards if needed but we do not upscale it. In this second case we build a padded image
        if img_width < x_range:
            stretch_x = True
            print('Warning: image width is smaller than the range of the TPs. The image will be stretched inwards.')

        else:
            x_margin = (img_width - x_range)/2
            stretch_x = False
        
        if img_height < y_range:
            stretch_y = True
            print('Warning: image height is smaller than the range of the TPs. The image will be stretched inwards.')

        else:
            y_margin = (img_height - y_range)/2
            stretch_y = False


        if stretch_x & stretch_y:
            for tp in tps:
                x=(tp[3] - x_min)/x_range * (img_width - 2*x_margin) + x_margin
                y_start = (tp[0] - t_start)/y_range * (img_height - 2*y_margin) + y_margin
                y_end = (tp[0] + tp[1] - t_start)/y_range * (img_height - 2*y_margin) + y_margin
                img[int(y_start):int(y_end), int(x)] = tp[4]/(y_end - y_start)
        elif stretch_x:
            for tp in tps:                
                x=(tp[3] - x_min)/x_range * (img_width - 2*x_margin) + x_margin
                y_start = (tp[0] - t_start) + y_margin
                y_end = (tp[0] + tp[1] - t_start) + y_margin
                img[int(y_start):int(y_end), int(x)] = tp[4]/(y_end - y_start)
        elif stretch_y:
            for tp in tps:
                x = (tp[3] - x_min) + x_margin
                y_start = (tp[0] - t_start)/y_range * (img_height - 2*y_margin) + y_margin
                y_end = (tp[0] + tp[1] - t_start)/y_range * (img_height - 2*y_margin) + y_margin
                img[int(y_start):int(y_end), int(x)] = tp[4]/(y_end - y_start)

        else:
            for tp in tps:
                x = (tp[3] - x_min) + x_margin
                y_start = (tp[0] - t_start) + y_margin
                y_end = (tp[0] + tp[1] - t_start) + y_margin
                img[int(y_start):int(y_end), int(x)] = tp[4]/(y_end - y_start)
   
    return img




def all_views_img_maker(tps, channel_map, min_tps=2, make_fixed_size=False, width=500, height=1000, x_margin=10, y_margin=200):
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
        img_u = from_tp_to_imgs(tps_u, make_fixed_size=make_fixed_size, width=width, height=height, x_margin=x_margin, y_margin=y_margin )

    if tps_v.shape[0] > min_tps: 
        img_v = from_tp_to_imgs(tps_v, make_fixed_size=make_fixed_size, width=width, height=height, x_margin=x_margin, y_margin=y_margin)

    if tps_x.shape[0] > min_tps:
        img_x = from_tp_to_imgs(tps_x, make_fixed_size=make_fixed_size, width=width, height=height, x_margin=x_margin, y_margin=y_margin)

    return img_u, img_v, img_x



def cluster_maker(all_tps, channel_map, ticks_limit=100, channel_limit=20, min_hits=4):
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
        tp = [tp[0], tp[1], tp[2], tp[3], tp[4], tp[5], tp[6], tp[7]]
        if len(buffer) == 0:
            buffer.append([tp])
        else:
            appended = False
            buffer_copy = buffer.copy()
            buffer = []
            for i, elem in enumerate(buffer_copy):
                if (tp[2] - elem[-1][2]) < ticks_limit:
                    if abs(tp[3] - elem[-1][3]) < channel_limit:                         
                        elem.append(tp)
                        appended = True
                    buffer.append(elem)
                elif len(elem) >= min_hits:
                    clusters.append(elem)
            if not appended:
                buffer.append([tp])
    
    return clusters


def show_or_save_img(all_TPs, channel_map, show=False, save=False, save_path='TP_to_img/images/', outname='test', min_tps=2, make_fixed_size=False, width=500, height=1000, x_margin=10, y_margin=200):
    '''
    :param img: image to show or save
    :param show: show the image
    :param save: save the image
    :param save_path: path where to save the image
    :param filename: name of the file
    :return: None
    '''
    #create images
    img_u, img_v, img_x = all_views_img_maker(all_TPs, channel_map, min_tps=min_tps, make_fixed_size=make_fixed_size, width=width, height=height, x_margin=x_margin, y_margin=y_margin)

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


def create_dataset(clusters, make_fixed_size=True, width=70, height=1000, x_margin=5, y_margin=50):
    '''
    :param clusters: list of clusters
    :param make_fixed_size: if True, the image will have fixed size, otherwise it will be as big as the TPs
    :param width: width of the image
    :param height: height of the image
    :param x_margin: margin on the x axis
    :param y_margin: margin on the y axis
    :return: dataset [[img],[label]] in numpy array format
    '''
    dataset_img = np.array([from_tp_to_imgs(np.array(clusters[0]), make_fixed_size=make_fixed_size, width=width, height=height, x_margin=x_margin, y_margin=y_margin)])
    dataset_label = np.array([[-1]])
    img = from_tp_to_imgs(np.array(clusters[0]))
    cluster = np.array(clusters[0])
    if len(np.unique(cluster[:, 7])) > 1:
        dataset_label[0, 0] = 10
    else:
        dataset_label[0, 0] = cluster[0, 7]
    for cluster in clusters[1:]:
        # create the image
        cluster = np.array(cluster)
        img = from_tp_to_imgs(cluster, make_fixed_size=make_fixed_size, width=width, height=height, x_margin=x_margin, y_margin=y_margin)
        # create the label
        if len(np.unique(cluster[:, 7])) > 1:
            label = 10
        else:
            label = cluster[0, 7]        
        # append to the dataset as an array of arrays
        dataset_img = np.concatenate((dataset_img, np.array([img])), axis=0)
        dataset_label = np.concatenate((dataset_label, np.array([[label]])), axis=0)
    print( dataset_img.size * dataset_img.itemsize/10e9)
    print(sys.getsizeof(dataset_img[0]))
    print(dataset_img.shape)
    return (dataset_img, dataset_label)

def create_dataset2(clusters, make_fixed_size=True, width=70, height=1000, x_margin=5, y_margin=50):
    '''
    :param clusters: list of clusters
    :param make_fixed_size: if True, the image will have fixed size, otherwise it will be as big as the TPs
    :param width: width of the image
    :param height: height of the image
    :param x_margin: margin on the x axis
    :param y_margin: margin on the y axis
    :return: dataset [[img],[label]] in numpy array format
    '''
    dataset_img = []
    dataset_label = []

    for i, cluster in enumerate(clusters):
        # create the image
        img = from_tp_to_imgs(np.array(cluster), make_fixed_size=make_fixed_size, width=width, height=height, x_margin=x_margin, y_margin=y_margin)
        # create the label
        if len(np.unique(np.array(cluster)[:, 7])) > 1:
            label = 10
        else:
            label = cluster[0][7]        
        # append to the dataset as an array of arrays
        dataset_img.append(img)
        dataset_label.append([label])
    print(sys.getsizeof(dataset_img))
    print(sys.getsizeof(dataset_img[0]))
    print(sys.getsizeof(dataset_img[0][0]))
    print(sys.getsizeof(dataset_img[0][0][0]))
    
    print((sys.getsizeof(dataset_img) * sys.getsizeof(dataset_img[0]))/10e9)
    return (dataset_img, dataset_label)

def create_dataset3(clusters, make_fixed_size=True, width=70, height=1000, x_margin=5, y_margin=50):
    '''
    :param clusters: list of clusters
    :param make_fixed_size: if True, the image will have fixed size, otherwise it will be as big as the TPs
    :param width: width of the image
    :param height: height of the image
    :param x_margin: margin on the x axis
    :param y_margin: margin on the y axis
    :return: dataset [[img],[label]] in numpy array format
    '''
    dataset_img = []
    dataset_label = []

    dataset_img_np = np.array([from_tp_to_imgs(np.array(clusters[0]), make_fixed_size=make_fixed_size, width=width, height=height, x_margin=x_margin, y_margin=y_margin)])
    dataset_label_np = np.array([[-1]])
    cluster = np.array(clusters[0])
    if len(np.unique(cluster[:, 7])) > 1:
        dataset_label_np[0, 0] = 10
    else:
        dataset_label_np[0, 0] = cluster[0, 7]

    for i, cluster in enumerate(clusters[1:]):
        # create the image
        img = from_tp_to_imgs(np.array(cluster), make_fixed_size=make_fixed_size, width=width, height=height, x_margin=x_margin, y_margin=y_margin)
        # create the label
        if len(np.unique(np.array(cluster)[:, 7])) > 1:
            label = 10
        else:
            label = cluster[0][7]        
        # append to the dataset as an array of arrays
        dataset_img.append(img)
        dataset_label.append([label])
        if i%1000 == 0:
            dataset_img_np = np.concatenate((dataset_img_np, np.array(dataset_img)), axis=0)
            dataset_label_np = np.concatenate((dataset_label_np, np.array(dataset_label)), axis=0)
            dataset_img = []
            dataset_label = []

    dataset_img_np = np.concatenate((dataset_img_np, np.array(dataset_img)), axis=0)
    dataset_label_np = np.concatenate((dataset_label_np, np.array(dataset_label)), axis=0)
    dataset_img.clear()
    dataset_label.clear()

    print( dataset_img_np.size * dataset_img_np.itemsize/10e9)
    print(sys.getsizeof(dataset_img_np[0]))


    return (dataset_img_np, dataset_label_np)

def create_dataset4(clusters, make_fixed_size=True, width=70, height=1000, x_margin=5, y_margin=50):
    '''
    :param clusters: list of clusters
    :param make_fixed_size: if True, the image will have fixed size, otherwise it will be as big as the TPs
    :param width: width of the image
    :param height: height of the image
    :param x_margin: margin on the x axis
    :param y_margin: margin on the y axis
    :return: dataset [[img],[label]] in numpy array format
    '''
    # create the full array beforehands
    dataset_img = np.empty((len(clusters), height, width))
    dataset_label = np.empty((len(clusters), 1))

    # for i,  cluster in enumerate(clusters):
    #     # create the image
    #     img = from_tp_to_imgs(np.array(cluster), make_fixed_size=make_fixed_size, width=width, height=height, x_margin=x_margin, y_margin=y_margin)
    #     # create the label
    #     if len(np.unique(np.array(cluster)[:, 7])) > 1:
    #         label = 10
    #     else:
    #         label = cluster[0][7]        
    #     # append to the dataset as an array of arrays
    #     dataset_img[i] = img
    #     dataset_label[i] = [label]

    print( dataset_img.size * dataset_img.itemsize/10e9)

    return (dataset_img, dataset_label)


if __name__=='__main__':

    #read first n_events lines in the data from txt, all are ints
    if n_events:
        all_TPs = np.loadtxt(filename, skiprows=0, max_rows=n_events, dtype=int)
    else:
        all_TPs = np.loadtxt(filename, skiprows=0, dtype=int)
    
    all_TPs[:, 3] = all_TPs[:, 3]%2560

    #read channel map
    channel_map = create_channel_map_array(chanMap)

    #create images
    clusters = cluster_maker(all_TPs, channel_map, ticks_limit=150, channel_limit=20, min_hits=2)
    print('Number of clusters: ', len(clusters))
    if show or save:    
        for i, cluster in enumerate(clusters):
            show_or_save_img(np.array(cluster), channel_map, show=show, save=save, save_path=save_path, outname='test'+str(i), min_tps=2, make_fixed_size=True, width=70, height=1000, x_margin=5, y_margin=50)

    # write the clusters to a file
    if write:
        with open('clusters.txt', 'w') as f:
            for i, cluster in enumerate(clusters):
                f.write('Cluster'+str(i)+':\n')
                for tp in cluster:
                    f.write(str(tp[0]) + ' ' + str(tp[1]) + ' ' + str(tp[2]) + ' ' + str(tp[3]) + ' ' + str(tp[4]) + ' ' + str(tp[5]) + ' ' + str(tp[6]) + ' ' + str(tp[7]) + '\n')
                f.write('\n')
        




    # do some statistics on the clusters

    n_spurious_clusters = 0
    hist_types = np.array([0,0,0,0,0,0,0,0,0,0])
    for cluster in clusters:
        temp = cluster[0][7]
        for tp in cluster:
            if tp[7] != temp:
                n_spurious_clusters += 1
                break
        else:
            hist_types[cluster[0][7]] += 1
    print('Number of clusters with different type: ', n_spurious_clusters)

    print('Types: ', hist_types)

    # create the dataset
    # print('-------- Numpy version --------')
    # dataset_img, dataset_label = create_dataset(clusters, make_fixed_size=True, width=70, height=1000, x_margin=5, y_margin=50)
    # print('--------- Python list version --------')
    # dataset_img, dataset_label = create_dataset2(clusters, make_fixed_size=True, width=70, height=1000, x_margin=5, y_margin=50)
    # # sleep(10)]
    # time.sleep(5)

    # print('--------- Python list version with numpy array --------')
    # dataset_img, dataset_label = create_dataset3(clusters, make_fixed_size=True, width=70, height=1000, x_margin=5, y_margin=50)

    print('--------- Python list version with numpy array and preallocated array --------')
    dataset_img, dataset_label = create_dataset4(clusters, make_fixed_size=True, width=70, height=1000, x_margin=5, y_margin=50)


    print('Dataset shape: ', (dataset_img).shape)
    print('Labels shape: ', (dataset_label).shape)

    np.save('dataset_img.npy', dataset_img)
    np.save('dataset_lab.npy', dataset_label)

    print("Done!")





