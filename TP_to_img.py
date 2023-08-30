import numpy as np
import matplotlib.pyplot as plt
import time
import os
import argparse

# write a parser to set up the running from command line
parser = argparse.ArgumentParser(description='Tranforms Trigger Primitives to images.')
parser.add_argument('-filename', type=str, default='fd-snb-mc-hits/snana_hits.txt', help='path to the file with TPs')
parser.add_argument('-show', action='store_true', help='show the image')
parser.add_argument('-save', action='store_true', help='save the image')
parser.add_argument('-save_path', type=str, default='TP_to_img/images/', help='path to save the image')
parser.add_argument('-n_events', type=int, default=1000, help='number of events to process')

args = parser.parse_args()
# unpack the arguments
filename = args.filename
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

#read first n_events lines in the data from txt
all_TPs = np.loadtxt(filename, skiprows=0, max_rows=n_events)



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
# #plot image

# img = from_tp_to_img(all_TPs, all_TPs[0, 2], all_TPs[500, 2])
# print(img.shape)

# plt.imshow(np.rot90(img), cmap='hot', interpolation='nearest')
# plt.show()

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

if __name__=='__main__':
    print(all_TPs.shape)

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


    
    candidates=find_candidates(all_TPs, 100, 1500, 1)

    print(len(candidates))

    for candidate in candidates[:9]:

        candidate = np.array(candidate)
        img = from_tp_to_img(candidate,1)

        
        fig= plt.figure(figsize=(20, 20))
        plt.imshow(np.rot90(img))
        plt.colorbar()
        plt.show()

