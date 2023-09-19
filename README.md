# Trigger primitives to image
## Conversion
The conversion is done by the script `tp_to_img.py`.
It takes as input a file containing the trigger primitives in the format:
```
TP format:  
[0] time_start
[1] time_over_threshold
[2] time_peak
[3] channel
[4] adc_integral
[5] adc_peak
[6] detID
[7] type
```
and a file containing the channel mapping in the format:
```
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
```
The outputs are the images in png format for visualization and in numpy format for machine learning.

NOTE: The numpy format works fine but it requires a lot of memory (one number per pixel). Since the pixel values are zero in most of the cases, it would be better to use a sparse format, if possible, or a different type of network. Not tested yet.



WARNING: right now it work with either the inverted or the upright channel mapping, in the full channel map with repetition that are not taken into account. The two are inverted with respect to each other, so you lose the information on the orientation of the channel, but you can still do classification.

WARNING: the script works fine with one plane (as in the /eos/user/d/dapullia/tp_dataset/snana_hits.txt file), but it is not tested with the full U, V, X information.


## Machine learning

Simple machine learning algorithms are implemented in the "cnn2d_classifier.py".
The input are the numpy files containing the images and the labels.

The output is a model, a figure of the loss and accuracy as a function of the epoch and the ROC curves for the test set.

## Usage

```
python TP_to_img.py  -chanmap channel_map_upright.txt -filename=/eos/user/d/dapullia/tp_dataset/snana_hits.txt  -n_events=1000000 -save_ds

python cnn2d_classifier.py 

```


