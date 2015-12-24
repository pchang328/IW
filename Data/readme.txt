DATA SET:
---------

Files:
------
data.txt: contains our features and binary results
features.txt: contains only our features without our binary results.

Description of features:
------------------------
Col 1: Name of file

We take the average intensity of each row in the reed's picture and for each 
reed make a plot of how these avg intensities change from top to bottom. We 
noticed that these avg intensities followed an exponentional distribution of the 
form ae^(-bx) + c. Columns 2 - 4 show our least-squares estimate of these
variables.

Col 2: a
Col 3: b
Col 4: c 

Col 5: Reed mass in grams
Col 6: Reed mass in grams after soaking in water for 10 seconds. 
Col 7: Percentage increase in mass (amount of water absorbed)

Reeds change throughout time. A reed that plays one way today may sound 
different when played on tomorrow. To account for these changes, we bin 
each reed into the following 5 categories:

0: We never played on the reed at the time of the recording.
1: We played on the reed once at the time of the recording.
2: We played on the reed twice at the time of the recording.
3: We played on the reed 3-5 times at the time of the recording (S)
4: We played on the reed >5 times at the time of the recording (U)

The reeds are binned in this way because from my own experiences, reeds tend
to change most drastically the less they have been "broken in" (i.e. played
on less). Hence, a reed's s.c. will vary more between the first 2 times we
play on the reed than the 10th and 11th times. Reeds in the 4th category will 
have similiar s.c whereas if we binned categories 0-2 together, these s.c. will
most likely be quite different.

Col 8: Time category

Description of results:
-----------------------
We generate the s.c. for each of the of reed's recordings. We first turn to
the classifcation problem where we categorize a reed as bright (1) or dark(0).
The threshold that we use is the average s.c. accross all recordings. Hence 
any recordng with an s.c greater than this average will be considered bright
and any recording with an s.c. less than thsi average will be dark. 

Col 9: 1 if Low G sound is bright, 0 if sound is dark. 
Col 10: 1 if High G sound is bright, 0 if sound is dark. 
