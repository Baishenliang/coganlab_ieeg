# IEEG scripts by Baishen Liang

Please clone the Cogan Lab ieeg repository (https://github.com/coganlab/IEEG_Pipelines) first, install ieeg, and use the ieeg environment as the interpreter.  

## Processing steps
1. Load and plot the packages  
2. Perform line noise filtering  
3. Delect and remove bad channels  
4. Remove muscle artifact bad channels  
5. Delect and remove bad trials  
6. Make the spectrograms for Auditory, Delay, and Response  

## 4. Remove muscle artifact bad channels
- Do the wavelet and get log spectrum (https://ieeg-pipelines.readthedocs.io/en/latest/auto_examples/plot_spectrograms_wavelet.html#calculate-spectra)  
- Should be on the end of a shank (usually higher end, but lower is possible if sticks out other side); activity that is suddenly higher than the rest of the shank; if no signal outside of brain saves electrode（在一串seeg电极的末端或者开始端，没有与脑组织接触的部分，如果出现非常强的反应，则判定为muscle artifacts）  
- Anterior temporal production signal should also have auditory response (在颞叶前端如果记录到强的production signal，那应该在encoding的时候也有比较强的auditory response，不然则判定为muscle artifacts)
- If see activity from 60 to 600 Hz mo matter if it is from anterior temporal or other places, should be muscle artifacts (remove it if no auditory responses for the stimuli)
- Remove the same electrodes for all events (i.g., Auditory, Delay, and Response)  

