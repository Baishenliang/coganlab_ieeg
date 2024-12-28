# IEEG processing scripts by Baishen Liang

Please clone the Cogan Lab ieeg repository (https://github.com/coganlab/IEEG_Pipelines) first, install ieeg, and use the ieeg environment as the interpreter.  

![IEEG analyses pipeline](materials/analyze_pipeline.png)   

## Step0：response coding and BIDS convert  
See: [Gitlab response coding instructions](https://coganlab.pages.oit.duke.edu/wiki/docs/ECoG_In_Unit/Response_Coding/)  
(Duke NetID required)  
  
## Step1: files transfer and upload
**1.** Copy the BIDS coded EEG files:  
![file transfer 1](materials/files_transfer_1.png)  
to Coganlab's box:  
![file transfer 2](materials/files_transfer_2.png)   
  
**2.** Use **Globus** to synchronize the files:  
(Duke NetID required)  
![Upload DCC 1](materials/upload_DCC_1.png)  
and 
![Upload DCC 2](materials/upload_DCC_2.png)  
  
## Step2: update preprocessing batch codes for patients.  
**1.** Check whether the patient had eeg channels by inspecting these two files:  
![Check eeg 1](materials/check_eeg_chs_1.png)  
Then write the report to this location:  
![Check eeg 2](materials/check_eeg_chs_2.png)  
If there are eeg channels, add the `specific eeg channels` to the csv file.  
If there are no eeg channels, simply add a `nan`.  

**2.** Updated the `batch_preproc.py`.  
![Updated preproce batch](materials\update_preproc_batch.png) 
