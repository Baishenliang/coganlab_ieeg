# IEEG processing scripts by Baishen Liang

Please clone the Cogan Lab ieeg repository (https://github.com/coganlab/IEEG_Pipelines) first, install ieeg, and use the ieeg environment as the interpreter.  

![IEEG analyses pipeline](materials/analyze_pipeline.png)   

## Step 0：response coding and BIDS convert  
See: [Gitlab response coding instructions](https://coganlab.pages.oit.duke.edu/wiki/docs/ECoG_In_Unit/Response_Coding/)  
(Duke NetID required)  
  
## Step 1: files transfer and upload
**1.** Copy the BIDS coded EEG files:  
![file transfer 1](materials/files_transfer_1.png)  
to Coganlab's box:  
![file transfer 2](materials/files_transfer_2.png)  

**2.** Updated participants.tsv in:  
`~\Box\CoganLab\BIDS-1.0_LexicalDecRepDelay\BIDS`  
````text
# Add the new participants lines:  
sub-D0084	n/a	n/a	n/a	n/a	n/a
sub-D0086	n/a	n/a	n/a	n/a	n/a
````
  
**2.** Use **Globus** to synchronize the files to the Duke Computing Cluster (DCC):  
(Duke NetID required)  
![Upload DCC 1](materials/upload_DCC_1.png)  
and also the **participants.tsv**!!  
and  
![Upload DCC 2](materials/upload_DCC_2.png)  
  
## Step 2: update preprocessing batch codes for patients.  
**1.** Check whether the patient had eeg channels by inspecting these two files:  
![Check eeg 1](materials/check_eeg_chs_1.png)  
Then write the report to this location:  
![Check eeg 2](materials/check_eeg_chs_2.png)  
If there are eeg channels, add the `specific eeg channels` to the csv file.  
If there are no eeg channels, simply add a `nan`.  

**2.** Updated the `batch_preproc.py`.  
![Updated preproce batch](materials/update_preproc_batch.png) 

**3.** Commit and push. 
````bash
# cd to the local repository
git status
git add .
git commit -m "Patient D84 D86 added"
git push origin main
````

**4.** connect to the DCC.  
````bash
# use a new windows powershell window
ssh bl314@dcc-login.oit.duke.edu
# Input password and 2FA code
````

**5.** pull the repository.
````bash
# use the same powershell window as the DCC login
cd ~/bsliang_ieeg
# this is the soft link to:
# /hpc/group/coganlab/bl314/codes/bsliang_ieeg, where the github repository is cloned
git status
git pull
````
## Step 3: Run the batch script.
````bash
# still the powershell window as the DCC login
# see whether the patients' BIDS data have been uploaded
cd ~/workspace/BIDS-1.0_LexicalDecRepDelay/BIDS
ls
````
![Check eeg 2](materials/check_dcc_workspace.png)  
** Run the batch script for line noise filtering, outlier channels removal, and wavelet.  
````bash
cd ~/bsliang_ieeg/
sbatch sbatch_preproc.sh
````
You can `squeue -u bl314` to get the current status.  
![Check eeg 2](materials/DCC_squeue.png)  
 
Or you can check the script logs for python outputs or errors.  
````bash
# Check the sbatch output
cd ~/bsliang_ieeg/data/DCCbatchout/
cat slurm_84.err #errors
cat test84.out

# Check the python script output
cd cd ~/bsliang_ieeg/data/logs/batch_preproc_YYYY_MM_DD #change to the processing day
cat D0084.txt
````

