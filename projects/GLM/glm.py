import os
import glm_utils as glm

# Read data
event='Auditory'
stat='power' # or 'zscore'
task_Tag='Yes_No'

subjs, data_list, filtered_events_list, chs, times = glm.fifread(event,stat,task_Tag)

