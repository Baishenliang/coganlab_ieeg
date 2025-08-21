os_type <- Sys.info()['sysname']

if (os_type == "Windows") {
  execution_mode <- "WINDOWS_LOCAL"
  library(tidyverse)
  library(parallel)
  # library(pbapply)
  library(foreach)
  library(doParallel)
  home_dir <- "D:/bsliang_Coganlabcode/coganlab_ieeg/projects/lme/"
  task_ID <- -1
  num_cores <- detectCores()-1
} else if (os_type == "Linux")  {
  library(tidyverse, lib.loc = "~/lab/bl314/rlib")
  library(parallel, lib.loc = "~/lab/bl314/rlib")
  # library(pbapply, lib.loc = "~/lab/bl314/rlib")
  library(foreach, lib.loc = "~/lab/bl314/rlib")
  library(doParallel, lib.loc = "~/lab/bl314/rlib")
  slurm_job_id <- Sys.getenv("SLURM_JOB_ID")
  home_dir <- "~/workspace/lme/"
  if (slurm_job_id != "") {
    execution_mode <- "HPC_SLURM_JOB"
    task_ID <- as.integer(Sys.getenv("SLURM_ARRAY_TASK_ID"))
    num_cores <- as.integer(Sys.getenv("SLURM_CPUS_PER_TASK", unset = 10))
  } else{
    execution_mode <- "LINUX_INTERACTIVE"
    task_ID <- -1
    num_cores <- 10
  }
}

#%% Modeling func
model_func <- function(current_data,feature){
  
  # Loading packages
  os_type <- Sys.info()['sysname']
  if (os_type == "Windows"){
    library(lme4)
    library(lmerTest)
    library(tidyverse)
  }else if (os_type == "Linux"){
    library(lme4, lib.loc = "~/lab/bl314/rlib")
    library(lmerTest, lib.loc = "~/lab/bl314/rlib")
    library(tidyverse, lib.loc = "~/lab/bl314/rlib")
  }
  
  model_type='full'
  
  tp <- current_data$time[1]
  if (model_type=='simple'){
    fml_bsl<-as.formula(paste0('value ~ bsl1+bsl2+bsl3+bsl4'))
    if (feature=='aco'){
      fml<-as.formula(paste0('value ~ bsl1+bsl2+bsl3+bsl4+',paste0(paste0("aco", 1:9), collapse = " + "),"+wordness"))
    }else if (feature=='pho'){
      fml<-as.formula(paste0('value ~ bsl1+bsl2+bsl3+bsl4+',paste0(paste0("pho", 1:23), collapse = " + "),"+wordness"))
    }else if (feature=='word'){
      fml<-as.formula('value ~ bsl1+bsl2+bsl3+bsl4 + wordness')
    }
  }else if (model_type=='full'){
    if (feature=='aco'){
      fml_bsl<-as.formula(paste0('value ~ bsl1+bsl2+bsl3+bsl4+',paste0(paste0("pho", 1:23), collapse = " + "),"+wordness"))
    }else if (feature=='pho'){
      fml_bsl<-as.formula(paste0('value ~ bsl1+bsl2+bsl3+bsl4+',paste0(paste0("aco", 1:9), collapse = " + "),"+wordness"))
    }else if (feature=='word'){
      fml_bsl<-as.formula(paste0('value ~ bsl1+bsl2+bsl3+bsl4+',paste0(paste0("aco", 1:9), collapse = " + "),"+",paste0(paste0("pho", 1:23), collapse = " + ")))
    }
    fml <-as.formula(paste0('value ~ bsl1+bsl2+bsl3+bsl4+',paste0(paste0("aco", 1:9), collapse = " + "),"+",paste0(paste0("pho", 1:23), collapse = " + "),"+wordness"))
  }
  
  m <- lm(fml, data = current_data,na.action = na.exclude)
  m_bsl <- lm(fml_bsl, data = current_data,na.action = na.exclude)
  anova_results <- anova(m_bsl, m)
  r_squared_obs <- anova_results$`F`[2]

  perm_compare_df_i <- data.frame(
    perm = 0,
    time_point = tp,
    chi_squared_obs = r_squared_obs
  )
  
  # Permutation
  cat('Start perm \n')
  n_perm <- 5000
  
  for (i_perm in 1:n_perm) {
    set.seed(10000 + i_perm)
    current_data_perm <- current_data %>%
      group_by(subject, electrode) %>%
      mutate(
        perm_indices = sample(1:n()),
        across(starts_with(feature), ~ .x[perm_indices])
      ) %>%
      ungroup() %>%
      select(-perm_indices)
    
    m_perm <- lm(fml, data = current_data_perm,na.action = na.exclude)
    m_perm_bsl <- lm(fml_bsl, data = current_data_perm,na.action = na.exclude)
    anova_results <- anova(m_perm_bsl, m_perm)
    r_squared_obs <- anova_results$`F`[2]
    
    perm_compare_df_i <- rbind(
      perm_compare_df_i,
      data.frame(
        perm = i_perm,
        time_point = tp,
        chi_squared_obs = r_squared_obs
      )
    )
    
  }

  return(perm_compare_df_i)
}

#%% Parameters
phase<-'full'
elec_grps <- c('Auditory_delay','Sensorymotor_delay','Motor_delay','Delay_only')
features <- c('aco','pho','word')
a = 0

#Load acoustic parameters
aco_path <- paste(home_dir,
                   "data/envelope_feature_dict_pca.csv",
                   sep = "")
aco_fea <- read.csv(aco_path,row.names = 1)
aco_fea_T <- as.data.frame(t(aco_fea))
aco_fea_T$stim <- rownames(aco_fea_T)
aco_fea_T <- aco_fea_T[, c("stim", setdiff(names(aco_fea_T), "stim"))]

#Load phonemic parameters
pho_path <- paste(home_dir,
                  "data/phoneme_one_hot_dict_pca.csv",
                  sep = "")
pho_fea <- read.csv(pho_path,row.names = 1)
pho_fea_T <- as.data.frame(t(pho_fea))
pho_fea_T$stim <- rownames(pho_fea_T)
pho_fea_T <- pho_fea_T[, c("stim", setdiff(names(pho_fea_T), "stim"))]


#%% Start looping
for (elec_grp in elec_grps){
  for (feature in features){
    #%% Run computations
    a <- a + 1
    if (task_ID > 0 && a != task_ID) {
      next
    }
    
    #%% Load files
    cat("loading files \n")
    # slurm task selection
    file_path <- paste(home_dir,
                       "data/epoc_LexDelayRep_Aud_",phase,"_",elec_grp,"_long.csv",
                       sep = "")
    long_data <- read.csv(file_path)
    
    #%% append acoustic features
    long_data <- left_join(long_data,aco_fea_T,by='stim')
    
    #%% append phonemic features
    long_data <- left_join(long_data,pho_fea_T,by='stim')
    
    long_data$time <- as.numeric(long_data$time)
    time_points <- unique(long_data$time)
    
    #%% get baseline
    long_data <- long_data %>%
      group_by(subject, electrode, stim) %>%
      mutate(bsl1 = mean(value[time > -0.1 & time <= 0],na.rm = TRUE),
             bsl2 = mean(value[time > -0.2 & time <= -0.1],na.rm = TRUE),
             bsl3 = mean(value[time > -0.3 & time <= -0.2],na.rm = TRUE),
             bsl4 = mean(value[time > -0.4 & time <= -0.3],na.rm = TRUE))%>%
      mutate(across(starts_with("bsl"), ~if_else(is.nan(.), 0, .))) %>%
      ungroup()
    
    cat("Re-formatting long data \n")
    data_by_time <- split(long_data, long_data$time)
    rm(long_data)
    
    cat("Starting modeling \n")
    #%% Get core environment
    cl <- makeCluster(num_cores)
    registerDoParallel(cl)
    clusterExport(cl, varlist = c("model_func"))
    # Fot Duke HPC sbatch:
    # No. CPU set as 30, memory limits set as 30GB, it takes 4~5 hours to complete one set of model fitting followed by 100 permutations with 1.2 seconds of trial length.
    # 13 tasks can be paralled at once.
    perm_compare_df<-parLapply(cl, data_by_time, model_func,feature=feature)
    stopCluster(cl)
    perm_compare_df <- do.call(rbind, perm_compare_df)
    perm_compare_df <- perm_compare_df %>% arrange(time_point)
    
    print(perm_compare_df)
    
    write.csv(perm_compare_df,paste(home_dir,"results/",elec_grp,"_",phase,"_",feature,".csv",sep = ''),row.names = FALSE)
  }
}