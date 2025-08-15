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
model_func <- function(current_data){
  
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
  
  tp <- current_data$time[1]
  
  # Modelling
  lme_model <- lmer(
    value ~ fea + (1 | electrode),
    data = current_data,
    REML = FALSE
  )
  
  # Model comparison (to null)
  null_model <- lmer(
    value ~ 1 + (1 | electrode),
    data = current_data,
    REML = FALSE
  )
  model_comparison <- anova(null_model, lme_model)
  p_value_comp <- model_comparison$`Pr(>Chisq)`[2]
  observed_chisq  <- model_comparison$Chisq[2]
  
  # Write down original model X2
  perm_compare_df_i <- data.frame(
    perm = 0,
    time_point = tp,
    chi_squared_obs = observed_chisq,
    p_value_perm = p_value_comp
  )
  
  # Permutation
  cat('Start perm \n')
  n_perm <- 200
  for (i_perm in 1:n_perm) {
    
    current_data_perm <- current_data %>%
      mutate(fea_perm = sample(fea))
    
    lme_model_perm <- lmer(
      value ~ fea_perm  + (1 | electrode),
      data = current_data_perm,
      REML = FALSE
    )
    null_model_perm <- lmer(
      value ~ 1  + (1 | electrode),
      data = current_data_perm,
      REML = FALSE
    )
    model_comparison_perm <- anova(null_model_perm, lme_model_perm)
    p_value_comp_perm <- model_comparison_perm$`Pr(>Chisq)`[2]
    observed_chisq_perm  <- model_comparison_perm$Chisq[2]
    
    perm_compare_df_i <- rbind(
      perm_compare_df_i,
      data.frame(
        perm = i_perm,
        time_point = tp,
        chi_squared_obs = observed_chisq_perm,
        p_value_perm = p_value_comp_perm
      )
    )
    
  }

  return(perm_compare_df_i)
}

#%% Parameters
set.seed(42)
phase<-'full'
elec_grps <- c('Auditory_all')
align_to_onsets <- c('pho0')
features <- c('pho1', 'pho2', 'pho3', 'pho4', 'pho5')
post_align_T_threshold <- c(-0.2, 1)
a = 0
for (elec_grp in elec_grps){
  #%% Load files
  cat("loading files \n")
  file_path <- paste(home_dir,
                     "data/epoc_LexDelayRep_Aud_",phase,"_",elec_grp,"_long.csv",
                     sep = "")
  long_data_org <- read.csv(file_path)
  
  #%% Run computations
  for (align_to_onset in align_to_onsets) {
    for (feature in features) {
      # slurm task selection
      a <- a + 1
      if (task_ID > 0 && a != task_ID) {
        next
      }
      
      #%% re-align to onsets and get timepoint
      cat("Re-aligning time points \n")
      long_data <- long_data_org %>%
        mutate(
          time_point = case_when(
            align_to_onset == 'pho0' ~ time_point,
            align_to_onset == 'pho1' ~ round(time_point - pho_t1, 2),
            align_to_onset == 'pho2' ~ round(time_point - pho_t2, 2),
            align_to_onset == 'pho3' ~ round(time_point - pho_t3, 2),
            align_to_onset == 'pho4' ~ round(time_point - pho_t4, 2),
            align_to_onset == 'pho5' ~ round(time_point - pho_t5, 2),
          )
        ) %>%
        filter((time_point >= post_align_T_threshold[1]) &
                 (time_point <= post_align_T_threshold[2])
        )
      long_data$time <- as.numeric(long_data$time)
      time_points <- unique(long_data$time)
      
      # Add fea
      cat("Adding feature column \n")
      long_data <- long_data %>%
        mutate(
          fea = case_when(
            feature == 'pho1'     ~ pho1,
            feature == 'pho2'     ~ pho2,
            feature == 'pho3'     ~ pho3,
            feature == 'pho4'     ~ pho4,
            feature == 'pho5'     ~ pho5,
            feature == 'syl1'     ~ paste0(pho1,pho2),
            feature == 'syl2'     ~ paste0(pho3,pho4,pho5),
            feature == 'wordness' ~ wordness
          )
        )
      
      if (os_type == "Linux"){
        rm(long_data_org)
      }
      
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
      perm_compare_df<-parLapply(cl, data_by_time, model_func)
      stopCluster(cl)
      perm_compare_df <- do.call(rbind, perm_compare_df)
      perm_compare_df <- perm_compare_df %>% arrange(time_point)
      
      print(perm_compare_df)
      
      write.csv(perm_compare_df,paste(home_dir,"results/",elec_grp,"_",phase,"_",feature,"_",align_to_onset,"aln.csv",sep = ''),row.names = FALSE)
    }
  }
}