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

  # partial glm
  aco_formula <- as.formula(paste0("value ~ ", paste0(paste0("aco", 1:16), collapse = " + "), "+", paste0(paste0("pho", 1:5), collapse = " + ")))
  model_aco <- lm(aco_formula, data = current_data)
  current_data$value_res <- residuals(model_aco)
  pho_formula <- as.formula("value_res ~  wordness")
  model_pho <- lm(pho_formula, data = current_data)
  model_pho_summary <- summary(model_pho)
  r_squared_obs <- model_pho_summary$r.squared

  perm_compare_df_i <- data.frame(
    perm = 0,
    time_point = tp,
    chi_squared_obs = r_squared_obs
  )
  
  # Permutation
  cat('Start perm \n')
  n_perm <- 5000
  for (i_perm in 1:n_perm) {
    
    perm_indices <- sample(1:nrow(current_data), nrow(current_data))
    
    current_data_perm <- current_data %>%
      select(everything(), -value_res) %>%
      mutate(across(starts_with("aco") | starts_with("pho") | wordness, ~ .x[perm_indices]))
    
    aco_formula <- as.formula(paste0("value ~ ", paste0(paste0("aco", 1:16), collapse = " + "), "+", paste0(paste0("pho", 1:5), collapse = " + ")))
    model_aco <- lm(aco_formula, data = current_data_perm)
    current_data_perm$value_res <- residuals(model_aco)
    pho_formula <- as.formula("value_res ~ wordness")
    model_pho <- lm(pho_formula, data = current_data_perm)
    model_pho_summary <- summary(model_pho)
    r_squared_obs <- model_pho_summary$r.squared
    
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
set.seed(42)
phase<-'full'
elec_grps <- c('Auditory_delay','Sensorymotor_delay','Delay_only','Motor_delay')
align_to_onsets <- c('pho0')
feature <- c('lexstus')
post_align_T_threshold <- c(-0.2, 1.5)
a = 0

#Load acoustic parameters
acoust_path <- paste(home_dir,
                   "data/envelope_feature_dict.csv",
                   sep = "")
acoust_fea <- read.csv(acoust_path)
acoust_fea_T <- acoust_fea %>%
  mutate(row_id = 1:n()) %>%
  pivot_longer(
    cols = -row_id,
    names_to = "stim",
    values_to = "aco_value"
  ) %>%
  pivot_wider(
    names_from = row_id,
    values_from = aco_value,
    names_prefix = "aco"
  )

#%% Start looping
for (elec_grp in elec_grps){
  #%% Load files
  cat("loading files \n")
  file_path <- paste(home_dir,
                     "data/epoc_LexDelayRep_Aud_",phase,"_",elec_grp,"_long.csv",
                     sep = "")
  long_data_org <- read.csv(file_path)
  
  #%% append acoustic features
  long_data_org <- left_join(long_data_org,acoust_fea_T,by='stim')
  
  #%% Run computations
  for (align_to_onset in align_to_onsets) {
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