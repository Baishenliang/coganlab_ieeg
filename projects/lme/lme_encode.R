os_type <- Sys.info()['sysname']

if (os_type == "Windows") {
  execution_mode <- "WINDOWS_LOCAL"
  library(tidyverse)
  library(lme4)
  library(lmerTest)
  library(parallel)
  library(foreach)
  library(doParallel)
  home_dir <- "D:/bsliang_Coganlabcode/coganlab_ieeg/projects/lme/"
  task_ID <- -1
  num_cores <- detectCores()
} else if (os_type == "Linux")  {
  library(tidyverse, lib.loc = "~/lab/bl314/rlib")
  library(lme4, lib.loc = "~/lab/bl314/rlib")
  library(lmerTest, lib.loc = "~/lab/bl314/rlib")
  library(parallel, lib.loc = "~/lab/bl314/rlib")
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

#%% Get core environment
cl <- makeCluster(num_cores - 1)
registerDoParallel(cl)

#%% Parameters
n_perm <- 10
set.seed(42)
features <- c('pho1', 'pho2', 'pho3', 'pho4', 'pho5')
align_to_onsets <- c('org', 'pho1')
post_align_T_threshold <- c(-0.2, 1)

#%% Load files
cat("loading files \n")
file_path <- paste(home_dir,
                   "data/epoc_LexDelayRep_Aud_full_Auditory_delay_long.csv",
                   sep = "")
long_data_org <- read.csv(file_path)

#%% Run computations
cat("run computations \n")
a = 1
for (feature in features) {
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
          align_to_onset == 'org' ~ time_point,
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
    perm_compare_df <- foreach(
      current_data = data_by_time,
      .combine = 'rbind',
      .packages = c('lme4', 'lmerTest', 'dplyr'),
      .errorhandling = 'pass'
    ) %dopar% {

      tp <- current_data$time[1]
      
      # Modelling
      lme_model <- lmer(
        value ~ fea + (1 | subject) + (1 | electrode) + (1 | stim),
        data = current_data,
        REML = FALSE
      )
      
      # Model comparison (to null)
      null_model <- lmer(
        value ~ 1 + (1 | subject) + (1 | electrode) + (1 | stim),
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
      for (i_perm in 1:n_perm) {

        current_data_perm <- data.frame(
          value_perm <- current_data$value,
          fea_perm <- sample(current_data$fea),
          subject <- current_data$subject,
          electrode <- current_data$electrode,
          stim <- current_data$stim
        )
        
        lme_model_perm <- lmer(
          value_perm ~ fea_perm + (1 | subject) + (1 | electrode) + (1 | stim),
          data = current_data_perm,
          REML = FALSE
        )
        null_model_perm <- lmer(
          value_perm ~ 1 + (1 | subject) + (1 | electrode) + (1 | stim),
          data = current_data_perm,
          REML = FALSE
        )
        model_comparison_perm <- anova(null_model_perm, lme_model_perm)
        p_value_comp_perm <- model_comparison_perm$`Pr(>Chisq)`[2]
        observed_chisq_perm  <- model_comparison_perm$Chisq[2]
        
      }
      perm_compare_df_i <- rbind(
        perm_compare_df_i,
        data.frame(
          perm = i_perm,
          time_point = tp,
          chi_squared_obs = observed_chisq_perm,
          p_value_perm = p_value_comp_perm
        )
      )
      return(perm_compare_df_i)
    }
    
    perm_compare_df <- perm_compare_df %>% arrange(time_point)
    
    print(perm_compare_df)
    
    write.csv(perm_compare_df,paste(home_dir,"results/","Auditory_delay_full_",feature,"_",align_to_onset,"aln.csv",sep = ''),row.names = FALSE)
  }
}