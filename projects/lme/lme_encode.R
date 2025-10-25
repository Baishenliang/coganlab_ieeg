os_type <- Sys.info()['sysname']

if (os_type == "Windows") {
  execution_mode <- "WINDOWS_LOCAL"
  library(tidyverse)
  library(stringr)
  library(parallel)
  # library(pbapply)
  library(foreach)
  library(doParallel)
  home_dir <- "D:/bsliang_Coganlabcode/coganlab_ieeg/projects/lme/"
  task_ID <- -1
  num_cores <- detectCores()-10
} else if (os_type == "Linux")  {
  library(tidyverse, lib.loc = "~/lab/bl314/rlib")
  library(stringr, lib.loc = "~/lab/bl314/rlib")
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
  
  fml <- as.formula(paste0('value ~ 1+',paste0(c(paste0('aco', 1:9), paste0("pho", 1:4)), collapse = ' + ')))

  m <- lm(fml, data = current_data,na.action = na.exclude)
  pattern <- paste0("^", feature)
  mean_F_stat <- mean(abs(coef(m)[grep(pattern, names(coef(m)))]))
  
  perm_compare_df_i <- data.frame(
    perm = 0,
    time_point = tp,
    chi_squared_obs = mean_F_stat
  )
  
  # Permutation
  cat('Start perm \n')
  n_perm <- 10#e3
  
  for (i_perm in 1:n_perm) {
    set.seed(10000 + i_perm)
    current_data_perm <- current_data %>%
      group_by(subject, electrode) %>%
      mutate(
        perm_indices = sample(1:n()),
        across(starts_with('aco') | starts_with('vow') | starts_with('con'), ~ .x[perm_indices]) #across(starts_with(feature), ~ .x[perm_indices])
      ) %>%
      ungroup() %>%
      select(-perm_indices)
    
      m <- lm(fml, data = current_data_perm,na.action = na.exclude)
      m_bsl <- lm(fml_bsl, data = current_data_perm,na.action = na.exclude)
      anova_results <- anova(m_bsl, m)
      mean_F_stat <- anova_results$`F`[2]
      # pattern <- paste0("^", feature)
      # mean_F_stat <- mean(abs(coef(m)[grep(pattern, names(coef(m)))]))
    
    perm_compare_df_i <- rbind(
      perm_compare_df_i,
      data.frame(
        perm = i_perm,
        time_point = tp,
        chi_squared_obs = mean_F_stat
      )
    )
    
  }

  return(perm_compare_df_i)
}

#%% Parameters
elec_grps <- c('Auditory_delay','Sensorymotor_delay','Motor_delay','Delay_only')
features <- c("vow","con")
a = 0

#Load acoustic parameters
aco_path <- paste(home_dir,
                   "data/envelope_feature_dict_pca.csv",
                   sep = "")
aco_fea <- read.csv(aco_path,row.names = 1)
aco_fea_T <- as.data.frame(t(aco_fea))
aco_fea_T$stim <- rownames(aco_fea_T)
aco_fea_T <- aco_fea_T[, c("stim", setdiff(names(aco_fea_T), "stim"))]

#Load vowel parameters
vow_path <- paste(home_dir,
                  "data/vowel_one_hot_dict_pca.csv",
                  sep = "")
vow_fea <- read.csv(vow_path,row.names = 1)
vow_fea_T <- as.data.frame(t(vow_fea))
vow_fea_T$stim <- rownames(vow_fea_T)
vow_fea_T <- vow_fea_T[, c("stim", setdiff(names(vow_fea_T), "stim"))]

#Load consonant parameters
con_path <- paste(home_dir,
                  "data/consonant_one_hot_dict_pca.csv",
                  sep = "")
con_fea <- read.csv(con_path,row.names = 1)
con_fea_T <- as.data.frame(t(con_fea))
con_fea_T$stim <- rownames(con_fea_T)
con_fea_T <- con_fea_T[, c("stim", setdiff(names(con_fea_T), "stim"))]


#%% Start looping
for (feature in features){
  for (elec_grp in elec_grps){
    
    #%% Load files
    cat("loading files \n")
    # slurm task selection
    file_path_long <- paste(home_dir,
                       "data/epoc_LexDelayRep_Aud_",elec_grp,"_long.csv",
                       sep = "")
    long_data <- read.csv(file_path_long)
    long_data$time <- as.numeric(long_data$time)
    
    #%% get only word part of the "stim"
    long_data <- long_data %>%
      mutate(
        stim = str_split_fixed(string = stim, pattern = "-", n = 2)[, 1]
      )

    #%% append acoustic features
    long_data <- left_join(long_data,aco_fea_T,by='stim')
    
    #%% append vowel features
    long_data <- left_join(long_data,vow_fea_T,by='stim')

    #%% append consonant features
    long_data <- left_join(long_data,con_fea_T,by='stim')
    
    long_data$time <- as.numeric(long_data$time)
    time_points <- unique(long_data$time)
    
    for (lex in c("Word","Nonword",'All')){
      #%% Run computations
      a <- a + 1
      if (task_ID > 0 && a != task_ID) {
        next
      }
      # If it is nonword then skip the Frq
      if (lex=='Nonword' && feature=='Frq'){
        next
      }
      # If it is not for all word and nonword data then skip the wordness
      # Although now we don't do 'Alls's
      if (lex!='All' && feature=='wordness'){
        next
      }
      
      if (lex=='Word' || lex=='Nonword'){
        word_data <- long_data %>%
          filter(wordness == lex)
      }else{
        word_data <- long_data
      }
      
      if (task_ID > 0){rm(long_data)}
      
      cat("Re-formatting long data \n")
      data_by_time <- split(word_data, word_data$time)
      rm(word_data)
      
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
      
      write.csv(perm_compare_df,paste(home_dir,"results/",elec_grp,"_",feature,"_",lex,".csv",sep = ''),row.names = FALSE)
    }
  }
}