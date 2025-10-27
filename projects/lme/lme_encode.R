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
  
  tp <- current_data$time[1]
  
  fml <- as.formula(paste0('value ~ 1+',paste0(c(paste0('aco', 1:16), paste0("pho", 1:32),"wordness"), collapse = ' + ')))

  current_data_vWM <- current_data[current_data$vWM == 1, ]
  current_data_novWM <- current_data[current_data$vWM == 0, ]
  
  m_vWM <- lm(fml, data = current_data_vWM,na.action = na.exclude)
  m_novWM <- lm(fml, data = current_data_novWM,na.action = na.exclude)
  
  coef_vWM <- abs(coef(m_vWM))
  se_vWM <- summary(m_vWM)$coefficients[, "Std. Error"]
  
  coef_novWM <- abs(coef(m_novWM))
  se_novWM <- summary(m_novWM)$coefficients[, "Std. Error"]
  
  non_intercept_names <- names(coef_vWM)[names(coef_vWM) != '(Intercept)']
  diff_coef <- coef_vWM[non_intercept_names] - coef_novWM[non_intercept_names]
  var_diff <- se_vWM[non_intercept_names]^2 + se_novWM[non_intercept_names]^2
  Z_statistic <- diff_coef / sqrt(var_diff)
  Z_statistic_df <- as.data.frame(as.list(Z_statistic))
  
  perm_compare_df_i <- data.frame(
    perm = 0,
    time_point = tp
  )
  
  perm_compare_df_i <- bind_cols(perm_compare_df_i, Z_statistic_df)
  
  # Permutation
  cat('Start perm \n')
  n_perm <- 5e2
  
  for (i_perm in 1:n_perm) {
    set.seed(10000 + i_perm)
    
    original_map <- current_data %>%
      distinct(subject, electrode, vWM)
    
    vwm_list_to_shuffle <- original_map$vWM
    shuffled_vwm <- sample(vwm_list_to_shuffle)
    
    permuted_vwm_map <- original_map %>%
      mutate(vWM_permuted = shuffled_vwm) %>%
      select(subject, electrode, vWM_permuted)
    
    current_data_perm <- current_data %>%
      left_join(permuted_vwm_map, by = c("subject", "electrode"), relationship = "many-to-one") %>%
      mutate(vWM = vWM_permuted) %>%
      select(-vWM_permuted)
    
    current_data_vWM_perm <- current_data_perm[current_data_perm$vWM == 1, ]
    current_data_novWM_perm <- current_data_perm[current_data_perm$vWM == 0, ]
    
    m_vWM_perm <- lm(fml, data = current_data_vWM_perm,na.action = na.exclude)
    m_novWM_perm <- lm(fml, data = current_data_novWM_perm,na.action = na.exclude)
    
    coef_vWM_perm <- abs(coef(m_vWM_perm))
    se_vWM_perm <- summary(m_vWM_perm)$coefficients[, "Std. Error"]
    
    coef_novWM_perm <- abs(coef(m_novWM_perm))
    se_novWM_perm <- summary(m_novWM_perm)$coefficients[, "Std. Error"]
    
    diff_coef_perm <- coef_vWM_perm[non_intercept_names] - coef_novWM_perm[non_intercept_names]
    var_diff_perm <- se_vWM_perm[non_intercept_names]^2 + se_novWM_perm[non_intercept_names]^2
    Z_statistic_perm <- diff_coef_perm / sqrt(var_diff_perm)
    Z_statistic_df_perm <- as.data.frame(as.list(Z_statistic_perm))
    
    perm_compare_df_i_perm <- data.frame(
      perm = i_perm,
      time_point = tp
    )
    
    perm_compare_df_i <- rbind(
      perm_compare_df_i,
      bind_cols(perm_compare_df_i_perm, Z_statistic_df_perm)
    )
    
  }

  return(perm_compare_df_i)
}

#%% Parameters
elec_grps <- c('Auditory','Sensorymotor','Motor')
features <- c("pho")
a = 0

#Load acoustic parameters
aco_path <- paste(home_dir,
                   "data/envelope_feature_dict.csv",
                   sep = "")
aco_fea <- read.csv(aco_path,row.names = 1)
aco_fea_T <- as.data.frame(t(aco_fea))
aco_fea_T$stim <- rownames(aco_fea_T)
aco_fea_T <- aco_fea_T[, c("stim", setdiff(names(aco_fea_T), "stim"))]

#Load vowel parameters
pho_path <- paste(home_dir,
                  "data/phoneme_one_hot_dict.csv",
                  sep = "")
pho_fea <- read.csv(pho_path,row.names = 1)
pho_fea_T <- as.data.frame(t(pho_fea))
pho_fea_T$stim <- rownames(pho_fea_T)
pho_fea_T <- pho_fea_T[, c("stim", setdiff(names(pho_fea_T), "stim"))]


#%% Start looping
for (feature in features){
  for (elec_grp in elec_grps){
    
    #%% Load files
    cat("loading files \n")
    # slurm task selection
    # vwm electrodes
    file_path_long_vwm <- paste(home_dir,
                       "data/epoc_LexDelayRep_Aud_",elec_grp,"_vWM_long.csv",
                       sep = "")
    long_data_vwm <- read.csv(file_path_long_vwm)
    long_data_vwm$time <- as.numeric(long_data_vwm$time)
    
    # no vWM electrodes
    file_path_long_novwm <- paste(home_dir,
                                "data/epoc_LexDelayRep_Aud_",elec_grp,"_novWM_long.csv",
                                sep = "")
    long_data_novwm <- read.csv(file_path_long_novwm)
    long_data_novwm$time <- as.numeric(long_data_novwm$time)
    
    # Merge
    data_vwm_labeled <- long_data_vwm %>%
      mutate(vWM = 1)
    data_novwm_labeled <- long_data_novwm %>%
      mutate(vWM = 0)
    long_data <- bind_rows(data_vwm_labeled, data_novwm_labeled)
    
    
    #%% get only word part of the "stim"
    long_data <- long_data %>%
      mutate(
        stim = str_split_fixed(string = stim, pattern = "-", n = 2)[, 1]
      )

    #%% append acoustic features
    long_data <- left_join(long_data,aco_fea_T,by='stim')
    
    #%% append vowel features
    long_data <- left_join(long_data,pho_fea_T,by='stim')
    
    long_data$time <- as.numeric(long_data$time)
    time_points <- unique(long_data$time)
    
    #for (lex in c("Word","Nonword",'All')){
    lex<-'All'
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
    #}
  }
}