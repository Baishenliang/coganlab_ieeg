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
  
  model_type='simple'
  
  tp <- current_data$time[1]
  if (model_type=='simple'){
    fml_bsl<-as.formula(paste0('value ~ 1'))
    if (feature=='aco'){
      fml<-as.formula(paste0('value ~ 1+',paste0(paste0("aco", 1:9), collapse = " + ")))
    }else if (feature=='pho'){
      fml<-as.formula(paste0('value ~ 1+',paste0(paste0("pho", 1:23), collapse = " + ")))
    }else{
      fml<-as.formula(paste0('value ~ 1+',feature))
    }
  }else if (model_type=='full'){
    if (feature=='aco'){
      fml_bsl <- as.formula(paste0('value ~ 1+',paste0(c(paste0('pho', 1:23)), collapse = ' + ')))
      fml <- as.formula(paste0('value ~ 1+',paste0(c(paste0('aco', 1:9), paste0('pho', 1:23)), collapse = ' + ')))
    }else if (feature=='pho'){
      fml_bsl <- as.formula(paste0('value ~ 1+',paste0(c(paste0('aco', 1:9)), collapse = ' + ')))
      fml <- as.formula(paste0('value ~ 1+',paste0(c(paste0('aco', 1:9), paste0('pho', 1:23)), collapse = ' + ')))
    }else if (feature=='wordness'){
      fml_bsl <- as.formula(paste0('value ~ 1+',paste0(c(paste0('aco', 1:9), paste0('pho', 1:23)), collapse = ' + ')))
      fml <-as.formula(paste0('value ~ 1+',
                              paste0(paste0("aco", 1:9), collapse = " + "),"+",paste0(paste0("pho", 1:23), collapse = " + "),"+",
                                paste0(paste0("aco", 1:9,":",feature), collapse = " + "),"+",paste0(paste0("pho", 1:23,":",feature), collapse = " + "),"+",feature))
    }else if (feature=='Wordvec'){
      fml_bsl <- as.formula(paste0('value ~ 1+',paste0(c(paste0('aco', 1:9), paste0('pho', 1:23)), collapse = ' + ')))
      fml <- as.formula(paste0('value ~ 1+',paste0(c(paste0('aco', 1:9), paste0('pho', 1:23), paste0('Wordvec', 1:91)),collapse = ' + ')))
    }else{
      fml_bsl <-as.formula(paste0('value ~ 1+',paste0(paste0("aco", 1:9), collapse = " + "),"+",paste0(paste0("pho", 1:23), collapse = " + ")))
      fml <-as.formula(paste0('value ~ 1+',paste0(paste0("aco", 1:9), collapse = " + "),"+",paste0(paste0("pho", 1:23), collapse = " + "),"+",feature))
    }
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
  n_perm <- 1e4
  
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
ana_tag<-'NoDel'
phase<-'full'
del_nodel_tag <- 'epoc_LexNoDelay_Cue'
elec_grps <- c('Auditory_delay','Sensorymotor_delay','Motor_delay','Delay_only')
# features <- c('aco','pho','Frq','Uni_Pos_SC')
# features <- c('aco','pho','wordness','Wordvec')
features <- c('aud_onset','resp_onset')
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

#Load Word2vec parameters
word2vec_path <- paste(home_dir,
                  "data/word_to_embedding_pca.csv",
                  sep = "")
word2vec_fea <- read.csv(word2vec_path,row.names = 1)
word2vec_fea_T <- as.data.frame(t(word2vec_fea))
word2vec_fea_T$stim <- rownames(word2vec_fea_T)
word2vec_fea_T <- word2vec_fea_T[, c("stim", setdiff(names(word2vec_fea_T), "stim"))]

#Load word freq parameters
freq_path <- paste(home_dir,"data/word_freq.csv",sep = "")
freq_fea <- read.csv(freq_path)
cols_to_normalize <- c(
  "Frq", "Uni_SC", "Uni_Pos_SC", "Uni_FW", "Uni_Pos_FW",
  "Bi_SC", "Bi_Pos_SC", "Bi_FW", "Bi_Pos_FW", "Tri_SC",
  "Tri_Pos_SC", "Tri_FW", "Tri_Pos_FW"
)
normalized_freq_fea <- freq_fea
normalized_freq_fea[, cols_to_normalize] <- lapply(freq_fea[, cols_to_normalize], scale)
freq_fea<-normalized_freq_fea
rm(normalized_freq_fea)

#%% Start looping
for (feature in features){
  for (elec_grp in elec_grps){
    
    #%% Run computations
    a <- a + 1
    if (task_ID > 0 && a != task_ID) {
      next
    }
    
    #%% Load files
    cat("loading files \n")
    # slurm task selection
    file_path_long <- paste(home_dir,
                       "data/",del_nodel_tag,"_",phase,"_",elec_grp,"_long.csv",
                       sep = "")
    long_data <- read.csv(file_path_long)
    long_data$time <- as.numeric(long_data$time)
    
    na_fea_rows <- is.na(long_data[[feature]])
    long_data <- long_data[!na_fea_rows, ] 
    
    # #%% append acoustic features
    # long_data <- left_join(long_data,aco_fea_T,by='stim')
    # 
    # #%% append phonemic features
    # long_data <- left_join(long_data,pho_fea_T,by='stim')
    # 
    # #%% append word2vec features
    # long_data <- left_join(long_data,word2vec_fea_T,by='stim')
    
    # #%% append word frequency features
    # long_data <- long_data %>%
    #   left_join(
    #     freq_fea %>% select(stim, 'Uni_Pos_SC'),
    #     by = "stim"
    #   )
    
    long_data$time <- as.numeric(long_data$time)
    time_points <- unique(long_data$time)
      
    # Split electrodes
    electrode_list <- split(long_data, long_data$electrode)
    rm(long_data)
    for (electrode_name in names(electrode_list)) {
      
        current_electrode_df <- electrode_list[[electrode_name]]
        
        cat("Re-formatting long data \n")
        data_by_time <- split(current_electrode_df, current_electrode_df$time)
        rm(current_electrode_df)
        
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
        write.csv(perm_compare_df,paste(home_dir,"results/",ana_tag,'_',elec_grp,"_",electrode_name,"_",phase,"_",feature,".csv",sep = ''),row.names = FALSE)
    }
  }
}