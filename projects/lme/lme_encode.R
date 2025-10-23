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
  
  model_type='full'
  
  tp <- current_data$time[1]
  if (model_type=='simple'){
    fml_bsl<-as.formula(paste0('value ~ 1'))
    if (feature=='aco'){
      fml<-as.formula(paste0('value ~ 1+',paste0(paste0("aco", 1:9), collapse = " + ")))
    }else if (feature=='pho'){
      fml<-as.formula(paste0('value ~ 1+',paste0(paste0("pho", 1:4), collapse = " + ")))
    }else{
      fml<-as.formula(paste0('value ~ 1+',feature))
    }
  }else if (model_type=='full'){
    if (feature=='aco'){
      fml_bsl<-as.formula(paste0('value ~ 1'))
      # fml_bsl <- as.formula(paste0('value ~ 1+',paste0(c(paste0('pho', 1:9)), collapse = ' + ')))
      # fml <- as.formula(paste0('value ~ 1+',paste0(c(paste0('aco', 1:9), paste0('pho', 1:9)), collapse = ' + ')))
      fml <- as.formula(paste0('value ~ 1+',paste0(c(paste0('aco', 1:9)), collapse = ' + ')))
    }else if (feature=='vow'){
      fml_bsl <- as.formula(paste0('value ~ 1+',paste0(c(paste0('aco', 1:9)), collapse = ' + ')))
      fml <- as.formula(paste0('value ~ 1+',paste0(c(paste0('aco', 1:9), paste0("vow", 1:6)), collapse = ' + ')))
    }else if (feature=='con'){
      fml_bsl <- as.formula(paste0('value ~ 1+',paste0(c(paste0('aco', 1:9)), collapse = ' + ')))
      fml <- as.formula(paste0('value ~ 1+',paste0(c(paste0('aco', 1:9), paste0("con", 1:6)), collapse = ' + ')))
    }else if (feature=='wordness'){
      fml_bsl <- as.formula(paste0('value ~ 1+',paste0(c(paste0('aco', 1:9), paste0("pho", 1:4)), collapse = ' + ')))
      fml <-as.formula(paste0('value ~ 1+',
                              paste0(paste0("aco", 1:9), collapse = " + "),"+",paste0(paste0("pho", 1:4), collapse = " + "),"+",
                                paste0(paste0("aco", 1:9,":",feature), collapse = " + "),"+",paste0(paste0("pho", 1:9,":",feature), collapse = " + "),"+",feature))
    }else if (feature=='Wordvec'){
      fml_bsl <- as.formula(paste0('value ~ 1+',paste0(c(paste0('aco', 1:9), paste0("pho", 1:4)), collapse = ' + ')))
      fml <- as.formula(paste0('value ~ 1+',paste0(c(paste0('aco', 1:9), paste0("pho", 1:4), paste0('Wordvec', 1:91)),collapse = ' + ')))
    }else{
      fml_bsl <- as.formula(paste0('value ~ 1+',paste0(c(paste0('aco', 1:9)), collapse = ' + ')))
      fml <-as.formula(paste0('value ~ 1+',paste0(paste0("aco", 1:9), collapse = " + "),"+",paste0(paste0("pho", 1:4), collapse = " + ")))
    }
  }

  m <- lm(fml, data = current_data,na.action = na.exclude)
  m_bsl <- lm(fml_bsl, data = current_data,na.action = na.exclude)
  anova_results <- anova(m_bsl, m)
  mean_F_stat <- anova_results$`F`[2]
  # pattern <- paste0("^", feature)
  # mean_F_stat <- mean(abs(coef(m)[grep(pattern, names(coef(m)))]))
  
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
#elec_grps <- c('Hickok_Spt','Hickok_lPMC','Hickok_lIPL','Hickok_lIFG')
# features <- c('aco','pho','Frq','Uni_Pos_SC')
#features <- c('aco','pho','wordness','Wordvec')
#features <- paste0("pho", 1:4)
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
    
    #%% Load files
    cat("loading files \n")
    # slurm task selection
    file_path_long <- paste(home_dir,
                       "data/epoc_LexDelayRep_Aud_",elec_grp,"_long.csv",
                       sep = "")
    long_data <- read.csv(file_path_long)
    long_data$time <- as.numeric(long_data$time)
    
    bsl_corr=FALSE
    
    if (bsl_corr==TRUE){
    file_path_wide <- paste(home_dir,
                            "data/epoc_LexDelayRep_Aud_",elec_grp,"_wide.csv",
                            sep = "")
    wide_data <- read.csv(file_path_wide)

    #%% Baseline correction
    cat("correcting baseline \n")
    windowed_means <- long_data %>%
      filter(time > -0.4 & time <= 0) %>%
      mutate(
        time_window = floor((time - (-0.4)) / 0.1)
      ) %>%
      group_by(subject, electrode, stim, time_window) %>%
      summarise(mean_val = mean(value, na.rm = TRUE), .groups = 'drop') %>%
      pivot_wider(
        names_from = time_window,
        values_from = mean_val,
        names_prefix = "mean_window_"
      )
    rm(long_data)
    wide_data <- wide_data %>%
      left_join(windowed_means, by = c("subject", "electrode", "stim"))
    all_time_col_names <- names(wide_data)[str_detect(names(wide_data), "^X")]
    baseline_predictor_cols <- names(wide_data)[str_detect(names(wide_data), "mean_window_")]
    regression_data <- wide_data
    for (y_col in all_time_col_names) {
      fml <- as.formula(paste(y_col, "~", paste(baseline_predictor_cols, collapse = " + ")))
      model <- lm(fml, data = wide_data,na.action = na.exclude)
      regression_data[[y_col]] <- residuals(model)
    }
    wide_data <- regression_data %>%
      select(-starts_with("mean_window"))
    rm(regression_data)
    
    #%% transform to long
    cat("transforming to long \n")
    long_data <- wide_data %>%
      pivot_longer(
        cols = starts_with("X"),
        names_to = "time_label",
        values_to = "value"
      ) %>%
      mutate(
        time = case_when(
          str_detect(time_label, "X\\.") ~ as.numeric(str_replace(time_label, "X\\.", "-")),
          TRUE ~ as.numeric(str_replace(time_label, "X", ""))
        )
      )%>%
      select(-time_label)
    }
    
    #%% get only word part of the "stim"
    long_data <- long_data %>%
      mutate(
        stim = str_split_fixed(string = stim, pattern = "-", n = 2)[, 1]
      )
    
    #%% average across stim
    # long_data <- long_data %>%
    #   group_by(subject, electrode, time, wordness,stim) %>%
    #   summarise(
    #     value = mean(value, na.rm = TRUE),
    #     .groups = 'drop'
    #   ) %>%
    #   ungroup()


    #%% append acoustic features
    long_data <- left_join(long_data,aco_fea_T,by='stim')
    
    #%% append vowel features
    long_data <- left_join(long_data,vow_fea_T,by='stim')

    #%% append consonant features
    long_data <- left_join(long_data,con_fea_T,by='stim')
    
    #%% append word2vec features
    long_data <- left_join(long_data,word2vec_fea_T,by='stim')
    
    #%% append word frequency features
    long_data <- long_data %>%
      left_join(
        freq_fea %>% select(stim, 'Uni_Pos_SC'),
        by = "stim"
      )
    
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