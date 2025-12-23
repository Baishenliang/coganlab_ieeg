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
model_func <- function(current_data){
  
  # Loading packages
  os_type <- Sys.info()['sysname']
  if (os_type == "Windows"){
    library(lme4)
    library(lmerTest)
    library(tidyverse)
    library(glmnet)
  }else if (os_type == "Linux"){
    library(lme4, lib.loc = "~/lab/bl314/rlib")
    library(lmerTest, lib.loc = "~/lab/bl314/rlib")
    library(tidyverse, lib.loc = "~/lab/bl314/rlib")
    library(glmnet, lib.loc = "~/lab/bl314/rlib")
  }
  
  # Normalize data
  current_data <- current_data %>%
    group_by(subject) %>% # Add 'electrode' here if you have multiple electrodes: group_by(subject, electrode)
    mutate(
      session_mean = mean(value, na.rm = TRUE),
      session_sd = sd(value, na.rm = TRUE),
      # Overwrite the original 'value' column
      value = (value - session_mean) / session_sd
    ) %>%
    ungroup() %>%
    select(-session_mean, -session_sd)
  
  # Ridge regression model
  
  ridge_cv_predict <- function(fml, current_data, alpha, lambda_val, k_folds = 10, seed = 123) {
    
    set.seed(seed)
    
    mf <- model.frame(fml, data = current_data, na.action = na.exclude)
    X <- model.matrix(fml, data = mf)[, -1]  
    y <- model.response(mf)                
    n <- length(y)
    
    fold_id <- sample(rep(1:k_folds, length.out = n))
    pred_cv <- numeric(n) 
    lambdas_used <- numeric(k_folds)
    
    for (i in 1:k_folds) {
      train_idx <- which(fold_id != i)
      test_idx <- which(fold_id == i)
      
      X_train <- X[train_idx, ]
      y_train <- y[train_idx]
      X_test <- X[test_idx, ]
      
      current_fold_lambda <- NULL
      
      if (lambda_val <= 0) {
        inner_cv <- cv.glmnet(
          x = X_train, 
          y = y_train, 
          alpha = alpha, 
          nfolds = 5,
          standardize = TRUE 
        )
        current_fold_lambda <- inner_cv$lambda.min
      } else {
        current_fold_lambda <- lambda_val
      }
      
      lambdas_used[i] <- current_fold_lambda
      
      fit_fold <- glmnet(
        x = X_train, 
        y = y_train, 
        alpha = alpha, 
        lambda = current_fold_lambda,
        standardize = TRUE 
      )
      
      pred_test <- predict(object = fit_fold, s = current_fold_lambda, newx = X_test)
      
      pred_cv[test_idx] <- pred_test
    }
    
    fit_final <- glmnet(
      x = X, 
      y = y, 
      alpha = alpha, 
      lambda = median(lambdas_used),
      standardize = TRUE
    )
    
    coefficients <- coef(fit_final, s = median(lambdas_used))
    
    cor_result <- cor.test(pred_cv, y)
    rho <- as.numeric(cor_result$estimate)
    p_value <- cor_result$p.value
    if (rho < 0){
      rho = 0
      p_value = 0.9
    }
    
    result_list <- list(
      Lambda_Used = median(lambdas_used),
      Correlation_Coefficient = rho,
      P_Value = p_value,
      Coefficients = coefficients
    )
    
    return(result_list)
  }
  
  tp <- current_data$time[1]
  fml <- as.formula(paste0('value ~ 1+',paste0(c(paste0('aco', 1:9,"*wordness"), paste0("pho", 1:11,"*wordness"),"wordness"), collapse = ' + ')))
  #fml <- as.formula(paste("value ~ 1 +", paste(c(paste0("aco", 1:9), paste0("pho", 1:11), paste0("sem", 1:67)), collapse = " + ")))
  #fml <- as.formula(paste("value ~ 1 +", paste(c(paste0("aco", 1:9), paste0("pho", 1:11)), collapse = " + ")))
  ridge_alpha <- 0
  ridge_lambda_vWM <- current_data$ridge_lambda_vWM[1]
  # ridge_lambda_vWM <- -1
  # lambda < 0: do CV and get optimal lambda
  # lambda > 0 : do ridge with fixed lambda 

  # machine learning version
  ridge_vWM<-ridge_cv_predict(fml, current_data, ridge_alpha,ridge_lambda_vWM)

  perm_compare_df_i <- data.frame(
    perm = 0,
    time_point = tp,
    ACC_vWM = ridge_vWM$Correlation_Coefficient,
    ACC_vWM_p = ridge_vWM$P_Value,
    vWM_lambda = ridge_vWM$Lambda_Used)
  
  coef_vWM_rn_sparse <- ridge_vWM$Coefficients
  coef_vWM_vec <- as.numeric(coef_vWM_rn_sparse)
  names(coef_vWM_vec) <- rownames(coef_vWM_rn_sparse)
  names(coef_vWM_vec) <- paste0(names(coef_vWM_vec), '_vWM')
  coef_vWM_rn <- as_tibble_row(coef_vWM_vec)
  
  perm_compare_df_i <- bind_cols(perm_compare_df_i,coef_vWM_rn)
  
  # Permutation
  cat('Start perm \n')
  n_perm <- 0#3e2#1e3
  
  if (n_perm>0){
    for (i_perm in 1:n_perm) {
      set.seed(10000 + i_perm)
      
      # Permutation 1: permute word-trial mapping
      
      relable_perm <- function(data) {
        
        data_perm_relabeled <- data %>%
          group_by(subject, electrode) %>%
          mutate(
            perm_indices = sample(1:n()),
            across(
              starts_with('aco') | starts_with('pho') | starts_with('word'), 
              #starts_with('aco') | starts_with('pho') | starts_with('sem'), 
              #starts_with('aco') | starts_with('pho'), 
              ~ .x[perm_indices]
            )
          ) %>%
          ungroup() %>%
          select(-perm_indices)
        
        return(data_perm_relabeled)
      }
      
      current_data_vWM_perm_relable <- relable_perm(current_data_vWM)

      ridge_vWM_relable_perm<-ridge_cv_predict(fml, current_data_vWM_perm_relable, ridge_alpha,ridge_lambda_vWM)

      # Store permutation data
      
      perm_compare_df_i_perm <- data.frame(
        perm = i_perm,
        time_point = tp,
        ACC_vWM = ridge_vWM_relable_perm$Correlation_Coefficient,
        ACC_vWM_p = ridge_vWM_relable_perm$P_Value,
        vWM_lambda = ridge_vWM_relable_perm$Lambda_Used)
      
      coef_vWM_rn_sparse <- ridge_vWM_relable_perm$Coefficients
      coef_vWM_vec <- as.numeric(coef_vWM_rn_sparse)
      names(coef_vWM_vec) <- rownames(coef_vWM_rn_sparse)
      names(coef_vWM_vec) <- paste0(names(coef_vWM_vec), '_vWM')
      coef_vWM_rn_perm <- as_tibble_row(coef_vWM_vec)
      
      perm_compare_df_i_perm <- bind_cols(perm_compare_df_i_perm,coef_vWM_rn_perm)
      
      perm_compare_df_i <- rbind(
        perm_compare_df_i,
        perm_compare_df_i_perm
      )
      
      
    }
  }
  return(perm_compare_df_i)
}

#%% Parameters
delay_nodelays <- c("LexDelayRep")#c("LexDelayRep","LexNoDelay")
alignments <- c("Aud","Go","Resp")
#alignments <- c("Resp")
# alignments <- c("Aud")
#elec_grps <- c('Auditory','Sensorymotor','Motor','Delay_only','Wgw_p55b','Wgw_a55b','SM_vWM_Auditory_early','SM_vWM_Auditory_late','SM_vWM_Delay','SM_vWM_Motor')
elec_grps <- c('Auditory','Sensorymotor','Motor','Delay_only','Wgw_p55b','Wgw_a55b')
#elec_grps <- c('Wgw_p55b','Wgw_a55b')
#elec_grps <- c('Motor')

a = 0
#Make fixed lambda (from cv or anything optimized)

ridge_lambda_speech <- data.frame( # lambda adjusted according to electrode size
  vWM = c(1e-5,  # Auditory vWM
          0.1, # Sensorymotor vWM
          1, # Motor vWM
          10, # Delay only vWM
          0.001,# Wgw_p55b
          0.001)# Wgw_a55b
)
rownames(ridge_lambda_speech) <- c("Auditory", "Sensorymotor", "Motor","Delay_only",'Wgw_p55b','Wgw_a55b')#,'SM_vWM_Auditory_early','SM_vWM_Auditory_late','SM_vWM_Delay','SM_vWM_Motor')


ridge_lambda_nonword <- data.frame( # lambda adjusted according to electrode size
  vWM = c(0.001,  # Auditory vWM
          0.001, # Sensorymotor vWM
          0.001, # Motor vWM
          0.001, # Delay only vWM
          0.001,# Wgw_p55b
          0.001)# Wgw_a55b
)
rownames(ridge_lambda_nonword) <- c("Auditory", "Sensorymotor", "Motor","Delay_only",'Wgw_p55b','Wgw_a55b')#,'SM_vWM_Auditory_early','SM_vWM_Auditory_late','SM_vWM_Delay','SM_vWM_Motor')


ridge_lambda_semantics <- data.frame( # lambda adjusted according to electrode size
  vWM = c(1e-5,  # Auditory vWM
          1, # Sensorymotor vWM
          1e-5, # Motor vWM
          10, # Delay only vWM
          0.001,# Wgw_p55b
          10)# Wgw_a55b
)
rownames(ridge_lambda_semantics) <- c("Auditory", "Sensorymotor", "Motor","Delay_only",'Wgw_p55b','Wgw_a55b')#,'SM_vWM_Auditory_early','SM_vWM_Auditory_late','SM_vWM_Delay','SM_vWM_Motor')

#Load acoustic parameters
aco_path <- paste(home_dir,
                  "data/envelope_feature_dict_pca.csv",
                  sep = "")
aco_fea <- read.csv(aco_path,row.names = 1)
aco_fea_T <- as.data.frame(t(aco_fea))
aco_fea_T$stim <- rownames(aco_fea_T)
aco_fea_T <- aco_fea_T[, c("stim", setdiff(names(aco_fea_T), "stim"))]

#Load phonetic parameters
pho_path <- paste(home_dir,
                  "data/phoneme_one_hot_dict_pca.csv",
                  sep = "")
pho_fea <- read.csv(pho_path,row.names = 1)
pho_fea_T <- as.data.frame(t(pho_fea))
pho_fea_T$stim <- rownames(pho_fea_T)
pho_fea_T <- pho_fea_T[, c("stim", setdiff(names(pho_fea_T), "stim"))]

#Load semantic parameters
sem_path <- paste(home_dir,
                  "data/syllables_sem_pca.csv",
                  sep = "")
sem_fea <- read.csv(sem_path,row.names = 1)
sem_fea_T <- as.data.frame(t(sem_fea))
sem_fea_T$stim <- rownames(sem_fea_T)
sem_fea_T <- sem_fea_T[, c("stim", setdiff(names(sem_fea_T), "stim"))]

#%% Start looping
#for (ridge_lambda in list(ridge_lambda_nonword)){#list(ridge_lambda1,ridge_lambda2)){
for (lambda_test in c(0.00001,0.0001,0.001,0.01,0.1,1,10,100,1000,10000)){
#for (lambda_test in c(20,40,60,80,200,500,1000,10000)){
#for (lambda_test in c(0.2,0.4,0.6,0.8,2,4,6,8)){
  for (delay_nodelay in delay_nodelays){
    for (alignment in alignments){
      for (elec_grp in elec_grps){
        
        if (delay_nodelay=='LexNoDelay' && elec_grp=='Delay_only'){
          next
        }
        
        if (delay_nodelay=='LexNoDelay' && alignment=='Go'){
          next
        }
        
        a <- a + 1
        if (task_ID > 0 && a != task_ID) {
          next
        }
        
        
        #%% Load files
        cat("loading files \n")
        # slurm task selection
        # vwm electrodes
        if (elec_grp=='Delay_only' || elec_grp== 'Wgw_p55b' || elec_grp=='Wgw_a55b' || elec_grp=='SM_vWM_Auditory_early' || 
            elec_grp=='SM_vWM_Auditory_late' || elec_grp=='SM_vWM_Delay' || elec_grp=='SM_vWM_Motor'){
          file_path_long_vwm <- paste(home_dir,
                                      "data/epoc_LexDelayRep_",alignment,"_",elec_grp,"_pow_long.csv",
                                      sep = "")
        }else{
          file_path_long_vwm <- paste(home_dir,
                                      "data/epoc_LexDelayRep_",alignment,"_",elec_grp,"_vWM_pow_long.csv",
                                      sep = "")
        }
        long_data <- read.csv(file_path_long_vwm)
        
        #%% get only word part of the "stim"
        long_data <- long_data %>%
          mutate(
            stim = str_split_fixed(string = stim, pattern = "-", n = 2)[, 1]
          )
        
        #%% append acoustic features
        long_data <- left_join(long_data,aco_fea_T,by='stim')
        
        #%% append phonemic features
        long_data <- left_join(long_data,pho_fea_T,by='stim')
        
        long_data$time <- as.numeric(long_data$time)
        time_points <- unique(long_data$time)
        
        #for (lex in c("Word","Nonword",'All')){
        lex<-'All'
        if (lex=='Word'){
          word_data <- long_data[long_data['wordness']==lex,]
          #%% append semantic features
          word_data <- left_join(word_data,sem_fea_T,by='stim')
        } else {
          word_data <- long_data
        }
        rm(long_data)
        #%% Run computations
        
        #%% append ridge lambdas
        # word_data$ridge_lambda_vWM<-ridge_lambda[elec_grp,'vWM']
        word_data$ridge_lambda_vWM<-lambda_test
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
        perm_compare_df<-parLapply(cl, data_by_time, model_func)
        stopCluster(cl)
        perm_compare_df <- do.call(rbind, perm_compare_df)
        perm_compare_df <- perm_compare_df %>% arrange(time_point)
        
        print(perm_compare_df)
        
        #write.csv(perm_compare_df,paste(home_dir,"results/",delay_nodelay,"_",elec_grp,"_",alignment,"_",lex,"_pow_vWMλ_",ridge_lambda[elec_grp,'vWM'],".csv",sep = ''),row.names = FALSE)
        write.csv(perm_compare_df,paste(home_dir,"results/",delay_nodelay,"_",elec_grp,"_",alignment,"_",lex,"_pow_testλ_",lambda_test,".csv",sep = ''),row.names = FALSE)
  
        }
      }
    }
  }
#}
  #}