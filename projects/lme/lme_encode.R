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
  
  # Ridge regression model
  ridge <- function(fml, current_data, alpha, lambda_val) {
    
    mf <- model.frame(fml, data = current_data, na.action = na.exclude)
    X <- model.matrix(fml, data = mf)[, -1]
    y <- model.response(mf)
    
    final_lambda <- NULL
    model_fit <- NULL
    
    if (lambda_val <= 0) {
      
      lambda_seq <- 10^seq(4, -4, by = -.1) 
      
      ridge_cv <- cv.glmnet(
        x = X, 
        y = y, 
        alpha = alpha,
        lambda = lambda_seq,
        nfolds = 10 
      )
      
      final_lambda <- ridge_cv$lambda.min
      model_fit <- ridge_cv
      
      pred <- predict(object = model_fit, s = final_lambda, newx = X)
      
    } else {
      
      final_lambda <- lambda_val
      
      fit <- glmnet(
        x = X, 
        y = y, 
        alpha = alpha,
        lambda = final_lambda
      )
      
      model_fit <- fit
      pred <- predict(object = model_fit, s = final_lambda, newx = X)
    }
    
    raw_coefs <- coef(model_fit, s = final_lambda) 
    coefs <- as.vector(raw_coefs)
    names(coefs) <- rownames(raw_coefs)
    
    actual <- y
    rss <- sum((pred - actual) ^ 2)
    tss <- sum((actual - mean(actual)) ^ 2)
    rsq <- 1 - rss/tss
    n <- length(y)
    mse_train <- rss / n
    p <- ncol(X)
    adj_rsq <- 1 - ((1 - rsq) * (n - 1) / (n - p - 1))
    
    result_list <- list(
      R2_Train = adj_rsq,
      Lambda_Used = final_lambda,
      Coefficients = coefs
    )
    
    return(result_list)
  }
  
  ridge_cv_predict <- function(fml, current_data, alpha, lambda_val, k_folds = 10) {
    
    mf <- model.frame(fml, data = current_data, na.action = na.exclude)
    X <- model.matrix(fml, data = mf)[, -1] 
    y <- model.response(mf)                
    n <- length(y)
    
    final_lambda <- NULL
    if (lambda_val <= 0) {
      lambda_seq <- 10^seq(4, -4, by = -.1) 
      ridge_cv <- cv.glmnet(
        x = X, 
        y = y, 
        alpha = alpha,
        lambda = lambda_seq,
        nfolds = 10 
      )
      final_lambda <- ridge_cv$lambda.min
    } else {
      final_lambda <- lambda_val
    }
    
    fold_id <- sample(rep(1:k_folds, length.out = n))
    pred_cv <- numeric(n) 
    
    for (i in 1:k_folds) {
      train_idx <- which(fold_id != i)
      test_idx <- which(fold_id == i)
      
      X_train <- X[train_idx, ]
      y_train <- y[train_idx]
      X_test <- X[test_idx, ]
      
      fit_fold <- glmnet(
        x = X_train, 
        y = y_train, 
        alpha = alpha, 
        lambda = final_lambda 
      )
      
      pred_test <- predict(object = fit_fold, s = final_lambda, newx = X_test)
      
      pred_cv[test_idx] <- pred_test
    }
    
    cor_result <- cor.test(pred_cv, y)
    
    rho <- cor_result$estimate
    p_value <- cor_result$p.value
    
    result_list <- list(
      Lambda_Used = final_lambda,
      Correlation_Coefficient = as.numeric(rho),
      P_Value = p_value
    )
    
    return(result_list)
  }
  
  tp <- current_data$time[1]
  fml <- as.formula(paste0('value ~ 1+',paste0(c(paste0('aco', 1:9,"*wordness"), paste0("pho", 1:11,"*wordness"),"wordness"), collapse = ' + ')))
  ridge_alpha <- 0
  ridge_lambda <- -1
  # lambda < 0: do CV and get optimal lambda
  # lambda > 0 : do ridge with fixed lambda 
  current_data_vWM <- current_data[current_data$vWM == 1, ]
  current_data_novWM <- current_data[current_data$vWM == 0, ]
  
  # regression version
  # ridge_vWM<-ridge(fml, current_data_vWM, ridge_alpha,ridge_lambda)
  # ridge_novWM<-ridge(fml, current_data_novWM, ridge_alpha,ridge_lambda)
  # perm_compare_df_i <- data.frame(
  #   perm = 0,
  #   time_point = tp,
  #   R2_vWM = ridge_vWM$R2_Train,
  #   R2_novWM = ridge_novWM$R2_Train,
  #   R2_diff = ridge_vWM$R2_Train-ridge_novWM$R2_Train
  # )
  # 
  
  # machine learning version
  ridge_vWM<-ridge_cv_predict(fml, current_data_vWM, ridge_alpha,ridge_lambda)
  ridge_novWM<-ridge_cv_predict(fml, current_data_novWM, ridge_alpha,ridge_lambda)
  
  perm_compare_df_i <- data.frame(
    perm = 0,
    time_point = tp,
    vWM_ACC = ridge_vWM$Correlation_Coefficient,
    vWM_p = ridge_vWM$P_Value,
    vWM_lambda = ridge_vWM$Lambda_Used,
    novWM_ACC = ridge_novWM$Correlation_Coefficient,
    novWM_p = ridge_novWM$P_Value,
    novWM_lambda = ridge_novWM$Lambda_Used
  )
  
  # coef_vWM_rn <- ridge_vWM$Coefficients
  # names(coef_vWM_rn) <- paste0(names(coef_vWM_rn), 'vWM')
  # coef_vWM_rn <- as.data.frame(as.list(coef_vWM_rn))
  # coef_novWM_rn <- ridge_novWM$Coefficients
  # names(coef_novWM_rn) <- paste0(names(coef_novWM_rn), 'novWM')
  # coef_novWM_rn <- as.data.frame(as.list(coef_novWM_rn))
  # perm_compare_df_i <- bind_cols(perm_compare_df_i,coef_vWM_rn,coef_novWM_rn)
  # perm_compare_df_i <- bind_cols(perm_compare_df_i,coef_vWM_rn)
  # 
  
  # Permutation
  cat('Start perm \n')
  n_perm <- 0#1e3
  
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
              ~ .x[perm_indices]
            )
          ) %>%
          ungroup() %>%
          select(-perm_indices)
        
        return(data_perm_relabeled)
      }
      
      current_data_vWM_perm_relable <- relable_perm(current_data_vWM)
      current_data_novWM_perm_relable <- relable_perm(current_data_novWM)
      
      ridge_vWM_relable_perm<-ridge(fml, current_data_vWM_perm_relable, ridge_alpha,ridge_lambda)
      ridge_novWM_relable_perm<-ridge(fml, current_data_novWM_perm_relable, ridge_alpha,ridge_lambda)
      
      
      # Permutation 2: permute delay and nodelay electrodes (balancing electrode numbers)
      sample_control_strategy<-'adjustR' # 'adjustSample'
      
      if (sample_control_strategy=='adjustSample'){
        current_data_base<-current_data
        current_data_vWM_base <- current_data_base[current_data_base$vWM == 1, ]
        current_data_novWM_base <- current_data_base[current_data_base$vWM == 0, ] 
        
        n_vWM <- nrow(current_data_vWM_base)
        n_novWM <- nrow(current_data_novWM_base)
        
        target_n <- n_vWM
        
        if (n_novWM < target_n) {
          
          sample_indices <- sample(
            x = 1:n_novWM, 
            size = target_n, 
            replace = TRUE
          )
          
          current_data_novWM_base <- current_data_novWM_base[sample_indices, ]
          
        } else if (n_novWM > target_n) {
          
          sample_indices <- sample(
            x = 1:n_novWM, 
            size = target_n, 
            replace = FALSE
          )
          
          current_data_novWM_base <- current_data_novWM_base[sample_indices, ]
          
        } else {
          current_data_novWM_base <- current_data_novWM_base
        }
        
        current_data_base<-rbind(current_data_vWM_base,current_data_novWM_base)
        
        original_map <- current_data_base %>%
          distinct(subject, electrode, vWM)
        
        vwm_list_to_shuffle <- original_map$vWM
        shuffled_vwm <- sample(vwm_list_to_shuffle)
        
        permuted_vwm_map <- original_map %>%
          mutate(vWM_permuted = shuffled_vwm) %>%
          select(subject, electrode, vWM_permuted)
        
        current_data_perm_shufflevWM <- current_data_base %>%
          left_join(permuted_vwm_map, by = c("subject", "electrode"), relationship = "many-to-one") %>%
          mutate(vWM = vWM_permuted) %>%
          select(-vWM_permuted)
        
        current_data_vWM_perm_shufflevWM <- current_data_perm_shufflevWM[current_data_perm_shufflevWM$vWM == 1, ]
        current_data_novWM_perm_shufflevWM <- current_data_perm_shufflevWM[current_data_perm_shufflevWM$vWM == 0, ] 
        
        ridge_vWM_shufflevWM_perm<-ridge(fml, current_data_vWM_perm_shufflevWM, ridge_alpha,ridge_lambda)
        ridge_novWM_shufflevWM_perm<-ridge(fml, current_data_novWM_perm_shufflevWM, ridge_alpha,ridge_lambda)
      }
      else if (sample_control_strategy=='adjustR'){
        original_map <- current_data %>%
          distinct(subject, electrode, vWM)
        
        vwm_list_to_shuffle <- original_map$vWM
        shuffled_vwm <- sample(vwm_list_to_shuffle)
        
        permuted_vwm_map <- original_map %>%
          mutate(vWM_permuted = shuffled_vwm) %>%
          select(subject, electrode, vWM_permuted)
        
        current_data_perm_shufflevWM <- current_data %>%
          left_join(permuted_vwm_map, by = c("subject", "electrode"), relationship = "many-to-one") %>%
          mutate(vWM = vWM_permuted) %>%
          select(-vWM_permuted)
        
        current_data_vWM_perm_shufflevWM <- current_data_perm_shufflevWM[current_data_perm_shufflevWM$vWM == 1, ]
        current_data_novWM_perm_shufflevWM <- current_data_perm_shufflevWM[current_data_perm_shufflevWM$vWM == 0, ] 
        
        ridge_vWM_shufflevWM_perm<-ridge(fml, current_data_vWM_perm_shufflevWM, ridge_alpha,ridge_lambda)
        ridge_novWM_shufflevWM_perm<-ridge(fml, current_data_novWM_perm_shufflevWM, ridge_alpha,ridge_lambda)
      }
      # Store permutation data
      
      perm_compare_df_i_perm <- data.frame(
        perm = i_perm,
        time_point = tp,
        R2_vWM = ridge_vWM_relable_perm$R2_Train,
        R2_novWM = ridge_novWM_relable_perm$R2_Train,
        R2_diff =  ridge_vWM_shufflevWM_perm$R2_Train-ridge_novWM_shufflevWM_perm$R2_Train
      )
      
      # coef_vWM_rn_perm <- coef_vWM_perm
      # names(coef_vWM_rn_perm) <- paste0(names(coef_vWM_rn_perm), 'vWM')
      # coef_vWM_rn_perm <- as.data.frame(as.list(coef_vWM_rn_perm))
      # coef_novWM_rn_perm <- coef_novWM_perm
      # names(coef_novWM_rn_perm) <- paste0(names(coef_novWM_rn_perm), 'novWM')
      # coef_novWM_rn_perm <- as.data.frame(as.list(coef_novWM_rn_perm))
      # perm_compare_df_i_perm <- bind_cols(perm_compare_df_i_perm,coef_vWM_rn_perm,coef_novWM_rn_perm)
      # # perm_compare_df_i_perm <- bind_cols(perm_compare_df_i_perm,coef_vWM_rn_perm)
      perm_compare_df_i <- rbind(
        perm_compare_df_i,
        perm_compare_df_i_perm
      )
  
      
    }
  }
  return(perm_compare_df_i)
}

#%% Parameters
alignments <- c("Aud","Go","Resp")
elec_grps <- c('Auditory','Sensorymotor','Motor')
a = 0
ridge_lambda <- data.frame(
  vWM = c(0.125892541179417,  # Auditory vWM
          0.0158489319246111, # Sensorymotor vWM
          0.158489319246111), # Motor vWM
  
  novWM = c(0.0158489319246111, # Auditory novWM
            0.0794328234724281, # Sensorymotor novWM
            0.0001)             # Motor novWM
)
rownames(ridge_lambda) <- c("Auditory", "Sensorymotor", "Motor")

#Load acoustic parameters
aco_path <- paste(home_dir,
                  "data/envelope_feature_dict_pca.csv",
                  sep = "")
aco_fea <- read.csv(aco_path,row.names = 1)
aco_fea_T <- as.data.frame(t(aco_fea))
aco_fea_T$stim <- rownames(aco_fea_T)
aco_fea_T <- aco_fea_T[, c("stim", setdiff(names(aco_fea_T), "stim"))]

#Load vowel parameters
pho_path <- paste(home_dir,
                  "data/phoneme_one_hot_dict_pca.csv",
                  sep = "")
pho_fea <- read.csv(pho_path,row.names = 1)
pho_fea_T <- as.data.frame(t(pho_fea))
pho_fea_T$stim <- rownames(pho_fea_T)
pho_fea_T <- pho_fea_T[, c("stim", setdiff(names(pho_fea_T), "stim"))]


#%% Start looping
for (alignment in alignments){
  for (elec_grp in elec_grps){
    a <- a + 1
    if (task_ID > 0 && a != task_ID) {
      next
    }
    
    
    #%% Load files
    cat("loading files \n")
    # slurm task selection
    # vwm electrodes
    file_path_long_vwm <- paste(home_dir,
                                "data/epoc_LexDelayRep_",alignment,"_",elec_grp,"_vWM_long.csv",
                                sep = "")
    long_data_vwm <- read.csv(file_path_long_vwm)
    long_data_vwm$time <- as.numeric(long_data_vwm$time)
    
    # no vWM electrodes
    file_path_long_novwm <- paste(home_dir,
                                  "data/epoc_LexDelayRep_",alignment,"_",elec_grp,"_novWM_long.csv",
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
    word_data <- long_data
    rm(long_data)
    
    #for (lex in c("Word","Nonword",'All')){
    lex<-'All'
    #%% Run computations
    
    #%% append ridge lambdas
    word_data$ridge_lambda_vWM<-ridge_lambda[elec_grp,'vWM']
    word_data$ridge_lambda_novWM<-ridge_lambda[elec_grp,'novWM']
    
    
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
    
    write.csv(perm_compare_df,paste(home_dir,"results/",elec_grp,"_",alignment,"_",lex,".csv",sep = ''),row.names = FALSE)
    #}
  }
}