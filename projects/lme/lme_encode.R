# install.packages("tidyverse")
# install.packages("lmerTest")
# install.packages("foreach")
# install.packages("doParallel")

library(tidyverse)
library(lme4)
library(lmerTest)
library(parallel)
library(foreach)
library(doParallel)

#%% Get core environment
num_cores <- detectCores()
cl <- makeCluster(num_cores - 1) 
registerDoParallel(cl)

#%% Parameters
n_perm <-30
set.seed(42)
features<-c('pho1','pho3')
align_to_onsets<-c('pho1','pho3')
post_align_T_threshold<-c(-0.2,1)

#%% Load files
home_dir <-"D:/bsliang_Coganlabcode/coganlab_ieeg/projects/lme/"
file_path <- paste(home_dir,"data/epoc_LexDelayRep_Aud_full_Auditory_delay_long.csv",sep="")
long_data_org <- read.csv(file_path)

for (feature in features){
  for (align_to_onset in align_to_onsets){
    #%% re-align to onsets and get timepoint
    long_data <- long_data_org %>%
      mutate(
        time_point = case_when(
          align_to_onset == 'org' ~ time_point - pho_t1,
          align_to_onset == 'pho1' ~ round(time_point - pho_t1, 2),
          align_to_onset == 'pho2' ~ round(time_point - pho_t2, 2),
          align_to_onset == 'pho3' ~ round(time_point - pho_t3, 2),
          align_to_onset == 'pho4' ~ round(time_point - pho_t4, 2),
          align_to_onset == 'pho5' ~ round(time_point - pho_t5, 2),
        )
      ) %>%
      filter((time_point >= post_align_T_threshold[1]) & (time_point <= post_align_T_threshold[2]))
    long_data$time <- as.numeric(long_data$time)
    time_points <- unique(long_data$time)
    
    # Add fea
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
    # results_df <- data.frame(
    #   time_point = numeric(),
    #   estimate = numeric(),
    #   p_value = numeric()
    # )
    
    compare_df <- data.frame(
      time_point = numeric(),
      chi_squared_comp =  numeric(),
      p_value_comp = numeric()
    )
    
    perm_compare_df <- data.frame(
      perm = numeric(),
      time_point = numeric(),
      chi_squared_obs = numeric(),
      p_value_perm = numeric()
    )
    
    for (tp in time_points) {
      cat(paste('Processing time point:', tp,'\n'))
      current_data <- filter(long_data, time == tp)
      
      # Modelling
      lme_model <- lmer(value ~ fea + (1 | subject) + (1 | electrode) + (1 | stim),
                        data = current_data,
                        REML = FALSE)
      # model_summary <- summary(lme_model)
      # pho1_effect_row <- grepl("pho1", rownames(model_summary$coefficients))
      # estimate <- model_summary$coefficients[pho1_effect_row, "Estimate"]
      # p_value <- model_summary$coefficients[pho1_effect_row, "Pr(>|t|)"]
      # 
      # results_df <- rbind(results_df, data.frame(
      #   time_point = tp,
      #   estimate = estimate,
      #   p_value = p_value
      # ))
      
      # Model comparison (to null)
      null_model <- lmer(value ~ 1 + (1 | subject) + (1 | electrode) + (1 | stim),
                         data = current_data,
                         REML = FALSE)
      model_comparison <- anova(null_model, lme_model)
      p_value_comp <- model_comparison$`Pr(>Chisq)`[2]
      observed_chisq  <- model_comparison$Chisq[2]
      
      compare_df <- rbind(compare_df, data.frame(
        time_point = tp,
        chi_squared_comp =  observed_chisq ,
        p_value_comp = p_value_comp
      ))
    
      # Permutation
      cat('Start perm')
      perm_compare_df_tp <- foreach(
        i_perm = 1:n_perm,
        .combine = 'rbind',
        .packages = c('lme4', 'dplyr')
      ) %dopar% {
        cat(paste(i_perm, ' perm in ', n_perm, '\n'))
        
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
        
        return(
          data.frame(
            perm = i_perm,
            time_point = tp,
            chi_squared_obs = observed_chisq_perm,
            p_value_perm = p_value_comp_perm
          )
        )
      }
      perm_compare_df<-rbind(perm_compare_df,perm_compare_df_tp)
      
    }
    
    # results_df <- results_df %>% arrange(time_point)
    compare_df <- compare_df %>% arrange(time_point)
    perm_compare_df <- perm_compare_df %>% arrange(time_point)
    
    # print(results_df)
    print(compare_df)
    print(perm_compare_df)
    
    write.csv(compare_df, paste(home_dir,"results","Auditory_delay_full_org_",feature,"_",align_to_onset,"aln.csv",sep=''), row.names = FALSE)
    write.csv(perm_compare_df, paste(home_dir,"results","Auditory_delay_full_perm",feature,"_",align_to_onset,"3aln.csv",sep=''), row.names = FALSE)
  }
}