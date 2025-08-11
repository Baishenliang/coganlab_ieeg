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
n_perm <-500
set.seed(42)

#%% Load files
file_path <- "D:/bsliang_Coganlabcode/coganlab_ieeg/projects/PCA_LDA/Auditory_Delay_long.csv"
long_data <- read.csv(file_path)
long_data$time <- as.numeric(long_data$time)
time_points <- unique(long_data$time)
# Add pho1
long_data <- long_data %>% mutate(pho1 = substr(stim, 1, 1))

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
  lme_model <- lmer(value ~ pho1 + (1 | subject) + (1 | electrode) + (1 | stim),
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
      fea_perm <- sample(current_data$pho1),
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

write.csv(compare_df, "Aud_delay_org.csv", row.names = FALSE)