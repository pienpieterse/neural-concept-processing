library(lme4)
library(lmerTest)
library(dplyr)
library(tidyr)
library(ggplot2)
library(emmeans)
library(multcomp)
library(coin)
library(stringr)
library(patchwork)
library(ggsignif)

df <- read.csv("results/results.csv", row.names=NULL)

df <- df %>%
  separate(
    participant_group.participant.roi.roi_type.model.index.value.semantic_model.semantic_value,
    into = c("group", "participant", "roi", "roi_type", "model", "index", "value",
             "semantic_model", "semantic_value"),
    sep = ";",
    convert = TRUE
  )

df$roi <- as.factor(df$roi)
df$roi_type <- as.factor(df$roi_type)
df$group <- as.factor(df$group)
df$model <- as.factor(df$model)
df$participant <- as.factor(df$participant)
df$value <- as.numeric(as.character(df$value))

head(df)

#QUESTION 1: 
df1 <- df %>% 
  filter(grepl("binder_concreteness|binder_abstractness", model))

head(df1)


model1 <- lmer(value ~ group * model * roi_type + (1 | participant) + (1 | roi), data = df1)
summary(model1)  

emm <- emmeans(model1, ~ roi_type * model)
emm

pairs_emm <- contrast(emm, method = "pairwise", adjust = "fdr")
pairs_emm

cld_emm <- cld(emm,
               Letters = letters,
               adjust = "fdr")
cld_emm

emm_df <- as.data.frame(cld_emm)

p_raw <- ggplot(df1, aes(x = roi_type, y = value, fill = model)) +
  geom_violin(alpha = 0.4, position = position_dodge(width = 0.8)) +
  geom_boxplot(width = 0.15, outlier.shape = NA,
               position = position_dodge(width = 0.8)) +
  theme_bw(base_size = 14) +
  labs(
    title = "Raw Neural Response Distributions",
    x = "ROI Type",
    y = "Neural Response (value)"
  )

p_pred <- ggplot(emm_df,
                 aes(x = roi_type,
                     y = emmean,
                     color = model,
                     group = model)) +
  geom_point(size = 3,
             position = position_dodge(width = 0.4)) +
  geom_errorbar(aes(ymin = asymp.LCL, ymax = asymp.UCL),
                width = 0.15,
                position = position_dodge(width = 0.4)) +
  geom_text(aes(label = .group),
            position = position_dodge(width = 0.4),
            vjust = -1.5,
            size = 5) +
  theme_bw(base_size = 14) +
  labs(
    title = "Estimated Marginal Means (with significance letters)",
    x = "ROI Type",
    y = "Model-Estimated Neural Response"
  )


p_final <- p_raw | p_pred
p_final


#QUESTION 2

df2 <- df %>% 
  filter(!grepl("binder_concreteness|binder_abstractness", model))

head(df2)

#andrea suggestion 2
\begin{align*}
\text{fMRI-foundation model alignment at timepoint t} \sim\ & \text{participant group} + \text{roi type} + \text{foundation model} \times \text{concreteness at timepoint t} + \text{foundation model} \times \text{abstractness at timepoint t} + (1 \mid \text{participant})  + (1 \mid \text{run where timepoint t is found})\\
& + (1 \mid \text{roi})
\end{align*}

model2 <- lmer(value ~ group * model * roi_type + (1 | participant) + (1 | roi), data = df2)
summary(model1)
