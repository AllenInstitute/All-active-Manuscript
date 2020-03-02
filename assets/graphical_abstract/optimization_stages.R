library(tidyverse)
library(formattable)


# Setting working directory -----------------------------------------------

this.dir <- dirname(parent.frame(2)$ofile)
setwd(this.dir)


# This function is taken from https://github.com/renkun-ken/formattable/issues/26

export_formattable <- function(f, file, width = "100%", height = NULL, 
                               background = "white", delay = 0.2)
{
  w <- as.htmlwidget(f, width = width, height = height)
  path <- html_print(w, background = background, viewer = NULL)
  url <- paste0("file:///", gsub("\\\\", "/", normalizePath(path)))
  webshot(url,
          file = file,
          selector = ".formattable_widget",
          delay = delay)
}



# Stage 0 -----------------------------------------------------------------


stage0_features <- tibble(Features = c('voltage_base, steady_state_voltage',
                                       'voltage_deflection',
                                       'decay_time_constant_after_stim'))

t_feat_stage0 <- formattable(stage0_features,
                             align =c("l"))
print(t_feat_stage0)
export_formattable(t_feat_stage0,"stage0_features.pdf")


stage0_params <- tibble(Parameters = c('cm, gpas', 
                                       'epas, Ra'))

t_params_stage0 <- formattable(stage0_params,
                              align =c("l"))
print(t_params_stage0)
export_formattable(t_params_stage0,"stage0_params.pdf")


# Stage 1 -----------------------------------------------------------------


stage1_features <- tibble(Features = c('voltage_base, steady_state_voltage',
                                       'voltage_deflection, sag_amplitude',
                                       'decay_time_constant_after_stim'))

t_feat_stage1 <- formattable(stage1_features,
                             align =c("l"))
print(t_feat_stage1)
export_formattable(t_feat_stage1,"stage1_features.pdf")


stage1_params <- tibble(Parameters = c('ḡIh'))

t_params_stage1 <- formattable(stage1_params,
                               align =c("l"))
print(t_params_stage1)
export_formattable(t_params_stage1,"stage1_params.pdf")



# Stage 2 -----------------------------------------------------------------


stage2_features <- tibble(Features = c('voltage_base, steady_state_voltage',
                                       'mean_frequency, Spikecount',
                                       'AP_amplitude, AP_width, AHP_depth',
                                       'adaptation_index2, ISI_CV',
                                       'depolarization_block, check_AISInitiation'))

t_feat_stage2 <- formattable(stage2_features,
                             align =c("l"))
print(t_feat_stage2)
export_formattable(t_feat_stage2,"stage2_features.pdf")


stage2_params <- tibble(Parameters = c('ḡNaT, ḡNaP, ḡNaV, ḡKT', 
                                       'ḡKP,  ḡKv3.1, ḡKv2like, ḡIm',
                                       'ḡImv2, ḡSK, ḡCaHVA, ḡCaLVA'))

t_params_stage2 <- formattable(stage2_params,
                             align =c("l"))
print(t_params_stage2)
export_formattable(t_params_stage2,"stage2_params.pdf")
