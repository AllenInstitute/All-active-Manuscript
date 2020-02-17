library(tidyverse)
library(ggplot2)
library(moonBook)
library(webr)


# Setting working directory -----------------------------------------------

# this.dir <- dirname(parent.frame(2)$ofile)
# setwd(this.dir)


# Data paths --------------------------------------------------------------

project_path <- file.path(getwd(),'..','aggregated_data')
mouse_data_filename <- file.path(project_path, 'Mouse_class_data.csv')

# Reading data ------------------------------------------------------------

mouse_data <- read_csv(mouse_data_filename,col_types = cols(Cell_id = col_character()))
mouse_data <- mouse_data %>% filter(hof_index==0) %>% select(Cell_id, Dendrite_type, Cre_line)
mouse_data <- mouse_data %>% rename(Cell_type = Dendrite_type) 
mouse_data <- mouse_data %>% arrange(desc(Cell_type))

# Pie Donut plot ----------------------------------------------------------

PieDonut(mouse_data,aes(pies=Cell_type,donuts=Cre_line),
         r0 = 0.18,r1=.45,r2=.6,
         showRatioThreshold = .03,
         labelposition=1,
         donutLabelSize = 4)

