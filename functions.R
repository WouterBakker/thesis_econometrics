country_pipeline = function(data_init, country_){
  
  data = data_init %>% filter(country == country_) %>% select(!country)
  
  ## We want to look at LR_POS: filter all rows where LR_POS is missing 
  data = data %>% filter(IMD3002_LR_CSES != 9)
  
  
  
  ## Remove like/dislike for parties (not relevant)
  data = data[,!grepl("IMD3008", colnames(data))] 
  data = data[,!grepl("IMD3007", colnames(data))]
  ## Remove political information (question on how well-read people are on politics, could be of interest later)
  data = data[,!grepl("IMD3015", colnames(data))]
  
  ## Set missing observations to NA
  data2 = sapply(data, function(x){
    last_item = table(x) %>% tail(1) # takes last value in table, is always the code for a missing value (e.g., 999)
    
    if(grepl("9", names(last_item))){
      NA_val = names(last_item)
      x[x == NA_val] = NA
    }
    return(x)
  })
  
  
  data2 = data.frame(data2)
  
  
  # NA removal
  
  ## We want to retain household income quintile
  data2 = data2 %>% filter(IMD2006 <= 5)
  
  ## Clean education (everything above 5 is not relevant)
  data2 = data2 %>% filter(IMD2003 <= 5)
  
  
  ## Remove cols with too many NAs
  data3 = data2[,colSums(is.na(data2)) / nrow(data2) < 0.1]
  
  
  ## Remove rows which contain NA
  data4.1 = na.omit(data3)
  
  ## Remove columns with a high mean (means NA values that were undetected)
  # Variable rescaling
  # colMeans(data4)
  data4 = data4.1[,colMeans(data4.1) < 100]
  data4$year = data4.1$election_year
  
  
  
  # Modifying dependent variable lr_pos: 1 for right 0 for left
  data4 = data4[which(data4$IMD3002_LR_CSES != 2),] # remove middle (left/middle/right)
  data4$lr_pos = ifelse(data4$IMD3002_LR_CSES == 3, 1, 0)
  
  # Removing columns
  ## Remove columns that could lead to problems (cointegration, spurious correlations)
  
  
  ## Remove columns with SD==0 (means cardinality=1)
  data4 = data4[,which(sapply(data4, sd) != 0)]
  
  
  data5 = 
    data4 %>% select(!c(IMD2001_1, IMD2001_2)) %>% #remove age categories, already have ohc'd generation (IMD2001_X, etc.)
    select(!IMD2012_2) %>% #number in household under 18 (number in household total present)  
    select(!IMD2019_1) #Union membership, obviously these are left-leaning 
  
  ## Remove columns that measure the same variable as lrpos
  data5 = data5 %>% select(!c(IMD3002_LR_CSES, IMD3002_LR_MARPOR, IMD3002_IF_CSES, IMD3100_LR_CSES, IMD3100_LR_MARPOR, IMD3100_IF_CSES, IMD3002_OUTGOV))
  
  
  datafin = data5
  
  return(datafin)
}



all_country_pipeline = function(data_init, country_ = NULL){
  
  # if(!is.null(country_)){
  #   data = data_init %>% filter(country == country_) %>% select(!country)
  # } else {
  #   data = data_init
  # }
  
  
  ## We want to look at LR_POS: filter all rows where LR_POS is missing 
  data = data %>% filter(IMD3002_LR_CSES != 9)
  
  
  
  ## Remove like/dislike for parties (not relevant)
  data = data[,!grepl("IMD3008", colnames(data))] 
  data = data[,!grepl("IMD3007", colnames(data))]
  ## Remove political information (question on how well-read people are on politics, could be of interest later)
  data = data[,!grepl("IMD3015", colnames(data))]
  
  ## Set missing observations to NA
  data2 = sapply(data, function(x){
    last_item = table(x) %>% tail(1) # takes last value in table, is always the code for a missing value (e.g., 999)
    if(length(last_item) != 0){
      if(grepl("9", names(last_item))){
        NA_val = names(last_item)
        x[x == NA_val] = NA
      }
    }
    return(x)
  })
  
  
  data2 = data.frame(data2)
  
  # Variable cleaning
  ## We want to retain household income quintile
  data3 = data2 %>% filter(IMD2006 <= 5)
  
  ## Clean education (everything above 5 is not relevant)
  data3 = data3 %>% filter(IMD2003 <= 5)
  
  # NA removal
  ## Remove cols with too many NAs
  data4 = data3[,colSums(is.na(data3)) / nrow(data3) < 0.1]
  
  ## Remove rows which contain NA
  data4 = na.omit(data4)
  
  ## Remove columns with a high mean (means NA values that were undetected)
  # Variable rescaling
  # data4 = data4.1[,colMeans(data4.1) < 100]
  # data4$year = data4.1$election_year
  # 
  

  
  # Modifying dependent variable lr_pos: 1 for right 0 for left
  data5 = data4[which(data4$IMD3002_LR_CSES != 2),] # remove middle (left/middle/right)
  data5$lr_pos = ifelse(data5$IMD3002_LR_CSES == 3, 1, 0)
  
  
  # ## Remove columns with SD==0 (means cardinality=1)
  # data4 = data4[,which(sapply(data4, sd) != 0)]
  
  # data6 =
  #   data5 %>% 
  #   ## Remove columns that measure the same variable as lrpos
  #   select(!c(IMD3002_LR_CSES, IMD3002_IF_CSES, IMD3100_LR_CSES, IMD3100_IF_CSES, IMD3002_OUTGOV)) %>%
  #   select(!c(IMD2001_1, IMD2001_2))  #remove age categories, already have ohc'd generation (IMD2001_X, etc.)
  # # select(!IMD2012_2) %>% #number in household under 18 (number in household total present)
  # # select(!IMD2019_1) #Union membership, obviously these are left-leaning
  # 
  # 
  datafin = data5
  
  return(datafin)
}







