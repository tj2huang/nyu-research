library(zoo)
library(forecast)
library(Rwave)
library(wmtsa)

time_index <- function(df){
  #indexes a dataframe by 30 days and 24 hours
  
  times = data.frame()
  for (d in 1:30){
    for (h in 0:23){
      times = rbind(times, c(d, h, 0))
    }
  }
  colnames(times) = c('day', 'hour', 'tweets')
  colnames(df) = c('day', 'hour', 'tweets')
  merged = merge(times, df, by=c('day', 'hour'), all='TRUE')
  merged[is.na(merged)] = 0
  merged['tweets'] = pmax(merged$tweets.x,merged$tweets.y)
  return (merged)
}


preprocessing <- function(ts_csv, total_csv){
  
  # removes rows with low observations and locfs them
  # returns time series and total time series, as a list
  
  
  ts = read.csv(ts_csv)
  total <- read.csv(total_csv)
  df = time_index(data.frame(ts))
  df_total = time_index(data.frame(total))
  
  missing = df_total[,'tweets']<5000
  print(which(missing))
  df_total[missing, 'tweets'] <- NA
  df[missing,] <-NA
  df_total = na.locf(df_total[,'tweets'])
  df_filled = (na.locf(df[, ncol(df)]))
  
  return(list(series=df_filled, total=df_total))
}

split_by_day <- function(ts_month){
  # ts_month: vector-like of hourly observations over month
  # returns matrix where cols are "days" and rows are hours
  days = data.frame(matrix(nrow=24, ncol=1))
  for (i in 1:(length(ts_month)/24)){
    days[[i]] = ts_month[((i-1)*24+1):(i*24)]
  }
  return(days)
}
