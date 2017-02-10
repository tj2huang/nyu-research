preprocessing <- function(ts_csv, total_csv, thres=5000){
  
  # removes rows with low observations and locfs them
  # returns time series and total time series, as a list
  
  
  ts = read.csv(ts_csv)
  total <- read.csv(total_csv)
  df = time_index(data.frame(ts))
  df_total = time_index(data.frame(total))
  
  missing = df_total[,'tweets']<thres
  print(which(missing))
  df_total[missing, 'tweets'] <- NA
  df[missing,] <-NA
  df_total = na.locf(df_total[,'tweets'])
  df_filled = (na.locf(df[, ncol(df)]))
  
  return(list(series=df_filled, total=df_total))
}

june_ts <- function(){
  june = preprocessing('twitter_tsa/data/june_casual.csv', 'twitter_tsa/data/june/june_total.csv')
  june_cas = ts(june$series[33:(672+32)], frequency=24)
  june_tot = ts(june$total[33:(672+32)], frequency=24)
  return (june_cas/june_tot)
}

