library(zoo)
library(forecast)
library(Rwave)
library(wmtsa)

PLOT_DIR = '/Users/tom/PycharmProjects/nyu-research/plots'

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

clust <- function(dist){
  plot(agnes(dist, diss=TRUE), which.plots=2)
  med = pam(dist, 2, diss=TRUE)
  sil = silhouette(med)
  print(med$cluster)
  plot(sil)
  return(med)
}

save_png <- function(fn_plot, out_loc){
  # fn_plot: function with plotting side effect
  # out_loc: location of png
  png(filename=out_loc)
  fn_plot()
  dev.off()
}

june_ts <- function(){
  june_cas = ts(preprocessing('E:/summary/june_casual.csv', 'E:/summary/june_total.csv')$series[33:(672+32)], frequency=24)
  june_tot = ts(preprocessing('E:/summary/june_casual.csv', 'E:/summary/june_total.csv')$total[33:(672+32)], frequency=24)
  return (june_cas/june_tot)
}

june_w <- function(){
  june_cas = ts(preprocessing('E:/summary/june_casual.csv', 'E:/summary/june_total.csv')$series[49:(672+48)], frequency=24)
  june_tot = ts(preprocessing('E:/summary/june_casual.csv', 'E:/summary/june_total.csv')$total[49:(672+48)], frequency=24)
  return (june_cas/june_tot)
}

sept_ts <- function(){
  sept_cas = ts(preprocessing('E:/summary/sept_casual.csv', 'E:/summary/sept_total.csv')$series[9:680], frequency=24)
  sept_tot = ts(preprocessing('E:/summary/sept_casual.csv', 'E:/summary/sept_total.csv')$total[9:680], frequency=24)
  return(sept_cas/sept_tot)
}

# difference by week, log
stationary <- function(ts){
  return (diff(log(ts),  168))
}

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

split_by_day <- function(ts_month){
  # ts_month: vector-like of hourly observations over month
  # returns matrix where cols are "days" and rows are hours
  days = data.frame(matrix(nrow=24, ncol=1))
  for (i in 1:(length(ts_month)/24)){
    days[[i]] = ts_month[((i-1)*24+1):(i*24)]
  }
  return(days)
}

#clustering stuff
phase_dist <- function(freq){
  # gives dft phase distance function at a frequency
  return (function(ts1, ts2){
    # phase of freq from fft
    f1 = fft(ts1)[freq+1]
    f2 = fft(ts2)[freq+1]
    return (abs(Arg(f1/f2))/(2*pi))
  })
  
}

normalize <- function(ts){
  return(ts/norm(ts, type='2'))
}

dissimilarity <- function(m_ts, f){
  # calc dissimilarity matrix of list of time series given a distance function
  n = ncol(m_ts)
  mdiss = matrix(nrow=n, ncol=n)
  for (i in 1:n){
    for (j in 1:n){
      mdiss[i, j] = f(m_ts[,i], m_ts[,j])
    }
  }
  return (mdiss)       
}

day_of_week <- function(ts){
  # ts: time series by hour
  # returns 3d array
  nweeks = length(ts)/168
  weekdays = array(NA, dim=c(7, 24, nweeks))
  for (weekday in 1:7){
    days=matrix(nrow=24, ncol=nweeks)
    for (week in 1:nweeks){
      first_hour = 1+(week-1)*168+(weekday-1)*24
      weekdays[weekday, , week] =  window(ts, start=first_hour, end=(first_hour+23))
    }
    
  }
  return (weekdays)
  
}
