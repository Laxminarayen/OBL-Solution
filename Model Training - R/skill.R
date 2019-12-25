set.seed(15)
library(data.table)
library(tidyverse)
library(caret)
library(lubridate)
library(magrittr)
library(dplyr)
library(xgboost)
library(onehot)
setwd("C:\\Users\\rr221764\\Documents\\Skillenza DS")
tr <- read_csv('TRAINING.csv')
te <- read_csv("TEST.csv")
N <- 1:nrow(tr)
y <- tr$Value
tr_te <- tr %>% select(-Value) %>% bind_rows(te)
tr_te$same_city <- ifelse(tr_te$`Customer City`==tr_te$`Ship-to City`,1,0)
summary(tr_te$same_city)
rm(tr,te); gc()
setDT(tr_te)
#Start Date
invoice <- data.frame(data.table::tstrsplit(tr_te$`Invoice No.`,"/"),stringsAsFactors = FALSE)
names(invoice) <- c("charcode","codeloc","invoicenum","misc")
invoice$codeloc <- as.numeric(invoice$codeloc)
invoice$invoicenum <- as.numeric(invoice$invoicenum)
invoice <- invoice[,-4]
invoice$charcode <- invoice$charcode %>% factor() #%>% fct_lump(prop = 0.01)
tr_te %<>% select(-`Invoice No.`)%<>% bind_cols(invoice)
rm(invoice); gc()
item <- data.frame(data.table::tstrsplit(tr_te$`ERP Size`,"X"),stringsAsFactors = FALSE)
names(item) <- c("height",'width')
item$height <- as.numeric(item$height)
item$width <- as.numeric(item$width)
item$area <- item$height*item$width
tr_te %<>% select(-`ERP Size`)%<>% bind_cols(item)
rm(item); gc()
tr_te$`Re Territory` <- factor(tr_te$`Re Territory`) %>% fct_lump(p=0.01)
tr_te$`Category 2` <- tr_te$`Category 2`%>% factor()
tr_te$State <- factor(tr_te$State) %>% fct_lump(p=0.01)
summary(factor(tr_te$State))
library(textfeatures)
toDate <- function(year, month, day) {
  ISOdate(year, month, day)
}
date <- data.frame(data.table::tstrsplit(tr_te$Date,"/"),stringsAsFactors = FALSE)
names(date) <- c("day",'month','year')
date$date <- toDate(date$year,date$month,date$day)
date <- date %>%mutate(monthinterest = month(date), #%>% factor(),
  weekinterest = week(date),# %>% factor(),
  dayinterest = day(date),# %>% factor(),
  weekdayinterest = weekdays(date) %>% factor())
date <- date[,-c(1:4)]
tr_te <- tr_te %>% bind_cols(date)
tr_te %<>% select(-Date)
rm(date); gc()
summary(factor(tr_te$`Item Code`))
tr_te$Month <- tr_te$Month %>% factor()
library(stringr)
words <- c(substring(tr_te$`Item Description`, 10))
regexp <- "[[:digit:]]+"
words <- str_extract(words, regexp) %>% as.numeric()
textdf <- textfeatures(tr_te$`Item Description`,word_dims = 50)
tr_te <- tr_te %>% bind_cols(textdf)
####
library(text2vec)
prep_fun = tolower
tok_fun = word_tokenizer
tr_te$`Item Description` <- stringr::str_replace_all(tr_te$`Item Description`,"[^[:alpha:]]", " ")
tr_te$`Item Description` <- stringr::str_replace_all(tr_te$`Item Description`,"\\s+", " ")
it_train = itoken(tr_te$`Item Description`, 
                  preprocessor = prep_fun, 
                  tokenizer = tok_fun, 
                  ids = 1:nrow(tr_te), 
                  progressbar = FALSE)
vocab = create_vocabulary(it_train,ngram = c(1L, 3L))
vocab = prune_vocabulary(vocab, term_count_min = 200, 
                         doc_proportion_max = 0.5)
vectorizer = vocab_vectorizer(vocab)
dtm_train = create_dtm(it_train, vectorizer)
dtm_train <- data.matrix(dtm_train)
dtm_train <- data.frame(dtm_train)

tr_te %<>% select(-`Item Description`)
tr_te %<>% bind_cols(dtm_train)
rm(textdf,dtm_train); gc()


####
tr_te$`Customer Name & City` <- stringr::str_replace_all(tr_te$`Customer Name & City`,"[^[:alpha:]]", " ")
tr_te$`Customer Name & City` <- stringr::str_replace_all(tr_te$`Customer Name & City`,"\\s+", " ")
it_train = itoken(tr_te$`Customer Name & City`, 
                  preprocessor = prep_fun, 
                  tokenizer = tok_fun, 
                  ids = 1:nrow(tr_te), 
                  progressbar = FALSE)
vocab = create_vocabulary(it_train,ngram = c(1L, 3L))
vocab = prune_vocabulary(vocab, term_count_min = 200, 
                         doc_proportion_max = 0.5)
vectorizer = vocab_vectorizer(vocab)
dtm_train = create_dtm(it_train, vectorizer)
dtm_train <- data.matrix(dtm_train)
dtm_train <- data.frame(dtm_train)

tr_te %<>% select(-`Customer Name & City`)
tr_te %<>% bind_cols(dtm_train)
tr_te$n_pcs <- words
rm(dtm_train); gc()
tr_te$`Wall / Floor` <- factor(tr_te$`Wall / Floor`)
tr_te$`Tile Body` <- factor(tr_te$`Tile Body`)
tr_te$`Item Classification` <- tr_te$`Item Classification` %>% factor() %>% fct_lump(p=0.01)
tr_te$`Item Cat. Code` <- factor(tr_te$`Item Cat. Code`)
tr_te$`Quality Code` <- factor(tr_te$`Quality Code`)
tr_te$`Customer City` <- tr_te$`Customer City` %>% factor() %>% fct_lump(p=0.01)
tr_te$`Customer Type` <- tr_te$`Customer Type` %>% factor()
tr_te %<>% select(-`Customer Code`)
tr_te %<>% select(-`Quality Code`)
tr_te$`Sales Type` <- factor(tr_te$`Sales Type`)
summary(factor(tr_te$Category))
tr_te %<>% select(-invoicenum)
tr_te %<>% select(-Month_1)
tr_te %<>% select(-FY)
tr_te %<>% select(-Month_1)
tr_te$`Review Zone` <- tr_te$`Review Zone` %>% factor() %>% fct_lump(p=0.01)
tr_te$`Ship-to City` <- tr_te$`Ship-to City` %>% factor() %>% fct_lump(p=0.01)
tr_te$Category %<>% factor()
tr_te %<>% select(-`Item Code`)
tr_te$sqftperbox <- (tr_te$area/1e6)*tr_te$n_pcs
summary(tr_te$costpertile)
tr_te$costpertile <- tr_te$`MRP /BOX`/tr_te$n_pcs 
library(onehot)
onehotmod <- onehot(tr_te,max_levels = 50)
tr_te <- data.frame(predict(onehotmod,tr_te))


tr <- tr_te[N,]
te <- tr_te[-N,]
library(xgboost)
rm(tr_te); gc()
#rm(test); gc()
library(caret)
nrounds = 6
set.seed(1234)
tri <- createDataPartition(y,p=0.05,list = FALSE) %>% c()
tefinal = data.matrix(te)
tefinal <- xgb.DMatrix(data= tefinal)
rm(te); gc()
train <- tr
rm(tr); gc()
target <- y
RMSLE.xgb = function (preds, dtrain,th_err=1.5) {
  obs <- xgboost::getinfo(dtrain, "label")
  if ( sum(preds<0) >0 ) {
    preds = ifelse(preds >=0 , preds , th_err)
  }
  rmsle = sqrt(    sum( (log(preds+1) - log(obs+1))^2 )   /length(preds))
  return(list(metric = "RMSLE", value = rmsle))
}
library(ModelMetrics)
dtrain<- xgb.DMatrix(data= as.matrix(train[-tri,]), 
                     label= target[-tri])
#weight = w[dev])
dvalid <- xgb.DMatrix(data= as.matrix(train[tri,]) , 
                      label= target[tri])
rm(train); gc()
valids <- list(val = dvalid)
#### parameters are far from being optimal ####  
param = list(objective = "reg:linear", 
             eval_metric = 'rmse',
             max_depth = 12,
             eta = 0.1, 
             gamma = 4,min_child_weight=100,
             colsample_bytree = 0.7,
             colsample_bylevel = 0.7,
             lambda = 1, 
             alpha = 0,
             booster = "gbtree",
             silent = 0
) 
model<- xgb.train(data = dtrain,
                  params= param, 
                  nrounds = 2000, 
                  verbose = T, 
                  list(val1=dtrain , val2 = dvalid) ,       
                  early_stopping_rounds = 50 , 
                  print_every_n = 10,
                  maximize = F
)
pred_te = predict(model,tefinal)

cols <- colnames(dtrain)
imp <- xgb.importance(cols, model=model)
#write_csv(imp,"imp.csv")
xgb.importance(cols, model=model)%>% 
  xgb.plot.importance(top_n = 30)
subm <- read_csv("sample.csv")
subm$value <- pred_te
write.csv(subm,"subm.csv",row.names = FALSE)
