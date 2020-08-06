### R code from vignette source 'gettingstartedParallel.Rnw'

###################################################
### code chunk number 1: loadLibs
###################################################
library(doParallel)
cl <- makeCluster(2)
registerDoParallel(cl)
foreach(i=1:3) %dopar% sqrt(i)


###################################################
### code chunk number 2: gettingstartedParallel.Rnw:149-150
###################################################
stopCluster(cl)


###################################################
### code chunk number 3: gettingstartedParallel.Rnw:193-196
###################################################
library(doParallel)
cl <- makeCluster(2)
registerDoParallel(cl)


###################################################
### code chunk number 4: bootpar
###################################################
x <- iris[which(iris[,5] != "setosa"), c(1,5)]
trials <- 10000

ptime <- system.time({
  r <- foreach(icount(trials), .combine=cbind) %dopar% {
    ind <- sample(100, 100, replace=TRUE)
    result1 <- glm(x[ind,2]~x[ind,1], family=binomial(logit))
    coefficients(result1)
  }
})[3]
ptime


###################################################
### code chunk number 5: bootseq
###################################################
stime <- system.time({
  r <- foreach(icount(trials), .combine=cbind) %do% {
    ind <- sample(100, 100, replace=TRUE)
    result1 <- glm(x[ind,2]~x[ind,1], family=binomial(logit))
    coefficients(result1)
  }
})[3]
stime


###################################################
### code chunk number 6: getDoParWorkers
###################################################
getDoParWorkers()


###################################################
### code chunk number 7: getDoParName
###################################################
getDoParName()
getDoParVersion()


###################################################
### code chunk number 8: gettingstartedParallel.Rnw:274-275
###################################################
stopCluster(cl)


