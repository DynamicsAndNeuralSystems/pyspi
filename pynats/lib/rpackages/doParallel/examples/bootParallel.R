suppressMessages(library(doParallel))
cl <- makePSOCKcluster(4)
registerDoParallel(cl)

cat(sprintf('doParallel %s\n', packageVersion('doParallel')))
junk <- matrix(0, 1000000, 8)
cat(sprintf('Size of extra junk data: %d bytes\n', object.size(junk)))

x <- iris[which(iris[,5] != "setosa"), c(1,5)]

trials <- 10000

ptime <- system.time({
  r <- foreach(icount(trials), .combine=cbind,
               .export='junk') %dopar% {
    ind <- sample(100, 100, replace=TRUE)
    result1 <- glm(x[ind,2]~x[ind,1], family=binomial(logit))
    coefficients(result1)
  }
})[3]
cat(sprintf('parallel foreach:                    %6.1f sec\n', ptime))

ptime2 <- system.time({
  snowopts <- list(preschedule=TRUE)
  r <- foreach(icount(trials), .combine=cbind,
               .export='junk', .options.snow=snowopts) %dopar% {
    ind <- sample(100, 100, replace=TRUE)
    result1 <- glm(x[ind,2]~x[ind,1], family=binomial(logit))
    coefficients(result1)
  }
})[3]
cat(sprintf('parallel foreach with prescheduling: %6.1f sec\n', ptime2))


ptime3 <- system.time({
  chunks <- getDoParWorkers()
  r <- foreach(n=idiv(trials, chunks=chunks), .combine=cbind,
               .export='junk') %dopar% {
    y <- lapply(seq_len(n), function(i) {
      ind <- sample(100, 100, replace=TRUE)
      result1 <- glm(x[ind,2]~x[ind,1], family=binomial(logit))
      coefficients(result1)
    })
    do.call('cbind', y)
  }
})[3]
cat(sprintf('chunked parallel foreach:            %6.1f sec\n', ptime3))

ptime4 <- system.time({
  mkworker <- function(x, junk) {
    force(x)
    force(junk)
    function(i) {
      ind <- sample(100, 100, replace=TRUE)
      result1 <- glm(x[ind,2]~x[ind,1], family=binomial(logit))
      coefficients(result1)
    }
  }
  y <- parLapply(cl, seq_len(trials), mkworker(x, junk))
  r <- do.call('cbind', y)
})[3]
cat(sprintf('parLapply:                           %6.1f sec\n', ptime4))

stime <- system.time({
  y <- lapply(seq_len(trials), function(i) {
    ind <- sample(100, 100, replace=TRUE)
    result1 <- glm(x[ind,2]~x[ind,1], family=binomial(logit))
    coefficients(result1)
  })
  r <- do.call('cbind', y)
})[3]
cat(sprintf('sequential lapply:                   %6.1f sec\n', stime))

stime2 <- system.time({
  r <- foreach(icount(trials), .combine=cbind) %do% {
    ind <- sample(100, 100, replace=TRUE)
    result1 <- glm(x[ind,2]~x[ind,1], family=binomial(logit))
    coefficients(result1)
  }
})[3]
cat(sprintf('sequential foreach:                  %6.1f sec\n', stime2))

stopCluster(cl)
