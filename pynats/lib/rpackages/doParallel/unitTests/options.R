test.preschedule <- function() {
  x <- list(1:3, 1:9, 1:19)
  cs <- 1:20
  dpn <- getDoParName()

  for (chunkSize in cs) {
    ## preschedule is TRUE for MC by default and 
    ## FALSE for SNOW, so we test by setting them otherwise
    if (identical(dpn, "doParallelMC")) {
      opts <- list(preschedule=FALSE)
    } else {
      opts <- list(preschedule=TRUE)
    }
    for (y in x) {
      if (identical(dpn, "doParallelMC")) {
        actual <- foreach(i=y, .options.multicore=opts) %dopar% i
      }
      else {
        actual <- foreach(i=y, .options.snow=opts) %dopar% i
      }
      checkEquals(actual, as.list(y))
      if (identical(dpn, "doParallelMC")) {
        actual <- foreach(i=y, .combine="c", .options.multicore=opts) %dopar% i
      }
      else {
        actual <- foreach(i=y, .combine="c", .options.snow=opts) %dopar% i
      }
      checkEquals(actual, y)
    }
  }
}

test.attach <- function() {
    if (identical(getDoParName(), "doParallelMC")) {
        return(TRUE)
    } else {
        myFun <- function(x){
            myFun1(x+1)
           }
        myFun1 <- function(x){
            2*x
         }
        testFun <- function(){
                inRes1 <- checkTrue("exportEnv" %in% search())
                if (!inRes1) {
                    stop("Attaching exportEnv failed")
                }
                inRes2 <- checkTrue(exists("myFun1", where=2))
                if (!inRes1) {
                    stop("myFun1 not found in exportEnv")
                }
                myFun(1)
               }
        res <- suppressWarnings(foreach(i=1:4, .combine="c", .packages="RUnit", 
                   .export="myFun1", .options.snow=list(attachExportEnv=TRUE)) %dopar%  testFun())

        checkEquals(res, c(4,4, 4, 4))
    }
}

pkgname.test.stress <- function() {
    if (!require(caret, quietly=TRUE)) {
        return(TRUE)
    } else {
        library(mlbench)
        data(BostonHousing)

        lmFit <- train(medv ~ . + rm:lstat,
                       data = BostonHousing, 
                       "lm")

        library(rpart)
        rpartFit <- train(medv ~ .,
                          data = BostonHousing,
                          "rpart",
                          tuneLength = 9)
    }
}

"test.pkgname.test.stress" <- function()
{
    res <- try(pkgname.test.stress())
    checkTrue(!is(res, "try-error"), msg="pkgname stress test failed")    
}