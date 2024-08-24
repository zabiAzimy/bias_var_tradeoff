# check if packages can be loaded, i.e. they are already installed 
library(FNN) # KNN regression 
library(ISLR2) 
library(boot) # 3D visualisation # Bootstrap functions

#==============================================================================#
# Bias-Variance Trade-Off: introductery example                                #
#==============================================================================#
rm(list = ls(all.names = TRUE))

# 00: packages -----------------------------------------------------------------
library(FNN)

# 01: data creation ------------------------------------------------------------

# true values ----
# with n = 50 equally spaced x values
n_x <- 50
x_grid <- seq(0, 2*pi,length = n_x)
y_true <- sin(x_grid)

# visualisation: data and function
plot(x_grid, y_true, type = "p", col = "black")
curve(sin, from = 0, to = 2*pi, add = TRUE, col = "grey")

# adding noise ----
# one simulation of an observed y  (true function plus noise)
sd_noise <- 0.2
set.seed(20241015)
y_train_1 <- y_true + rnorm(n_x, 0, sd_noise)

# visualisation: data and function
plot(x_grid, y_true, type = "p", col = "black", 
     ylim = range(c(y_true, y_train_1)),
     ylab = "y", xlab = "x")
curve(sin, from = 0, to = 2*pi, add = TRUE, col = "grey")
points(x_grid, y_train_1, col = "red", pch = 20)
legend("topright", legend = c("data generating process", 
                              "true values", 
                              "oberserved values"),
       col = c("grey", "black", "red"),
       lty = c(1, NA, NA), pch = c(NA, 1, 19))

# 02: Simple estimate - mean value ---------------------------------------------

# Simple estimate: average of y_train_1 for all x 
y_est_simple <- rep(mean(y_train_1), n_x) 

# visualisation
plot(x_grid, y_true, type = "p", col = "black", 
     ylim = range(c(y_true, y_train_1)),
     ylab = "y", xlab = "x")
curve(sin, from = 0, to = 2*pi, add = TRUE, col = "grey")
points(x_grid, y_train_1, col = "red", pch = 20)
points(x_grid, y_est_simple, pch = 3, col = "blue")
legend("topright", legend = c("data generating process", 
                              "true values", 
                              "oberserved values",
                              "simple (= mean) estimate"),
       col = c("grey", "black", "red", "blue"),
       lty = c(1, NA, NA, NA), pch = c(NA, 1, 19, 3))



# This has a high mean squared bias because it doesn't fit the shape
bias_sq_y_train_simple <- mean((y_true - y_est_simple)^2)
bias_sq_y_train_simple

# 02a: Simulation with M = 100 -------------------------------------------------
M <- 100
y_est_simple_sim <- y_train_sim <- matrix(NA, nrow = M, ncol = n_x)
for(m in 1:M){
  y_train_sim[m, ] <- y_true + rnorm(n_x, 0, sd_noise)
  y_est_simple_sim[m, ] <- rep(mean(y_train_sim[m, ]), n_x) 
}  

# Squared Bias and Variance for each x value

## calculate the variance and the squared bias
bias_sq_at_x_simple <- var_at_x_simple <- rep(NA, n_x)
for(x in 1:n_x){
  bias_sq_at_x_simple[x] <- mean(y_est_simple_sim[, x] - y_true[x])^2
  var_at_x_simple[x] <- var(y_est_simple_sim[, x])
} 

# Visualisation ----
plot(x_grid, y_true, type = "p", col = "black", 
     ylim = range(c(y_true, y_train_1)),
     ylab = "y", xlab = "x", main = "Bias-Variance Trade Off: simple model")
curve(sin, from = 0, to = 2*pi, add = TRUE, col = "grey")
points(x_grid, bias_sq_at_x_simple, pch = 20, col = "red")
points(x_grid, var_at_x_simple, pch = 12, col = "blue")
legend("bottomleft", legend = c("data generating process", 
                                "true values", 
                                "Squared Bias at x",
                                "Variance at x"),
       col = c("grey", "black", "red", "blue"),
       lty = c(1, NA, NA, NA), pch = c(NA, 1, 19, 12))

# Mean squared bias and mean variance
mean(bias_sq_at_x_simple)
mean(var_at_x_simple)

# Question: which component in this scenario dominates the MSE?

# 03: Complex model: KNN with n=1 ----------------------------------------------

# At the other exteme we use just the observed data (i.e. KNN with $n=1$).
y_est_complex <- y_train_1   # same result as KNN with n=1

# visualisation
plot(x_grid, y_true, type = "p", col = "black", 
     ylim = range(c(y_true, y_train_1)),
     ylab = "y", xlab = "x")
curve(sin, from = 0, to = 2*pi, add = TRUE, col = "grey")
points(x_grid, y_train_1, col = "red", pch = 20)
points(x_grid, y_est_complex, pch = 3, col = "blue")
legend("topright", legend = c("data generating process", 
                              "true values", 
                              "oberserved values",
                              "complex (1-NN) estimate"),
       col = c("grey", "black", "red", "blue"),
       lty = c(1, NA, NA, NA), pch = c(NA, 1, 19, 3))


# has a low squared bias because it overfits the data
bias_sq_y_train_complex <- mean((y_true - y_est_complex)^2)
bias_sq_y_train_complex

# 03a: Simulation with M = 10 --------------------------------------------------
y_est_complex_sim <- matrix(NA, nrow = M, ncol = n_x)
for(m in 1:M){
  y_est_complex_sim[m, ] <- y_train_sim[m, ]
}  

# Squared Bias and Variance for each x value

## calculate the variance and the squared bias
bias_sq_at_x_complex <- var_at_x_complex <- rep(NA, n_x)
for(x in 1:n_x){
  bias_sq_at_x_complex[x] <- mean(y_est_complex_sim[, x] - y_true[x])^2
  var_at_x_complex[x] <- var(y_est_complex_sim[, x])
} 

# Visualisation ----
plot(x_grid, y_true, type = "p", col = "black", 
     ylim = range(c(y_true, y_train_1)),
     ylab = "y", xlab = "x", main = "Bias-Variance Trade Off: complex model")
curve(sin, from = 0, to = 2*pi, add = TRUE, col = "grey")
points(x_grid, bias_sq_at_x_complex, pch = 20, col = "red")
points(x_grid, var_at_x_complex, pch = 12, col = "blue")
legend("bottomleft", legend = c("data generating process", 
                                "true values", 
                                "Squared Bias at x",
                                "Variance at x"),
       col = c("grey", "black", "red", "blue"),
       lty = c(1, NA, NA, NA), pch = c(NA, 1, 19, 12))

# Mean squared bias and mean variance
mean(bias_sq_at_x_complex)
mean(var_at_x_complex)

# 04a: Model-Tuning for KNN algorithm ------------------------------------------
# We want to see, what n = 1, ..., 20 for KNN works best

# We expand the existing to code using a 3-dim array instead of a matrix
k_max <- 20
y_est_k_KNN_sim <- array(NA, dim = c(k_max, M, n_x))

# Model estimation:
for(k in 1:k_max){
  for(m in 1:M){
    y_est_k_KNN_sim[k, m, ] <- knn.reg(x_grid, test = matrix(x_grid),
                                       y = y_train_sim[m, ], k = k)$pred
  }  
}

# Model evaluation: Squared Bias and Variance at each x value for each k
bias_sq_at_x_k_KNN <- var_at_x_k_KNN <- matrix(NA, nrow = k_max, ncol = n_x)

for(k in 1:k_max){
  for(x in 1:n_x){
    bias_sq_at_x_k_KNN[k, x] <- mean(y_est_k_KNN_sim[k, , x] - y_true[x])^2
    var_at_x_k_KNN[k, x] <- var(y_est_k_KNN_sim[k, , x])
  } 
}

# Summarizing the M simulations by looking at the row averages
bias_sq_k <- rowMeans(bias_sq_at_x_k_KNN)
var_k <- rowMeans(var_at_x_k_KNN)
mse_k <- bias_sq_k + var_k + sd_noise^2

# Visualisation
plot(1:k_max, mse_k, type = "b", col = "black", ylim = c(0, max(mse_k)),
     ylab = "mse and mse components", xlab = "k-NN parameter",
     las = 1, lwd = 2)
lines(rep(sd_noise^2, k_max), col = "grey", pch = 20, type = "b")
lines(bias_sq_k, col = "red", pch = 20, type = "b")
lines(var_k, col = "blue", pch = 20, type = "b")
abline(v = which.min(mse_k), lty = 2, col = "grey")
legend("topleft", 
       legend = c("MSE", "Squared Bias", "Variance", "Noise"),
       col = c("black", "red", "blue", "grey"),
       lty = 1, pch = 20)
mtext("lowest MSE", at = which.min(mse_k), side = 1, line = 2)


# with my simulations k=5 gives the minimum variance plus squared bias
# but k=6 or 7 are similar