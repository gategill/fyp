h <- sort(table(all_ratings$item_id),decreasing=T)

hist(h,breaks = 30,
     main = "Frequency Distribution of Movies",
     freq =  FALSE,
     xlab = "Movies")
################################################################################
# USERKNN BEST K=20 at 0.698504 
# ITEM best K=15 at 0.659006

user.knn = KNN_Complete[KNN_Complete$algorithm == "UserKNN" , ]
user.knn.means = aggregate(user.knn$mae, list(user.knn$k), FUN=mean)$x
user.knn.means.times = aggregate(user.knn$time_elapsed_s, list(user.knn$k), FUN=mean)$x
user.knn.best.k = which.min(user.knn.means) * 5
user.knn.best.result = min(user.knn.means)
user.knn.best.results = user.knn[user.knn$k == user.knn.best.k, ]$mae

item.knn = KNN_Complete[KNN_Complete$algorithm == "ItemKNN" , ]
item.knn.means = aggregate(item.knn$mae, list(item.knn$k), FUN=mean)$x
item.knn.means.times = aggregate(item.knn$time_elapsed_s, list(item.knn$k), FUN=mean)$x
item.knn.best.k = which.min(item.knn.means) * 5
item.knn.best.result = min(item.knn.means)
item.knn.best.results = item.knn[item.knn$k == item.knn.best.k, ]$mae

((user.knn.best.result-item.knn.best.result)/user.knn.best.result)*100
t.test(user.knn.best.results, item.knn.best.results,  paired = TRUE)

user.knn.best.result
user.knn.best.k
item.knn.best.result
item.knn.best.k


x = c(5,10,15,20,25)
#par(mfrow = c(1,1))
par(mar = c(5, 4, 4, 4) + 0.3)              # Additional space for second y-axis

plot(x, user.knn.means,type = "l",
     main = "K vs MAE of User KNN and Item KNN",
     col= 1, pch=1, 
     xlab = "Neighbourhood Size",
     ylab="Mean MAE", lty=1,
     ylim = c(0.65,0.75))

  points(x, user.knn.means, col=1, cex = 2, lw = 2)
  lines(x, user.knn.means, col=1, lw = 3)


points(x, item.knn.means, col=2, pch=2, cex = 2, lw = 2)
lines(x, item.knn.means, col=2,lty=2, lw = 3)


par(new = TRUE)                             # Add new plot
plot(x, item.knn.means.times, pch = 17, col = 3,              # Create second plot without axes
     axes = FALSE, xlab = "", ylab = "")
points(x, user.knn.means.times, col=15, pch=5, cex = 2, lw = 2)
axis(side = 4, at = pretty(range(item.knn.means.times)))      # Add second axis
mtext("seconds", side = 4, line = 3)      
legend('topright', legend = c("User KNN", "Item KNN"), col = c(1:2), pch = c(1:2), lty = c(1:2))

hist( user.knn.means, breaks = 5)
shas = rbind(user.knn.means, item.knn.means)

require(plotrix)

mycol <- c("green ", "blue ")
mydata<- list(rnorm(5, 10,5),rnorm(5, 10, 8),rnorm(5, 15,5) )
multhist(shas, col= mycol, x = )
abline(a = 0, b = 0)
require(ggplot2)
hist(mydata[[1]], mydata[[2]], mydata[[3]], col = c(1,2,3), breaks = 100)




par(mar = c(5, 4, 4, 4) + 0.3)              # Additional space for second y-axis

barplot(ylim = c(0.60, 0.75),shas, beside = TRUE, names = c("5","10","15","20","25"), col = c(1,2),
        xlab = "Neigbourhood Size",
        ylab = "MAE",
        main = "K vs MAE and Time for User and Item KNN", xpd = FALSE)
abline(a = 0.6, b = 0)
abline(h = c(0.6, 0.62, 0.64, 0.66, 0.68, 0.70, 0.72, 0.74), lty = 2)

par(new = TRUE)                             # Add new plot
plot(x, item.knn.means.times, pch = 15, col = 6,              # Create second plot without axes
     axes = FALSE, xlab = "", ylab = "",cex = 2)
points(x, user.knn.means.times, col=8, pch=17, cex = 2, lw = 2)
axis(side = 4, at = pretty(range(item.knn.means.times)))      # Add second axis
lines(x, item.knn.means.times, col=6,lty=1, lw = 3)
lines(x, user.knn.means.times, col=8,lty=1, lw = 3)
mtext("Task Time (seconds)", side = 4, line = 3)   
legend('topright', legend = c("User MAE", "Item MAE"), border = "black", fill = 1:2)
legend(22.4,104, legend = c("User Time", "Item Time"), col = c(6,8),  pch = c(15,17), lty = 1)


plot(x, item.knn.means.times, pch = 15, col = 1, xlab = "Neighbourhood Size", ylab = "Task Time (seconds)",
     main = "K vs Task Time for User and Item KNN",cex = 2)
lines(x, item.knn.means.times, col = 1, lw = 2, lty = 1)

points(x, user.knn.means.times, col=2, pch=16, cex = 2)
lines(x, user.knn.means.times, col=2, lw = 3, lty = 2)

legend("topright", legend = c("User KNN", "Item KNN"), lty = 1:2, col = 1:2, pch = 15:16)


########################################################################################
# USERBOOTSTRAPPED, best = e1 with k=20 at 0.695216, slightly better than e=0

user.bootstrap = Bootstrap.Complete[Bootstrap.Complete$algorithm == "UserBootstrap" , ]
user.bootstrap.means.k = aggregate(user.bootstrap$mae, list(user.bootstrap$k), FUN=mean)$x
user.bootstrap.means.e= aggregate(user.bootstrap$mae, list(user.bootstrap$e), FUN=mean)$x
user.bootstrap.means.k_e = aggregate(user.bootstrap$mae, list(user.bootstrap$k, user.bootstrap$e), FUN=mean)
user.bootstrap.means.k_e.times = aggregate(user.bootstrap$time_elapsed_s, list(user.bootstrap$k, user.bootstrap$e), FUN=mean)
user.bootstrap.best.k = which.min(user.bootstrap.means.k) * 5
user.bootstrap.best.e = which.min(user.bootstrap.means.e) * 1
user.bootstrap.best.result = min(user.bootstrap.means.k, user.bootstrap.means.e)
user.bootstrap.best.results = user.bootstrap[user.bootstrap$k == user.bootstrap.best.k & user.bootstrap$e == user.bootstrap.best.e, ]$mae

item.bootstrap = Bootstrap.Complete[Bootstrap.Complete$algorithm == "ItemBootstrap" , ]
item.bootstrap.means.k = aggregate(item.bootstrap$mae, list(item.bootstrap$k), FUN=mean)$x
item.bootstrap.means.e= aggregate(item.bootstrap$mae, list(item.bootstrap$e), FUN=mean)$x
item.bootstrap.means.k_e = aggregate(item.bootstrap$mae, list(item.bootstrap$k, item.bootstrap$e), FUN=mean)
item.bootstrap.means.k_e.times = aggregate(item.bootstrap$time_elapsed_s, list(item.bootstrap$k, item.bootstrap$e), FUN=mean)
item.bootstrap.best.k = which.min(item.bootstrap.means.k) * 5
item.bootstrap.best.e = which.min(item.bootstrap.means.e) * 1
item.bootstrap.best.result = min(item.bootstrap.means.k, item.bootstrap.means.e)
item.bootstrap.best.results = item.bootstrap[item.bootstrap$k == item.bootstrap.best.k & item.bootstrap$e == item.bootstrap.best.e, ]$mae

((user.knn.best.result - user.bootstrap.best.result)/user.knn.best.result)*100
((item.knn.best.result - item.bootstrap.best.result)/item.knn.best.result)*100
t.test(user.bootstrap.best.results, user.knn.best.results,  paired = TRUE)
t.test(item.bootstrap.best.results, item.knn.best.results,  paired = TRUE)
user.bootstrap.best.result
user.bootstrap.best.k
user.bootstrap.best.e
item.bootstrap.best.result
item.bootstrap.best.k
item.bootstrap.best.e


user.bootstrap.means.k_e.all_1 = user.bootstrap.means.k_e[user.bootstrap.means.k_e$Group.2 == 1, ]$mae
user.bootstrap.means.k_e.all_2 = user.bootstrap.means.k_e[user.bootstrap.means.k_e$Group.2 == 3, ]$mae
user.bootstrap.means.k_e.all_3 = user.bootstrap.means.k_e[user.bootstrap.means.k_e$Group.2 == 3, ]$mae

item.bootstrap.means.k_e.times[1,]

x = c(5,10,15,20,25)
par(mfrow = c(1,1))
plot(x, user.knn.means, type = "l",
     main = "K vs MAE of User Bootstrap",
     col= 1, pch=1, 
     xlab = "Neighbourhood Size",
     ylab="Mean MAE", lty=1,
     ylim = c(0.69,0.75))

points(x, user.knn.means, col=1, pch=1,  cex = 2, lw = 2)
lines(x, user.knn.means, col=1, pch=1,  lw = 3)

points(x, user.bootstrap.means.k_e.all_1, col=2, pch=2,  cex = 2, lw = 2)
lines(x, user.bootstrap.means.k_e.all_1, col=2,lty=2, lw= 3)

points(x, user.bootstrap.means.k_e.all_2, col=3, pch=3, cex = 2, lw = 2)
lines(x, user.bootstrap.means.k_e.all_2, col=3,lty=3,  lw= 3)

points(x, user.bootstrap.means.k_e.all_3, col=4, pch=4,  cex = 2, lw = 2)
lines(x,user.bootstrap.means.k_e.all_3, col=4,lty=4,  lw= 3)


legend('topright', legend = c( "Baseline", "Rounds=1", "Rounds=2", "Rounds=3"), col = c(1:4),
       pch = c(1:4), lty = c(1:4))



plot(x, user.knn.means.times, pch = 15, col = 1, xlab = "Neighbourhood Size", ylab = "Task Time (seconds)",
     main = "K vs Task Time for User KNN and Bootstrap",cex = 2, ylim = c(0,250))
lines(x, user.knn.means.times, col = 1, lw = 2, lty = 1)

points(x, user.bootstrap.means.k_e.times[1:5,]$x, col=2, pch=16, cex = 2)
lines(x, user.bootstrap.means.k_e.times[1:5,]$x, col=2, lw = 3, lty = 2)

points(x, user.bootstrap.means.k_e.times[6:10,]$x, col=3, pch=17, cex = 2)
lines(x, user.bootstrap.means.k_e.times[6:10,]$x, col=3, lw = 3, lty = 3)

points(x, user.bootstrap.means.k_e.times[11:15,]$x, col=4, pch=18, cex = 2)
lines(x, user.bootstrap.means.k_e.times[11:15,]$x, col=4, lw = 3, lty = 4)

legend("topright", legend = c("Baseline", "Iterations=1", "Iterations=2", "Iterations=3"), lty = 1:4, col = 1:4, pch = 15:18)

###############################################################################
# item bootstrap knn, min k= 15  at 0.654952, e = 1, slightly better than e=0
((0.659006-0.654952)/0.659006)*100
((y_prie-y)/y_prie)*100

t.test(k15, a, paired=TRUE)

par(mfrow = c(1,1))
t = numeric(5)
for (i in 1:5){
  a = ItemBootstrap_Complete[ItemBootstrap_Complete$k == 5*i, ]
  a = a[a$e == 1,]
  a = mean(a$mae)
  t[i] = a
}
a = ItemBootstrap_Complete[ItemBootstrap_Complete$k == 15, ]
a = a[a$e == 1,]
y = mean(a$mae)

y
y_prie
h = numeric(5)
for (i in 1:5){
  a = ItemBootstrap_Complete[ItemBootstrap_Complete$k == 5*i, ]
  a = a[a$e == 2,]
  a = mean(a$mae)
  h[i] = a
}
b = numeric(5)
for (i in 1:5){
  a = ItemBootstrap_Complete[ItemBootstrap_Complete$k == 5*i, ]
  a = a[a$e == 3,]
  a = mean(a$mae)
  b[i] = a
}


plot(x, item_knn, type='l',
     main = "K vs MAE of Item Bootstrap",
     col= 1, pch=1, 
     xlab = "Neighbourhood Size",
     ylab="Mean MAE", lty=1,
     ylim = c(0.65,0.705))

points(x, y.item_knn, col=1, pch=1,  cex = 2, lw = 2)
lines(x, y.item_knn, col=1, pch=1, lw = 3)
      
points(x, t, col=2, pch=2,  cex = 2, lw = 2)
lines(x, t, col=2,lty=2, lw= 3)

points(x, h, col=3, pch=3, cex = 2, lw = 2)
lines(x,h, col=3,lty=3,  lw= 3)

points(x, b, col=4, pch=4,  cex = 2, lw = 2)
lines(x,b, col=4,lty=4,  lw= 3)


legend('topright', legend = c( "Baseline", "Rounds=1", "Rounds=2", "Rounds=3"), col = c(1:4),
       pch = c(1:4), lty = c(1:4))


which.min(t) * 5; min(t)
which.min(h) * 5; min(h)
which.min(b) * 5; min(b)


###############
# user vs item bootstrapped

par(mfrow = c(1,2))
t = numeric(5)
for (i in 1:5){
  a = UserBootstrap_Complete[UserBootstrap_Complete$k == 5*i, ]
  a = a[a$e == 1,]
  a = mean(a$mae)
  t[i] = a
}
a = UserBootstrap_Complete[UserBootstrap_Complete$k == 20, ]
a = a[a$e == 1,]
y_prie = mean(a$mae)

h = numeric(5)
for (i in 1:5){
  a = UserBootstrap_Complete[UserBootstrap_Complete$k == 5*i, ]
  a = a[a$e == 2,]
  a = mean(a$mae)
  h[i] = a
}
b = numeric(5)
for (i in 1:5){
  a = UserBootstrap_Complete[UserBootstrap_Complete$k == 5*i, ]
  a = a[a$e == 3,]
  a = mean(a$mae)
  b[i] = a
}


plot(x, y.user_knn, type='l',
     main = "K vs MAE of User Bootstrap",
     col= 1, pch=1, 
     xlab = "Neighbourhood Size",
     ylab="Mean MAE", lty=1,
     ylim = c(0.65,0.75))

points(x, y.user_knn, col=1, pch=1,  cex = 2, lw = 2)
lines(x, y.user_knn, col=1, pch=1,  lw = 3)

points(x, t, col=2, pch=2,  cex = 2, lw = 2)
lines(x, t, col=2,lty=2, lw= 3)

points(x, h, col=3, pch=3, cex = 2, lw = 2)
lines(x,h, col=3,lty=3,  lw= 3)

points(x, b, col=4, pch=4,  cex = 2, lw = 2)
lines(x,b, col=4,lty=4,  lw= 3)


legend('topright', legend = c( "Baseline", "Rounds=1", "Rounds=2", "Rounds=3"), col = c(1:4),
       pch = c(1:4), lty = c(1:4))


t = numeric(5)
for (i in 1:5){
  a = ItemBootstrap_Complete[ItemBootstrap_Complete$k == 5*i, ]
  a = a[a$e == 1,]
  a = mean(a$mae)
  t[i] = a
}
a = ItemBootstrap_Complete[ItemBootstrap_Complete$k == 15, ]
a = a[a$e == 1,]
y = mean(a$mae)

y
y_prie
h = numeric(5)
for (i in 1:5){
  a = ItemBootstrap_Complete[ItemBootstrap_Complete$k == 5*i, ]
  a = a[a$e == 2,]
  a = mean(a$mae)
  h[i] = a
}
b = numeric(5)
for (i in 1:5){
  a = ItemBootstrap_Complete[ItemBootstrap_Complete$k == 5*i, ]
  a = a[a$e == 3,]
  a = mean(a$mae)
  b[i] = a
}


plot(x, y.item_knn, type='l',
     main = "K vs MAE of Item Bootstrap",
     col= 1, pch=1, 
     xlab = "Neighbourhood Size",
     ylab="Mean MAE", lty=1,
     ylim = c(0.65,0.75))

      
      points(x, y.item_knn, col=1, pch=1,  cex = 2, lw = 2)
      lines(x, y.item_knn, col=1, pch=1,  lw = 3)
      
      points(x, t, col=2, pch=2,  cex = 2, lw = 2)
      lines(x, t, col=2,lty=2, lw= 3)
      
      points(x, h, col=3, pch=3, cex = 2, lw = 2)
      lines(x,h, col=3,lty=3,  lw= 3)
      
      points(x, b, col=4, pch=4,  cex = 2, lw = 2)
      lines(x,b, col=4,lty=4,  lw= 3)

legend('topright', legend = c( "Baseline", "Rounds=1", "Rounds=2", "Rounds=3"), col = c(1:4),
       pch = c(1:4), lty = c(1:4))


#################################################################################

# CoRec

user.corec = CoRec_Complete[ , c("algorithm", "k", "mae_u", "fold_num", "a")]
user.corec.means.k = aggregate(user.corec$mae, list(user.corec$k), FUN=mean)$x
user.corec.means.a= aggregate(user.corec$mae, list(user.corec$a), FUN=mean)$x
user.corec.means.k_a = aggregate(user.corec$mae, list(user.corec$k, user.corec$a), FUN=mean)
user.corec.best.k = 15 # which.min(user.corec.means.k) * 5
user.corec.best.a = 50#which.min(user.corec.means.a)
user.corec.best.mean = min(user.corec.means.k, user.corec.means.k_a$x)
user.corec.best.result = min(user.corec.means.k, user.corec.means.a)
user.corec.best.results = user.corec[user.corec$k == user.corec.best.k & user.corec$a == user.corec.best.a, ]$mae

item.corec = CoRec_Complete[ , c("algorithm", "k", "mae_i", "fold_num", "a")]
item.corec.means.k = aggregate(item.corec$mae, list(item.corec$k), FUN=mean)$x
item.corec.means.a = aggregate(item.corec$mae, list(item.corec$a), FUN=mean)$x
item.corec.means.k_a = aggregate(item.corec$mae, list(item.corec$k, item.corec$a), FUN=mean)
# means, not individual
# have a best mean

which.min(item.corec.means.k_a$x)
item.corec.best.mean = min(item.corec.means.k_a$x)
item.corec.best.k = 15 # which.min(item.corec.means.k) * 5
item.corec.best.a = 50 # which.min(item.corec.means.a) * 10
item.corec.best.result = min(item.corec.means.k, item.corec.means.a) # best results = best mean
item.corec.best.results = item.corec[item.corec$k == item.corec.best.k & item.corec$a == item.corec.best.a, ]$mae

((user.knn.best.result - user.corec.best.mean)/user.knn.best.result)*100
((item.knn.best.result - item.corec.best.mean)/item.knn.best.result)*100
t.test(user.corec.best.results, user.knn.best.results,  paired = TRUE)
t.test(item.corec.best.results, item.knn.best.results,  paired = TRUE)

user.corec.best.mean
user.corec.best.k
user.corec.best.a

item.corec.best.mean
item.corec.best.k
item.corec.best.a

# user min = k=15 @ 0.691022
# item min = k=15 @ 0.645542 

# USER COREC
((0.698504-0.691022)/0.698504)*100
t.test(k20, r, paired=TRUE)


((0.659006-0.645542)/0.659006)*100
t.test(k15, s, paired=TRUE)

((0.691022-0.645542)/0.691022)*100

par(mfrow = c(1,1))
t = numeric(5)
for (i in 1:5){
  a = CoRec_Complete[CoRec_Complete$k == 5*i, ]
  a = a[a$a == 10,]
  a = mean(a$mae_u)
  t[i] = a
}
t

h = numeric(5)
for (i in 1:5){
  a = CoRec_Complete[CoRec_Complete$k == 5*i, ]
  a = a[a$a == 30,]
  a = mean(a$mae_u)
  h[i] = a
}
b = numeric(5)
for (i in 1:5){
  a = CoRec_Complete[CoRec_Complete$k == 5*i, ]
  a = a[a$a == 50,]
  a = mean(a$mae_u)
  b[i] = a
}

a = CoRec_Complete[CoRec_Complete$k == 15, ]
a = a[a$a == 50,]
r = a$mae_u

a = CoRec_Complete[CoRec_Complete$k == 15, ]
a = a[a$a == 50,]
s = a$mae_i

x = c(5,10,15,20,25)
which.min(b); min(b)
plot(x, user_knn, type='l',
     main = "K vs MAE of User CoRec",
     col= 1, pch=1, 
     xlab = "Neighbourhood Size",
     ylab="Mean MAE", lty=1,
     ylim = c(0.69,0.745))

points(x, user_knn, col=1, pch=1,  cex = 2, lw = 2)
lines(x, user_knn, col=1, pch=1,  lw = 3)

points(x, t, col=2, pch=2,  cex = 2, lw = 2)
lines(x, t, col=2,lty=2, lw= 3)

points(x, h, col=3, pch=3, cex = 2, lw = 2)
lines(x,h, col=3,lty=3,  lw= 3)

points(x, b, col=4, pch=4,  cex = 2, lw = 2)
lines(x,b, col=4,lty=4,  lw= 3)


legend('topright', legend = c( "Baseline", "Additions=10", "Additions=30", "Additions=50"), col = c(1:4),
       pch = c(1:4), lty = c(1:4))


####
# ITEM COREC
par(mfrow = c(1,1))
t = numeric(5)
for (i in 1:5){
  a = CoRec_Complete[CoRec_Complete$k == 5*i, ]
  a = a[a$a == 10,]
  a = mean(a$mae_i)
  t[i] = a
}

h = numeric(5)
for (i in 1:5){
  a = CoRec_Complete[CoRec_Complete$k == 5*i, ]
  a = a[a$a == 30,]
  a = mean(a$mae_i)
  h[i] = a
}
b = numeric(5)
for (i in 1:5){
  a = CoRec_Complete[CoRec_Complete$k == 5*i, ]
  a = a[a$a == 50,]
  a = mean(a$mae_i)
  b[i] = a
}

which.min(b); min(b)
plot(x, item_knn, type='l',
     main = "K vs MAE of Item CoRec",
     col= 1, pch=1, 
     xlab = "Neighbourhood Size",
     ylab="Mean MAE", lty=1,
     ylim = c(0.640,0.705))

points(x, item_knn, col=1, pch=1,  cex = 2, lw = 2)
lines(x, item_knn, col=1, pch=1,  lw = 3)

points(x, t, col=2, pch=2,  cex = 2, lw = 2)
lines(x, t, col=2,lty=2, lw= 3)

points(x, h, col=3, pch=3, cex = 2, lw = 2)
lines(x,h, col=3,lty=3,  lw= 3)

points(x, b, col=4, pch=4,  cex = 2, lw = 2)
lines(x,b, col=4,lty=4,  lw= 3)


legend('topright', legend = c( "Baseline", "Additions=10", "Additions=30", "Additions=50"), col = c(1:4),
       pch = c(1:4), lty = c(1:4))


#####
# TOGETHER
par(mfrow = c(1,2))
t = numeric(5)
for (i in 1:5){
  a = CoRec_Complete[CoRec_Complete$k == 5*i, ]
  a = a[a$a == 10,]
  a = mean(a$mae_u)
  t[i] = a
}
t

h = numeric(5)
for (i in 1:5){
  a = CoRec_Complete[CoRec_Complete$k == 5*i, ]
  a = a[a$a == 30,]
  a = mean(a$mae_u)
  h[i] = a
}
b = numeric(5)
for (i in 1:5){
  a = CoRec_Complete[CoRec_Complete$k == 5*i, ]
  a = a[a$a == 50,]
  a = mean(a$mae_u)
  b[i] = a
}

a = CoRec_Complete[CoRec_Complete$k == 15, ]
a = a[a$a == 50,]
r = a$mae_u

a = CoRec_Complete[CoRec_Complete$k == 15, ]
a = a[a$a == 50,]
s = a$mae_i

x = c(5,10,15,20,25)
which.min(b); min(b)
plot(x, user_knn, type='l',
     main = "K vs MAE of User CoRec",
     col= 1, pch=1, 
     xlab = "Neighbourhood Size",
     ylab="Mean MAE", lty=1,
     ylim = c(0.640,0.745))

points(x, user_knn, col=1, pch=1,  cex = 2, lw = 2)
lines(x, user_knn, col=1, pch=1,  lw = 3)

points(x, t, col=2, pch=2,  cex = 2, lw = 2)
lines(x, t, col=2,lty=2, lw= 3)

points(x, h, col=3, pch=3, cex = 2, lw = 2)
lines(x,h, col=3,lty=3,  lw= 3)

points(x, b, col=4, pch=4,  cex = 2, lw = 2)
lines(x,b, col=4,lty=4,  lw= 3)


legend('topright', legend = c( "Baseline", "Additions=10", "Additions=30", "Additions=50"), col = c(1:4),
       pch = c(1:4), lty = c(1:4))

t = numeric(5)
for (i in 1:5){
  a = CoRec_Complete[CoRec_Complete$k == 5*i, ]
  a = a[a$a == 10,]
  a = mean(a$mae_i)
  t[i] = a
}

h = numeric(5)
for (i in 1:5){
  a = CoRec_Complete[CoRec_Complete$k == 5*i, ]
  a = a[a$a == 30,]
  a = mean(a$mae_i)
  h[i] = a
}
b = numeric(5)
for (i in 1:5){
  a = CoRec_Complete[CoRec_Complete$k == 5*i, ]
  a = a[a$a == 50,]
  a = mean(a$mae_i)
  b[i] = a
}

which.min(b); min(b)
plot(x, item_knn, type='l',
     main = "K vs MAE of Item CoRec",
     col= 1, pch=1, 
     xlab = "Neighbourhood Size",
     ylab="Mean MAE", lty=1,
     ylim = c(0.640,0.745))

points(x, item_knn, col=1, pch=1,  cex = 2, lw = 2)
lines(x, item_knn, col=1, pch=1,  lw = 3)

points(x, t, col=2, pch=2,  cex = 2, lw = 2)
lines(x, t, col=2,lty=2, lw= 3)

points(x, h, col=3, pch=3, cex = 2, lw = 2)
lines(x,h, col=3,lty=3,  lw= 3)

points(x, b, col=4, pch=4,  cex = 2, lw = 2)
lines(x,b, col=4,lty=4,  lw= 3)


legend('topright', legend = c( "Baseline", "Additions=10", "Additions=30", "Additions=50"), col = c(1:4),
       pch = c(1:4), lty = c(1:4))



############################3
#Pearl pu
# 0.63389 min at k=15


((0.698504-0.63389)/0.698504)*100
t.test(k20, t, paired=TRUE)



#par(mfrow = c(1,1))
par(mar = c(5, 4, 4, 4) + 0.3)              # Additional space for second y-axis

x = c(5,10,15,20,25)
plot(x, user.knn.means, type='l',
     main = "K vs MAE of User Recursive KNN CS+",
     col= 1, pch=1, 
     xlab = "Neighbourhood Size",
     ylab="Mean MAE", lty=1,
     ylim = c(0.630,0.745))


points(x, user.knn.means, col=1, pch=1,  cex = 2, lw = 2)
lines(x, user.knn.means, col=1, pch=1,  lw = 3)

points(x, user.rec.means[user.rec.means$Group.2 == 1,]$x, col=2, pch=2,  cex = 2, lw = 2)
lines(x,  user.rec.means[user.rec.means$Group.2 == 1,]$x, col=2,lty=2, lw= 3)

points(x,  user.rec.means[user.rec.means$Group.2 == 2,]$x, col=3, pch=3,  cex = 2, lw = 2)
lines(x,  user.rec.means[user.rec.means$Group.2 == 2,]$x, col=3,lty=3, lw= 3)


legend('topright', legend = c("Baseline", "Recursion=1", "Recursion=2"), col = c(1:3),
       pch = c(1:3), lty = c(1:3))

which.min(t); min(t)

par(new = TRUE)  
x
# Add new plot
plot(x, user.rec.means.time[user.rec.means.time$Group.2 == 1,]$x, pch = 17, col = 3,              # Create second plot without axes
     axes = FALSE, xlab = "", ylab = "")
axis(side = 4, at = pretty(range(user.rec.means.time[user.rec.means.time$Group.2 == 1,]$x)))      # Add second axis
mtext("time", side = 4, line = 3)      


user.rec = Recursive.Complete
user.rec$time_elapsed_s = as.numeric(user.rec$time_elapsed_s)
user.rec.means = aggregate(user.rec$mae, list(user.rec$k, user.rec$r), FUN=mean)
user.rec.means.time = aggregate(user.rec$time_elapsed_s, list(user.rec$k, user.rec$r), FUN=mean)
user.rec.means
user.rec.best.k = 10
user.rec.best.r = 2
user.rec.best.result = 0.63389

which.min(user.rec$mae)

user.rec$time_elapsed_s
##############################
#EDA

par(mfrow = c(1,1))
ashs = (ratings$V2)
length(unique(ratings$V2))
length(ashs[ashs == 19])
#hist(ashs, breaks = 500)
asq = as.numeric(table(ashs))
asq
x <- sort(table(ashs),decreasing=FALSE)
x
as = as.data.frame(x)
as$ashs = as.numeric(as$ashs)
sum(as[as$ashs <= 10,]$Freq)/1682
length(as[as$Freq <= 100,]$ashs)/16.82 # THIS!!!
length(as[as$Freq <= 27,]$ashs)/16.82 # THIS!!!
sum(as[as$Freq >= 60,]$Freq)/10000


1682 - 52
# items 1682
# users 943

q1 = hist(x, breaks = 100, main = "Interactions Per Movie", xlab = "Number of Interactions")

ashs = ratings$V1
length(unique(ashs))
x <- sort(table(ashs),decreasing=FALSE)
x
as = as.data.frame(x)
as$ashs = as.numeric(as$ashs)

length(as[as$Freq <= 65,]$ashs)/9.43 # THIS!!!

length(ashs)
#hist(ashs, breaks = 500)
asq = as.numeric(table(ashs))
asq
#x <- sort(table(ashs),decreasing=FALSE)
p1 = hist(x , breaks = 150,  main = "Interactions Per User", xlab = "Number of Interactions")
plot(p1,  xlim = c(0,400), col = rgb(1,0,0,0.5), main = "Interactions Per User and Item", xlab = "Number of Interactions")
plot(q1, add = TRUE, col = rgb(0,0,1,0.5))
legend("topright", legend = c("Users", "Movies"), col = c(rgb(1,0,0,0.5), rgb(0,0,1,0.5)), fill = c(rgb(1,0,0,0.5),rgb(0,0,1,0.5)))

#asq
length(asq)
p = 0
c = 0
for (i in 1:length(asq))
  
x
{
c =  c + 1 
 p = p + x[i]
 print(x[i][2])
 if (p > 0.8*sum(asq)){break}
  
}
c
  
0.8*sum(asq)
# 943 users
# 64544 # ratings
# 

par(mfrow = c(1,1))
ashs = (ratings$V3)
ashs
length(ashs)
#hist(ashs, breaks = 500)
asq = as.numeric(table(ashs))
asq
x <- sort(table(ashs), decreasing=FALSE)
x
q=hist(ashs, freq = TRUE) 
plot(asq, ylab = "Frequency",xlab = "Ratings", main = "Distribution of Ratings")
lines(asq)
plot(q, add = TRUE)
asq / length(ashs)
hist(asq)  
m = c(1,2,3,4,5)
hist(rbind(m, asq))


##################
# cold starts

