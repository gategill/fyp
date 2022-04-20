################################################################################
# DEFINE FUNCTIONS
################################################################################

diffr = function(l1, l2){
  l1.min = min(l1)
  l2.min = min(l2)
  
  per = (l1.min-l2.min)/l1.min
  out = round(per*100, 2)
  
  print(paste("l2 outperforms l1 by ", out, "%", sep = ""))
  return(out)
}

################################################################################

make_plots = function(title, list_of) {
  n = length(list_of)
  
  mi = 100
  ma = 0
  for (i in 1:n){
    if (min(list_of[[i]]) < mi){
      mi = min(list_of[[i]])
    }
    if (max(list_of[[i]]) > ma){
      ma = max(list_of[[i]])
    }
  }
  
  par(mfrow = c(1,1))
  for (i in 1:n){
    if (i == 1){
      plot(nearestNeighbours, l[[i]], 
           #main = title,
           ylab = "Mean Absolute Error", 
           xlab = "Neighbourhood Size",
           ylim = c(mi*0.99, ma*1.01), 
           type = "b",
           col = i,
           pch = i, 
           lty = i,
           cex = 3,
           lwd = 3)
    }
    else {
      points(nearestNeighbours, l[[i]], col = i, cex = 3, pch = i)
      lines(nearestNeighbours, l[[i]], col = i, lwd = 3, lty = i)
    }
  }
}

################################################################################

final_plot = function(title, l, legs){
  sa = paste("images/", title, ".jpeg", sep = "")
  png(sa, width = 800, height = 600)
  n = length(l)
  
  make_plots(title, l)
  
  legend('topright',
         legend = legs,
         col = 1:n,
         pch = 1:n,
         lty = 1:n)
  
  dev.off()
}

################################################################################
# DEFINE DATASETS
################################################################################

nearestNeighbours  = c(5, 10, 15, 20, 25)

################################################################################

# HOT MOVIES
userKNN        = c(0.734, 0.714, 0.711, 0.709, 0.708)
itemKNN        = c(0.686, 0.672, 0.672, 0.673, 0.675)
userBootstrap1 = c(0.734, 0.713, 0.706, 0.707, 0.705)
userBootstrap2 = c(0.731, 0.713, 0.708, 0.707, 0.706)
userBootstrap3 = c(0.733, 0.713, 0.709, 0.709, 0.707)
itemBootstrap1 = c(0.681, 0.669, 0.668, 0.670, 0.675)
itemBootstrap2 = c(0.683, 0.668, 0.668, 0.670, 0.673)
itemBootstrap3 = c(0.676, 0.668, 0.666, 0.669, 0.672)
userCoRec10    = c(0.733, 0.709, 0.708, 0.706, 0.706)
userCoRec30    = c(0.728, 0.710, 0.706, 0.704, 0.705)
userCoRec50    = c(0.728, 0.705, 0.704, 0.702, 0.703)
itemCoRec10    = c(0.687, 0.672, 0.671, 0.672, 0.675)
itemCoRec30    = c(0.686, 0.669, 0.667, 0.669, 0.671)
itemCoRec50    = c(0.682, 0.661, 0.663, 0.665, 0.669)
userRecursive1 = c(0.658, 0.652, 0.655, 0.657, 0.659)
userRecursive2 = c(0.659, 0.653, 0.655, 0.656, 0.659)
itemRecursive1 = c(0.604, 0.604, 0.608, 0.613, 0.617)
itemRecursive2 = c(0.604, 0.605, 0.609, 0.613, 0.618)

################################################################################

# COLD MOVIES
cold.userKNN        = c(0.942,0.942,0.943,0.945,0.945)
cold.itemKNN        = c(0.923,0.923,0.924,0.924,0.925)
cold.userBootstrap1 = c(0.943,0.943,0.945,0.942,0.944)
cold.userBootstrap2 = c(0.940,0.943,0.944,0.945,0.946)
cold.userBootstrap3 = c(0.942,0.941,0.942,0.945,0.946)
cold.itemBootstrap1 = c(0.927,0.929,0.930,0.929,0.934)
cold.itemBootstrap2 = c(0.939,0.943,0.940,0.943,0.947)
cold.itemBootstrap3 = c(0.949,0.948,0.947,0.950,0.950)
cold.userCoRec10    = c(0.938,0.942,0.943,0.944,0.945)
cold.userCoRec30    = c(0.922,0.916,0.919,0.928,0.936)
cold.userCoRec50    = c(0.904,0.899,0.900,0.899,0.898)
cold.itemCoRec10    = c(0.902,0.905,0.907,0.910,0.905)
cold.itemCoRec30    = c(0.927,0.928,0.930,0.928,0.930)
cold.itemCoRec50    = c(0.924,0.925,0.925,0.926,0.926)
cold.userRecursive1 = c(0.937,0.936,0.938,0.938,0.938)
cold.userRecursive2 = c(0.938,0.937,0.938,0.938,0.938)
cold.itemRecursive1 = c(0.929,0.929,0.929,0.929,0.929)
cold.itemRecursive2 = c(0.929,0.929,0.929,0.929,0.929)

################################################################################

# HOT BOOKS
books.userKNN        = c(0.734, 0.714, 0.711, 0.709, 0.708)
books.itemKNN        = c(0.686, 0.672, 0.672, 0.673, 0.675)
books.userBootstrap1 = c(0.734, 0.713, 0.706, 0.707, 0.705)
books.userBootstrap2 = c(0.731, 0.713, 0.708, 0.707, 0.706)
books.userBootstrap3 = c(0.733, 0.713, 0.709, 0.709, 0.707)
books.itemBootstrap1 = c(0.681, 0.669, 0.668, 0.670, 0.675)
books.itemBootstrap2 = c(0.683, 0.668, 0.668, 0.670, 0.673)
books.itemBootstrap3 = c(0.676, 0.668, 0.666, 0.669, 0.672)
books.userCoRec10    = c(0.733, 0.709, 0.708, 0.706, 0.706)
books.userCoRec30    = c(0.728, 0.710, 0.706, 0.704, 0.705)
books.userCoRec50    = c(0.728, 0.705, 0.704, 0.702, 0.703)
books.itemCoRec10    = c(0.687, 0.672, 0.671, 0.672, 0.675)
books.itemCoRec30    = c(0.686, 0.669, 0.667, 0.669, 0.671)
books.itemCoRec50    = c(0.682, 0.661, 0.663, 0.665, 0.669)
books.userRecursive1 = c(0.658, 0.652, 0.655, 0.657, 0.659)
books.userRecursive2 = c(0.659, 0.653, 0.655, 0.656, 0.659)
books.itemRecursive1 = c(0.604, 0.604, 0.608, 0.613, 0.617)
books.itemRecursive2 = c(0.604, 0.605, 0.609, 0.613, 0.618)

################################################################################

# HOT JESTER
jester.userKNN        = c(0.734, 0.714, 0.711, 0.709, 0.708)
jester.itemKNN        = c(0.686, 0.672, 0.672, 0.673, 0.675)
jester.userBootstrap1 = c(0.734, 0.713, 0.706, 0.707, 0.705)
jester.userBootstrap2 = c(0.731, 0.713, 0.708, 0.707, 0.706)
jester.userBootstrap3 = c(0.733, 0.713, 0.709, 0.709, 0.707)
jester.itemBootstrap1 = c(0.681, 0.669, 0.668, 0.670, 0.675)
jester.itemBootstrap2 = c(0.683, 0.668, 0.668, 0.670, 0.673)
jester.itemBootstrap3 = c(0.676, 0.668, 0.666, 0.669, 0.672)
jester.userCoRec10    = c(0.733, 0.709, 0.708, 0.706, 0.706)
jester.userCoRec30    = c(0.728, 0.710, 0.706, 0.704, 0.705)
jester.userCoRec50    = c(0.728, 0.705, 0.704, 0.702, 0.703)
jester.itemCoRec10    = c(0.687, 0.672, 0.671, 0.672, 0.675)
jester.itemCoRec30    = c(0.686, 0.669, 0.667, 0.669, 0.671)
jester.itemCoRec50    = c(0.682, 0.661, 0.663, 0.665, 0.669)
jester.userRecursive1 = c(0.658, 0.652, 0.655, 0.657, 0.659)
jester.userRecursive2 = c(0.659, 0.653, 0.655, 0.656, 0.659)
jester.itemRecursive1 = c(0.604, 0.604, 0.608, 0.613, 0.617)
jester.itemRecursive2 = c(0.604, 0.605, 0.609, 0.613, 0.618)




################################################################################
# GETTING DIFFERENCES IN PERFORMANCE
################################################################################

diffr(cold.userKNN, cold.itemKNN)


################################################################################
# TEST FOR STATISTICAL  SIGNIFICANCE
################################################################################

t.test(userKNN, itemKNN, paired = TRUE)
t.test(userKNN, userBootstrap1, paired = TRUE)
t.test(itemKNN, itemBootstrap3, paired = TRUE)
t.test(userKNN, userRecursive1, paired = TRUE)
t.test(itemKNN, itemRecursive1, paired = TRUE)
t.test(userKNN, userCoRec50, paired = TRUE)
t.test(itemKNN, itemCoRec50, paired = TRUE)


t.test(userKNN, cold.userKNN, paired = TRUE)
t.test(itemKNN, cold.itemKNN, paired = TRUE)

t.test(cold.userKNN, cold.itemKNN, paired = TRUE)
t.test(cold.userKNN, cold.userBootstrap2, paired = TRUE)
t.test(cold.itemKNN, cold.itemBootstrap1, paired = TRUE)
t.test(cold.userKNN, cold.userRecursive1, paired = TRUE)
t.test(cold.itemKNN, cold.itemRecursive1, paired = TRUE)
t.test(cold.userKNN, cold.userCoRec50, paired = TRUE)
t.test(cold.itemKNN, cold.itemCoRec10, paired = TRUE)

################################################################################
 # PLOT RESULTS
################################################################################

# HOT MOVIES
# User vs Item KNN
title = "user_item_knn"
l = list(userKNN, itemKNN)
legs = c("User KNN", "Item KNN")
final_plot(title, l, legs)

title = "hot_and_cold_user_item_knn"
l = list(userKNN, itemKNN, cold.userKNN, cold.itemKNN)
legs = c("Hot User KNN", "Hot Item KNN", "Cold User KNN", "Cold Item KNN")
final_plot(title, l, legs)

# User Bootstrap
title = "user_bootstrap"
l = list(userKNN, userBootstrap1, userBootstrap2, userBootstrap3)
legs = c("Baseline", "Rounds = 1", "Rounds = 2", "Rounds = 3")
final_plot(title, l, legs)

# Item Bootstrap
title = "item_bootstrap"
l = list(itemKNN, itemBootstrap1, itemBootstrap2, itemBootstrap3)
legs = c("Baseline", "Rounds = 1", "Rounds = 2", "Rounds = 3")
final_plot(title, l, legs)

# user item boostrap
title = "user_item_bootstrap"
l = list(userKNN, userBootstrap1, userBootstrap2, userBootstrap3, itemKNN, itemBootstrap1, itemBootstrap2, itemBootstrap3)
legs = c("User Baseline", "Rounds = 1", "Rounds = 2", "Rounds = 3", "Item Baseline", "Rounds = 1", "Rounds = 2", "Rounds = 3")
final_plot(title, l, legs)

# User Recursive
title = "user_recursive"
l = list(userKNN, userRecursive1, userRecursive2)
legs = c("Baseline", "Recursion = 1", "Recursion = 2")
final_plot(title, l, legs)

# Item Recursive
title = "item_recursive"
l = list(itemKNN, itemRecursive1, itemRecursive2)
legs = c("Baseline", "Recursion = 1", "Recursion = 2")
final_plot(title, l, legs)

# User CoRec Recursive
title = "user_corec"
l = list(userKNN, userCoRec10, userCoRec30, userCoRec50)
legs = c("Baseline", "Additions = 10", "Additions = 30", "Additions = 50")
final_plot(title, l, legs)

# Item CoRec Recursive
title = "item_corec"
l = list(itemKNN, itemCoRec10, itemCoRec30, itemCoRec50)
legs = c("Baseline", "Additions = 10", "Additions = 30", "Additions = 50")
final_plot(title, l, legs)

################################################################################

# COLD Movies
# User vs Item KNN
title = "cold_user_item_knn"
l = list(cold.userKNN, cold.itemKNN)
legs = c("User KNN", "Item KNN")
final_plot(title, l, legs)

# User Bootstrap
title = "cold_user_bootstrap"
l = list(cold.userKNN, cold.userBootstrap1, cold.userBootstrap2, cold.userBootstrap3)
legs = c("Baseline", "Rounds = 1", "Rounds = 2", "Rounds = 3")
final_plot(title, l, legs)

# Item Bootstrap
title = "cold_item_bootstrap"
l = list(cold.itemKNN, cold.itemBootstrap1, cold.itemBootstrap2, cold.itemBootstrap3)
legs = c("Baseline", "Rounds = 1", "Rounds = 2", "Rounds = 3")
final_plot(title, l, legs)

# User Recursive
title = "cold_user_recursive"
l = list(cold.userKNN, cold.userRecursive1, cold.userRecursive2)
legs = c("Baseline", "Recursion = 1", "Recursion = 2")
final_plot(title, l, legs)

# Item Recursive
title = "cold_item_recursive"
l = list(cold.itemKNN, cold.itemRecursive1, cold.itemRecursive2)
legs = c("Baseline", "Recursion = 1", "Recursion = 2")
final_plot(title, l, legs)

# User CoRec Recursive
title = "cold_user_corec"
l = list(cold.userKNN, cold.userCoRec10, cold.userCoRec30, cold.userCoRec50)
legs = c("Baseline", "Additions = 10", "Additions = 30", "Additions = 50")
final_plot(title, l, legs)

# Item CoRec Recursive
title = "cold_item_corec"
l = list(cold.itemKNN, cold.itemCoRec10, cold.itemCoRec30, cold.itemCoRec50)
legs = c("Baseline", "Additions = 10", "Additions = 30", "Additions = 50")
final_plot(title, l, legs)


################################################################################

# HOT BOOKS
# User vs Item KNN
title = "books_user_item_knn"
l = list(books.userKNN, books.itemKNN)
legs = c("User KNN", "Item KNN")
final_plot(title, l, legs)

# User Bootstrap
title = "books_user_bootstrap"
l = list(books.userKNN, books.userBootstrap1, books.userBootstrap2, books.userBootstrap3)
legs = c("Baseline", "Rounds = 1", "Rounds = 2", "Rounds = 3")
final_plot(title, l, legs)

# Item Bootstrap
title = "books_item_bootstrap"
l = list(books.itemKNN, books.itemBootstrap1, books.itemBootstrap2, books.itemBootstrap3)
legs = c("Baseline", "Rounds = 1", "Rounds = 2", "Rounds = 3")
final_plot(title, l, legs)

# User Recursive
title = "books_user_recursive"
l = list(books.userKNN, books.userRecursive1, books.userRecursive2)
legs = c("Baseline", "Recursion = 1", "Recursion = 2")
final_plot(title, l, legs)

# Item Recursive
title = "books_item_recursive"
l = list(books.itemKNN, books.itemRecursive1, books.itemRecursive2)
legs = c("Baseline", "Recursion = 1", "Recursion = 2")
final_plot(title, l, legs)

# User CoRec Recursive
title = "books_user_corec"
l = list(books.userKNN, books.userCoRec10, books.userCoRec30, books.userCoRec50)
legs = c("Baseline", "Additions = 10", "Additions = 30", "Additions = 50")
final_plot(title, l, legs)

# Item CoRec Recursive
title = "books_item_corec"
l = list(books.itemKNN, books.itemCoRec10, books.itemCoRec30, books.itemCoRec50)
legs = c("Baseline", "Additions = 10", "Additions = 30", "Additions = 50")
final_plot(title, l, legs)


################################################################################

# HOT JESTER
# User vs Item KNN
title = "jester_user_item_knn"
l = list(jester.userKNN, jester.itemKNN)
legs = c("User KNN", "Item KNN")
final_plot(title, l, legs)

# User Bootstrap
title = "jester_user_bootstrap"
l = list(jester.userKNN, jester.userBootstrap1, jester.userBootstrap2, jester.userBootstrap3)
legs = c("Baseline", "Rounds = 1", "Rounds = 2", "Rounds = 3")
final_plot(title, l, legs)

# Item Bootstrap
title = "jester_item_bootstrap"
l = list(jester.itemKNN, jester.itemBootstrap1, jester.itemBootstrap2, jester.itemBootstrap3)
legs = c("Baseline", "Rounds = 1", "Rounds = 2", "Rounds = 3")
final_plot(title, l, legs)

# User Recursive
title = "jester_user_recursive"
l = list(jester.userKNN, jester.userRecursive1, jester.userRecursive2)
legs = c("Baseline", "Recursion = 1", "Recursion = 2")
final_plot(title, l, legs)

# Item Recursive
title = "jester_item_recursive"
l = list(jester.itemKNN, jester.itemRecursive1, jester.itemRecursive2)
legs = c("Baseline", "Recursion = 1", "Recursion = 2")
final_plot(title, l, legs)

# User CoRec Recursive
title = "jester_user_corec"
l = list(jester.userKNN, jester.userCoRec10, jester.userCoRec30, jester.userCoRec50)
legs = c("Baseline", "Additions = 10", "Additions = 30", "Additions = 50")
final_plot(title, l, legs)

# Item CoRec Recursive
title = "jester_item_corec"
l = list(jester.itemKNN, jester.itemCoRec10, jester.itemCoRec30, jester.itemCoRec50)
legs = c("Baseline", "Additions = 10", "Additions = 30", "Additions = 50")
final_plot(title, l, legs)


################################################################################
# Exploratory Data Analysis
################################################################################

par(mfrow = c(1,1))
ashs = (all_ratings$item_id)
asq = as.numeric(table(ashs))
xc <- sort(table(ashs),decreasing=FALSE)
q1 = hist(xc , breaks = 50, 
          main = "Interactions Per Movie",
          xlab = "Number of Interactions")

ashs = (all_ratings$user_id)
#hist(ashs, breaks = 500)
asq = as.numeric(table(ashs))
xc <- sort(table(ashs),decreasing=FALSE)
p1 = hist(xc , breaks = 50,
          main = "Interactions Per User",
          xlab = "Number of Interactions")

plot(q1, col = 2, xlim = c(0,400),
     main = "Interactions Per User and Item",
     xlab = "Numeber of Interactions")
plot(p1, add = TRUE, col = 3)
legend("topright",
       legend = c("Users", "Movies"), col = c(2,3), fill = c(2,3))


p = 0
c = 0
for (i in 1:length(asq)){
 c =  c + 1 
 p = p + NearestNeighbours[i]
 print(NearestNeighbours[i][2])
 if (p > 0.8*sum(asq)){break}
}

par(mfrow = c(1,1))
ashs = (ratings$V3)
#hist(ashs, breaks = 500)
asq = as.numeric(table(ashs))
xc <- sort(table(ashs), decreasing=FALSE)
q=hist(ashs, freq = TRUE) 
plot(asq, ylab = "Frequency",
     xlab = "Ratings",
     main = "Distribution of Ratings")

lines(asq)
plot(q, add = TRUE)
asq / length(ashs)
hist(asq)  
m = c(1,2,3,4,5)
hist(rbind(m, asq))

h <- sort(table(all_ratings$item_id),decreasing=T)
hist(h, breaks = 30,
     main = "Frequency Distribution of Movies",
     freq =  FALSE,
     xlab = "Movies")






