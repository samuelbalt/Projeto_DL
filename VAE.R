list.of.packages <- c("R.matlab", "fossil", "umap", "aricode")
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages)

library(R.matlab)
library(fossil)
library(umap)
library(aricode)


cll_sub_111 <- readMat("/home/samuel/Documentos/CIn/Deep Learning//CLL_SUB_111.mat")
cll_sub_111_x <- scale(cll_sub_111$X)
cll_sub_111_y <- cll_sub_111$Y

carcinom <- readMat("/home/samuel/Documentos/CIn/Deep Learning//CARCINOM.mat")
carcinom_x <- scale(carcinom$X)
carcinom_y <- carcinom$Y

tox_171 <- readMat("/home/samuel/Documentos/CIn/Deep Learning//TOX_171.mat")
tox_171_x <- tox_171$X
tox_171_x <- scale(tox_171_x)
tox_171_y  <- tox_171$Y

Prostate_GE <- readMat("/home/samuel/Documentos/CIn/Deep Learning//Prostate_GE.mat")
Prostate_GE_x <- Prostate_GE$X
Prostate_GE_x <- scale(Prostate_GE_x)
Prostate_GE_y  <- Prostate_GE$Y



set.seed(2012) #semente


dados_x <- carcinom_x
dados_y <- carcinom_y
nc <- length(summary(as.factor(carcinom_y)))
train_ind <- sample(1:nrow(dados_x), 0.8*nrow(dados_x))


# Implementação do VAE baseada no presente no livro "Deep Learning with R" (Chollet, 2017)
if (tensorflow::tf$executing_eagerly())
  tensorflow::tf$compat$v1$disable_eager_execution()

library(keras)
K <- keras::backend()

# Parametros --------------------------------------------------------------

batch_size <- 100L ##batch
original_dim <- ncol(dados_x) ##dimensão original
latent_dim <- 200L ##tamanho espaço latente/dimensão latente
intermediate_dim <- ncol(dados_x)/2  #tamanho dimensão intermediária
epochs <- 150L  #épocas de treinaento
epsilon_std <- 1.0 
LR <- 0.0005 ##taxa de aprendizagem

# modelo  --------------------------------------------------------

x <- layer_input(shape = c(original_dim))
h <- layer_dense(x, intermediate_dim, activation = "relu")
z_mean <- layer_dense(h, latent_dim)
z_log_var <- layer_dense(h, latent_dim)

sampling <- function(arg){
  z_mean <- arg[, 1:(latent_dim)]
  z_log_var <- arg[, (latent_dim + 1):(2 * latent_dim)]
  
  epsilon <- k_random_normal(
    shape = c(k_shape(z_mean)[[1]]), 
    mean=0.,
    stddev=epsilon_std
  )
  
  z_mean + k_exp(z_log_var/2)*epsilon
}

z <- layer_concatenate(list(z_mean, z_log_var)) %>% 
  layer_lambda(sampling)

decoder_h <- layer_dense(units = intermediate_dim, activation = "sigmoid")
decoder_mean <- layer_dense(units = original_dim, activation = "sigmoid")
h_decoded <- decoder_h(z)
x_decoded_mean <- decoder_mean(h_decoded)

vae <- keras_model(x, x_decoded_mean)

# encoder
encoder <- keras_model(x, z_mean)

decoder_input <- layer_input(shape = latent_dim)
h_decoded_2 <- decoder_h(decoder_input)
x_decoded_mean_2 <- decoder_mean(h_decoded_2)
generator <- keras_model(decoder_input, x_decoded_mean_2)

##função de perca
vae_loss <- function(x, x_decoded_mean){
  xent_loss <- (original_dim/1.0)*loss_binary_crossentropy(x, x_decoded_mean)
  kl_loss <- -0.5*k_mean(1 + z_log_var - k_square(z_mean) - k_exp(z_log_var), axis = -1L)
  xent_loss + kl_loss
}

vae %>% compile(optimizer = optimizer_adam(learning_rate = LR), loss = vae_loss)


###treinamento do modelo
vae %>% fit(
  dados_x[train_ind,], y=NULL,
  shuffle = TRUE, 
  epochs = epochs, 
  batch_size = batch_size
)

ClusterPurity <- function(clusters, classes) {
  sum(apply(table(classes, clusters), 2, max)) / length(clusters)
}

purity_vae <- NULL
rand_vae <- NULL
nmi_vae <- NULL

purity_umap <- NULL
rand_umap <- NULL
nmi_umap <- NULL

##treinamento umap
train.umap <- umap(dados_x[train_ind,], n_neighbors = sqrt(nrow(dados_x[train_ind,])) ,n_components = 200,
                   n_epochs = 150)


for(i in 1:50){
 
x_test_encoded <- predict(encoder, dados_x[-train_ind,])
test.umap <- predict(train.umap, dados_x[-train_ind,])
k_vae <- kmeans(x_test_encoded,nc)
k_umap <- kmeans(test.umap,nc)
purity_vae[i] <- ClusterPurity(k_vae$cluster,dados_y[-train_ind])
purity_umap[i] <- ClusterPurity(k_umap$cluster,dados_y[-train_ind])
rand_vae[i] <- rand.index(k_vae$cluster,dados_y[-train_ind])
rand_umap[i] <- rand.index(k_umap$cluster,dados_y[-train_ind])
nmi_vae[i] <- NMI(k_vae$cluster,dados_y[-train_ind])
nmi_umap[i] <- NMI(k_umap$cluster,dados_y[-train_ind])


}

desempenho <- cbind.data.frame(purity_vae, purity_umap, rand_vae, rand_umap, nmi_vae, nmi_umap)
round(apply(desempenho,2,mean), digits = 2)
round(apply(desempenho,2,sd), digits = 3)


wilcox.test(desempenho$purity_umap, desempenho$purity_vae, paired = TRUE, alternative = 'greater')
wilcox.test(desempenho$rand_umap, desempenho$rand_vae, paired = TRUE, alternative = 'greater')
wilcox.test(desempenho$nmi_umap, desempenho$nmi_vae, paired = TRUE, alternative = 'greater')




