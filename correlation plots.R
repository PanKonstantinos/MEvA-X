library(heatmap3)
library(gplots)
library(ggplot2)
library(devtools)
library(factoextra)
library(ggfortify)
library(ggpubr)

#install_github("vqv/ggbiplot")
########
# Functions
plot_PC_variance <- function(PCA, m_title) {
  pca_data = data.frame('values'=summary(PCA)$importance[2,],'names'=colnames(summary(PCA)$importance))
  ggplot(pca_data, aes(x=reorder(names,sort(values)), y=values))+
    geom_bar(stat = "identity", fill='LightBlue')+
    ggtitle(m_title)+
    scale_x_discrete("Principal Components")+
    scale_y_continuous("Percentage of explained variance")+
    theme(axis.text.x=element_text(size=15),
          axis.text.y=element_text(size=15),
          axis.title.x=element_text(size=15),
          axis.title.y=element_text(size=15),
          plot.title = element_text(size=20))+
    geom_text(aes(label=round(values,3)), colour="black", size=6 ,vjust=-0.3)
}

plot_PCA <- function(PCA, m_data, m_title) {
  color_by = row.names(m_data)[nrow(m_data)]
  autoplot(PCA, data = t(m_data), colour = color_by, size = 3,
           loadings=TRUE, loadings.colour = "black",
           loadings.label=TRUE, loadings.label.colour = 'black', scale = 0)+
    ggtitle(m_title)+
    theme(axis.text.x=element_text(size=15),
          axis.text.y=element_text(size=15),
          axis.title.x=element_text(size=15),
          axis.title.y=element_text(size=15),
          legend.text = element_text(size=15))#+theme_bw()
}

plot_feature_contribution_in_pca <- function(PCA,PCs) {
  p <- fviz_contrib(PCA,
                    choice = "var",
                    axes = PCs,
                    top = 50,
                    fill = "lightgray", color = "black")
  ggpar(p,xlab="Feature names")+
    theme_minimal()+
    scale_x_discrete(guide = guide_axis(angle=45))+
    theme(axis.text.x=element_text(size=15),
          axis.text.y=element_text(size=15),
          axis.title.x=element_text(size=15),
          axis.title.y=element_text(size=15))
}
plot_feature_contribution_in_pca(PCA,1)


########
# script

data <- read.csv('C:/Users/UTENTE/Desktop/PEZ/Master Thesis/Code and Data/diet/InSyBio_modified_dataset.txt', sep = '\t')
diet_labels <- read.csv('C:/Users/UTENTE/Desktop/PEZ/Master Thesis/Code and Data/diet/diet_labels.txt', sep = '\t', header = FALSE)
rownames(data) <- data$X
data <- data[,2:length(colnames(data))]


list_of_vars = c("AP2B1","BIK","LINC00588","MBD3","PSPN","PXMP4","RAC3","TFDP2","TMEM33")

data2 <- data.frame(data[list_of_vars,])
correlation <- data.frame(cor(t(data2), method = 'spearman'))


lmat = rbind(c(4,3),c(2,1))
lwid = c(1.5,1)
lhei = c(1.5,0.5)
font_size = 1.5
heatmap.2(cor(t(data2), method = 'spearman'), trace  = 'none',
          cellnote = round(correlation, digits=2),
          notecol = 'black', dendrogram="row", margins = c(8,8),
          notecex=font_size, cexRow = font_size, cexCol = font_size,
          revC = TRUE, key=T, key.title = 'Color Scale',
          breaks = seq(-1, 1, length.out = 101), 
          lmat = lmat, lwid = c(1,4), lhei = c(1,4))#breaks = seq(-1, 1, length.out = 101)
#width = ncol(cor(t(data2)))*5,height = nrow(cor(t(data2)))*10)
#dev.off()

data_for_pca <- data2
data_for_pca[nrow(data_for_pca) + 1,] <- diet_labels
renamed_indices <- rownames(data_for_pca)
renamed_indices[length(renamed_indices)] = 'Labels'

rownames(data_for_pca) <- renamed_indices

PCA = prcomp(t(data2), scale. = FALSE, rank. = nrow(data2))



summary(PCA)
summary(PCA)$importance[2,]

m_title = "Explanation of variance for the Principal Components of the precision diet (Ornish) dataset"
plot_PC_variance(PCA,m_title)

pca_data = data.frame('values'=summary(PCA)$importance[2,],'names'=colnames(summary(PCA)$importance))


m_title = "Principal Components Analysis (PC1 Vs PC2) of the precision diet (Ornish) dataset"
plot_PCA(PCA, data_for_pca, m_title)
# biplot(PCA, xlabs=c("x", "o")[as.numeric(as.factor(tail(data_for_pca,n=1)))],
#        cex=1,)
#biplot(PCA, xlabs=c("x", "o")[as.numeric(tail(data_for_pca,n=1))], cex=.75)
plot_feature_contribution_in_pca(PCA,1)
plot_feature_contribution_in_pca(PCA,2)
plot_feature_contribution_in_pca(PCA,1:2)





######################         ######################
######################  OPERA  ######################
######################         ######################
opera_data <- read.csv('C:/Users/UTENTE/Desktop/PEZ/Master Thesis/Code and Data/Opera/OPERA_Imputed_Normalized_modified.csv')
rownames(opera_data) <- opera_data$X.1
opera_data <- opera_data[,3:length(colnames(opera_data))]

opera1_labels <- read.csv('C:/Users/UTENTE/Desktop/PEZ/Master Thesis/Code and Data/Opera/opera_full_labels_binary_1.csv')
opera2_labels <- read.csv('C:/Users/UTENTE/Desktop/PEZ/Master Thesis/Code and Data/Opera/opera_full_labels_binary_2.csv')
opera3_labels <- read.csv('C:/Users/UTENTE/Desktop/PEZ/Master Thesis/Code and Data/Opera/opera_full_labels_binary_3.csv')
opera4_labels <- read.csv('C:/Users/UTENTE/Desktop/PEZ/Master Thesis/Code and Data/Opera/opera_full_labels_binary_4.csv')

list_of_vars1 <- c('TFFC3', 'Age',
                  'Gender', 'Other_bin', 'Pain_other',
                  'Int_Average_pain', 'Current_pain', 'Severity_Score',
                  'Int_Work', 'Int_Sleep', 'Int_Life_enjoyment',
                  'Avg_Interference', 'Narcotic_weight',
                  'Grand_Tot_Med_weight')

#list_of_vars1 <- c('TFourFormulationCategories_3 ', 'AgeatSurvey1 ',
#                   'GenderRECODE', 'Q5S1OtherRecode ', 'Q7S1 ',
#                   'Q11S1Average ', 'Q12S1RightNow', 'S1SeverityScoreOutof10 ',
#                   'Q17S1NormalWork ', 'Q19S1Sleep', 'Q20S1EnjoymentofLife ',
#                   'S1InferenceScoreOutof10 ', 'Q29S1NarcoticWeight10',
#                   'S1GrandTotalMedicinesWEIGHTED')

list_of_vars1 <- gsub(" ", "", list_of_vars1, fixed = TRUE)
opera1_data <- data.frame(opera_data[list_of_vars1,])
correlation <- data.frame(cor(t(opera1_data), method = 'spearman'))

lmat = rbind(c(4,3),c(2,1))
lwid = c(1.5,1)
lhei = c(1.5,0.5)
#layout(mat = lmat, widths = lwid, heights = lhei)
font_size = 1.5
marg = 15
heatmap.2(cor(t(opera1_data), method = 'spearman'), trace  = 'none',
          cellnote = round(correlation, digits=2),
          notecol = 'black', dendrogram="row", margins = c(15,15),
          notecex=1.5, cexRow = font_size, cexCol = font_size,
          keysize=0, revC = TRUE,key = T, key.title = 'Color Key',
          breaks = seq(-1, 1, length.out = 101), symkey=TRUE,
          lmat = lmat, lwid = c(1,4), lhei = c(1,4))#, breaks = seq(-1, 1, length.out = 101)), lmat = lmat,lwid=lwid,lhei=lhei,

dev.off()

data_for_pca_opera1 <- opera1_data
rownames(opera1_labels) <- opera1_labels$X
opera1_labels_n <- opera1_labels[,2:length(colnames(opera1_labels))]
opera1_labels_n <- as.factor(opera1_labels_n)
data_for_pca_opera1[nrow(data_for_pca_opera1) + 1,] <- opera1_labels_n


renamed_indices <- rownames(data_for_pca_opera1)
renamed_indices[length(renamed_indices)] = 'Change_In_Severity'#'Labels'

rownames(data_for_pca_opera1) <- renamed_indices

PCA = prcomp(t(opera1_data), scale. = FALSE, rank. = length(rownames(opera1_data)))

m_title = 'Explanation of variance for the Principal Components of the OPERA study dataset for Label_1'
plot_PC_variance(PCA, m_title)

summary(PCA)
summary(PCA)$importance[2,]

pca_data = data.frame('values'=summary(PCA)$importance[2,],'names'=colnames(summary(PCA)$importance))
# #reorder(names,values)
# ggplot(pca_data, aes(x=reorder(names,sort(values)), y=values))+
#   geom_bar(stat = "identity", fill='LightBlue')+
#   ggtitle("Explanation of variance for the Principal Components of the OPERA dataset and Label_1")+
#   scale_x_discrete("Principal Components")+
#   scale_y_continuous("Percentage of explained variance")+
#   theme(axis.text.x=element_text(size=15),
#         axis.text.y=element_text(size=15),
#         axis.title.x=element_text(size=15),
#         axis.title.y=element_text(size=15),
#         plot.title = element_text(size=20))+
#   geom_text(aes(label=round(values,3)), colour="black", size=6 ,vjust=-0.3)

m_title = "Principal Components Analysis (PC1 Vs PC2) of the OPERA study dataset on Label_1"
plot_PCA(PCA, data_for_pca_opera1, m_title)

#biplot(PCA, xlabs=c("x", "o")[as.numeric(tail(data_for_pca,n=1))], cex=.75)
plot_feature_contribution_in_pca(PCA,1)
plot_feature_contribution_in_pca(PCA,2)
plot_feature_contribution_in_pca(PCA,1:2)



######################
list_of_vars2 <- c('TFFC2','TFFC3','TFFC4','Gender', 'Arthritis_categ',
                  'MyoMuscul_cat', 'Tendinitis_bin',
                  'Grand_Tot_Compl', 'Worst_24h',
                  'Average_pain', 'Severity_Score', 'Int_Mood',
                  'Int_Work', 'Int_Sleep', 'Avg_Interference',
                  'AntiInflam_categ', 'Opioid_comb',
                  'Grand_Tot_Med_weight', 'AntiInflam_bin')

# list_of_vars2 <- c('TFourFormulationCategories_2 ',
#                    'TFourFormulationCategories_3', 
#                    'TFourFormulationCategories_4 ',
#                    'GenderRECODE ', 'Q1S1ArthritisTotal',
#                    'Q3S1MyoMusculPainORSpasmTotal ', 'Q4S1TendinitisRecode',
#                    'S1GrandTotalofAllComplaintsExceptOverallOther ', 'Q9S1Worst',
#                    'Q11S1Average ', 'S1SeverityScoreOutof10 ', 'Q15S1Mood ',
#                    'Q17S1NormalWork', 'Q19S1Sleep ', 'S1InferenceScoreOutof10 ',
#                    'Q28S1AntiInflamTotal', 'S1OnOpioidAloneorOpioidPlus ',
#                    'S1GrandTotalMedicinesWEIGHTED', 'Q28S1AntiInflamREcode')

list_of_vars2 <- gsub(" ", "", list_of_vars2, fixed = TRUE)

opera2_data <- data.frame(opera_data[list_of_vars2,])
correlation <- data.frame(cor(t(opera2_data), method = 'spearman'))

heatmap.2(cor(t(opera2_data), method = 'spearman'), trace  = 'none',
          cellnote = round(correlation, digits=2),
          notecol = 'black', dendrogram="row", margins = c(15,15),
          notecex=1.5, cexRow = font_size, cexCol = font_size,
          keysize=1, revC = TRUE,key=T, key.title = 'Color Key',
          breaks = seq(-1, 1, length.out = 101), symkey=TRUE,
          lmat = lmat, lwid = c(1,6), lhei = c(1,4.5))
#dev.off()

data_for_pca_opera2 <- opera2_data
rownames(opera2_labels) <- opera2_labels$X
opera2_labels_n <- opera2_labels[,2:length(colnames(opera2_labels))]
opera2_labels_n <- as.factor(opera2_labels_n)
data_for_pca_opera2[nrow(data_for_pca_opera2) + 1,] <- opera2_labels_n


renamed_indices <- rownames(data_for_pca_opera2)
renamed_indices[length(renamed_indices)] = 'Change_In_Interference'#'Labels'

rownames(data_for_pca_opera2) <- renamed_indices

PCA = prcomp(t(opera2_data), scale. = FALSE, rank. = length(rownames(opera2_data)))

m_title = 'Explanation of variance for the Principal Components of the OPERA study dataset for Label_2'
plot_PC_variance(PCA, m_title)

summary(PCA)
summary(PCA)$importance[2,]

pca_data = data.frame('values'=summary(PCA)$importance[2,],'names'=colnames(summary(PCA)$importance))
# #reorder(names,values)
# ggplot(pca_data, aes(x=reorder(names,sort(values)), y=values))+
#   geom_bar(stat = "identity", fill='LightBlue')+
#   ggtitle("Explanation of variance for the Principal Components of the OPERA dataset and Label_1")+
#   scale_x_discrete("Principal Components")+
#   scale_y_continuous("Percentage of explained variance")+
#   theme(axis.text.x=element_text(size=15),
#         axis.text.y=element_text(size=15),
#         axis.title.x=element_text(size=15),
#         axis.title.y=element_text(size=15),
#         plot.title = element_text(size=20))+
#   geom_text(aes(label=round(values,3)), colour="black", size=6 ,vjust=-0.3)

m_title = "Principal Components Analysis (PC1 Vs PC2) of the OPERA study dataset on Label_2"
plot_PCA(PCA, data_for_pca_opera2, m_title)

#biplot(PCA, xlabs=c("x", "o")[as.numeric(tail(data_for_pca,n=1))], cex=.75)
plot_feature_contribution_in_pca(PCA,1)
plot_feature_contribution_in_pca(PCA,2)
plot_feature_contribution_in_pca(PCA,1:2)



######################

list_of_vars3 <- c('NeuroRadic_cat', 'Arthritis_bin',
                  'Int_Relationship ', 'Int_Life_enjoyment',
                  'Narcotic_weight ', 'Grand_Tot_Med_weight',
                  'Narcotic_bin ', 'Tot_Compl_categ')

# list_of_vars3 <- c('Q2S1NeuroRadicTotal ', 'Q1S1ArthritisRecode',
#                    'Q18S1RelationshipsWithOtherPeople ', 'Q20S1EnjoymentofLife',
#                    'Q29S1NarcoticWeight10 ', 'S1GrandTotalMedicinesWEIGHTED',
#                    'Q29S1NarcoticRecode ', 'S1TotalComplaintCategoriesNotOther')

list_of_vars3 <- gsub(" ", "", list_of_vars3, fixed = TRUE)

opera3_data <- data.frame(opera_data[list_of_vars3,])
correlation <- data.frame(cor(t(opera3_data), method = 'spearman'))

heatmap.2(cor(t(opera3_data), method = 'spearman'), trace  = 'none',
          cellnote = round(correlation, digits=2),
          notecol = 'black', dendrogram="row", margins = c(15,15),
          notecex=1.5, cexRow = font_size, cexCol = font_size,
          keysize=1, revC = TRUE, key=T, key.title = 'Color Key',
          breaks = seq(-1, 1, length.out = 101), symkey=TRUE,
          lmat = lmat, lwid = c(1,4), lhei = c(1,4))
#dev.off()

data_for_pca_opera3 <- opera3_data
rownames(opera3_labels) <- opera3_labels$X
opera3_labels_n <- opera3_labels[,2:length(colnames(opera3_labels))]
opera3_labels_n <- as.factor(opera3_labels_n)
data_for_pca_opera3[nrow(data_for_pca_opera3) + 1,] <- opera3_labels_n


renamed_indices <- rownames(data_for_pca_opera3)
renamed_indices[length(renamed_indices)] = 'Change_In_Medicines'#'Labels'

rownames(data_for_pca_opera3) <- renamed_indices

PCA = prcomp(t(opera3_data), scale. = FALSE, rank. = length(rownames(opera3_data)))

m_title = 'Explanation of variance for the Principal Components of the OPERA study dataset for Label_3'
plot_PC_variance(PCA, m_title)

summary(PCA)
summary(PCA)$importance[2,]

pca_data = data.frame('values'=summary(PCA)$importance[2,],'names'=colnames(summary(PCA)$importance))
# #reorder(names,values)
# ggplot(pca_data, aes(x=reorder(names,sort(values)), y=values))+
#   geom_bar(stat = "identity", fill='LightBlue')+
#   ggtitle("Explanation of variance for the Principal Components of the OPERA dataset and Label_1")+
#   scale_x_discrete("Principal Components")+
#   scale_y_continuous("Percentage of explained variance")+
#   theme(axis.text.x=element_text(size=15),
#         axis.text.y=element_text(size=15),
#         axis.title.x=element_text(size=15),
#         axis.title.y=element_text(size=15),
#         plot.title = element_text(size=20))+
#   geom_text(aes(label=round(values,3)), colour="black", size=6 ,vjust=-0.3)

m_title = "Principal Components Analysis (PC1 Vs PC2) of the OPERA study dataset on Label_3"
plot_PCA(PCA, data_for_pca_opera3, m_title)

#biplot(PCA, xlabs=c("x", "o")[as.numeric(tail(data_for_pca,n=1))], cex=.75)
plot_feature_contribution_in_pca(PCA,1)
plot_feature_contribution_in_pca(PCA,2)
plot_feature_contribution_in_pca(PCA,1:2)



######################
list_of_vars4 <- c('Arthritis_bin ', 'Tendinitis_bin',
                   'Grand_Tot_Compl ', 'Pain_other ',
                   'Least_24h', 'Average_pain ', 'Current_pain ', 'Overall_Pain_interference ',
                   'Int_Mood ', 'Int_Walking_ability', 'Int_Work ',
                   'Int_Relationship',
                   'Avg_Interference ', 'OTC_categ ',
                   'AntiInflam_categ')

# list_of_vars4 <- c('Q1S1ArthritisRecode ', 'Q4S1TendinitisRecode',
#                   'S1GrandTotalofAllComplaintsExceptOverallOther ', 'Q7S1 ',
#                   'Q10S1Least', 'Q11S1Average ', 'Q12S1RightNow ', 'Q13S1 ',
#                   'Q15S1Mood ', 'Q16S1WalkingAbility', 'Q17S1NormalWork ',
#                   'Q18S1RelationshipsWithOtherPeople',
#                   'S1InferenceScoreOutof10 ', 'Q27S1OTCTotal ',
#                   'Q28S1AntiInflamTotal')

list_of_vars4 <- gsub(" ", "", list_of_vars4, fixed = TRUE)

opera4_data <- data.frame(opera_data[list_of_vars4,])
correlation <- data.frame(cor(t(opera4_data), method = 'spearman'))

par(cex.main=1, cex.lab=0.7, cex.axis=0.7)


heatmap.2(cor(t(opera4_data), method = 'spearman'), trace  = 'none',
          cellnote = round(correlation, digits=2),
          notecol = 'black', dendrogram="row", margins = c(16,16),
          notecex=1.5, cexRow = font_size, cexCol = font_size,
          keysize=1, revC = TRUE, key=T,key.title = 'Color Key',
          breaks = seq(-1, 1, length.out = 101), symkey=TRUE,
          lmat = lmat, lwid = c(1,4), lhei = c(1,4))#lwid = c(5,15), lhei = c(5,15)

#dev.off()


data_for_pca_opera4 <- opera4_data
rownames(opera4_labels) <- opera4_labels$X
opera4_labels_n <- opera4_labels[,2:length(colnames(opera4_labels))]
opera4_labels_n <- as.factor(opera4_labels_n)
data_for_pca_opera4[nrow(data_for_pca_opera4) + 1,] <- opera4_labels_n


renamed_indices <- rownames(data_for_pca_opera4)
renamed_indices[length(renamed_indices)] = 'Change_In_Complaints'#'Labels'

rownames(data_for_pca_opera4) <- renamed_indices

PCA = prcomp(t(opera4_data), scale. = FALSE, rank. = length(rownames(opera4_data)))

m_title = 'Explanation of variance for the Principal Components of the OPERA study dataset for Label_4'
plot_PC_variance(PCA, m_title)

summary(PCA)
summary(PCA)$importance[2,]

pca_data = data.frame('values'=summary(PCA)$importance[2,],'names'=colnames(summary(PCA)$importance))
# #reorder(names,values)
# ggplot(pca_data, aes(x=reorder(names,sort(values)), y=values))+
#   geom_bar(stat = "identity", fill='LightBlue')+
#   ggtitle("Explanation of variance for the Principal Components of the OPERA dataset and Label_1")+
#   scale_x_discrete("Principal Components")+
#   scale_y_continuous("Percentage of explained variance")+
#   theme(axis.text.x=element_text(size=15),
#         axis.text.y=element_text(size=15),
#         axis.title.x=element_text(size=15),
#         axis.title.y=element_text(size=15),
#         plot.title = element_text(size=20))+
#   geom_text(aes(label=round(values,3)), colour="black", size=6 ,vjust=-0.3)

m_title = "Principal Components Analysis (PC1 Vs PC2) of the OPERA study dataset on Label_4"
plot_PCA(PCA, data_for_pca_opera4, m_title)

#biplot(PCA, xlabs=c("x", "o")[as.numeric(tail(data_for_pca,n=1))], cex=.75)
plot_feature_contribution_in_pca(PCA,1)
plot_feature_contribution_in_pca(PCA,2)
plot_feature_contribution_in_pca(PCA,1:2)




