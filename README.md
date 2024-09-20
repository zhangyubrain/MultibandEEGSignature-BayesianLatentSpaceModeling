# Multiband EEG Signature - Bayesian Latent Space Modeling

## **Overview**
Here we present a combined itEMD and SBLEST framework that uses multiband (multi-intrinsic mode function) information for the purpose of predicting treatment outcome in psychiatric disorders. The framework begins with decomposing EEG data into instrinsic mode functions (IMF) via itEMD. The IMFs and the residual signal is combined into a diagonal block matrix which serves as the feature input into the modified SBLEST model. After training, we obtain both treatment prediction and treatment specific spatial filters respective to each IMF. 

This framework's application has been demonstrated in predicting rTMS treatment outcomes for those diagnosed with Major Depressive Disorder (MDD). For more details, please see the accompanying paper.

For more information on itEMD, refer to the following paper:
Marco S. Fabus, Andrew J. Quinn, Catherine E. Warnaby, and Mark W. Woolrich (2021). Automatic decomposition of electroptysiological data into distinct nonsinusoidal oscillatory modes. Journal of Neurophysiology 2021 126:5, 1670-1684. https://doi.org/10.1152/jn.00315.2021.

For more information on SBLEST, refer to the following paper:
W. Wang, F. Qi, D. Wipf, C. Can, T. Yu, Z. Gu, Y. Li, Z. Yu, W. Wu. Sparse Bayesian Learning for End-to-End EEG Decoding, IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 45, no. 12, pp. 15632-15649, 2023.
