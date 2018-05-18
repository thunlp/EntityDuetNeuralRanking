# EntityDuetNeuralRanking
There are source codes for Entity-Duet Neural Ranking Model (EDRM).
![model](https://github.com/thunlp/EntityDuetNeuralRanking/blob/master/model.png)

## Baselines

There are code for our main baseline K-NRM and Conv-KNRM:

- [End-to-end neural ad-hoc ranking with kernel pooling](http://www.cs.cmu.edu/afs/cs/user/cx/www/papers/K-NRM.pdf)
- [Convolutional Neural Networks for Soft-Matching N-Grams in Ad-hoc Search](http://www.cs.cmu.edu/~zhuyund/papers/WSDM_2018_Dai.pdf)


## EDRM

There are codes for our work based on Conv-KNRM.


## Results

The ranking results. All results are in trec format.

\begin{table*}[h]
\centering
 \caption{Ranking accuracy of EDRM-KNRM, EDRM-CKNRM and baseline methods. Relative performances compared with K-NRM are in percentages. $\dagger$, $\ddagger$, $\mathsection$, $\mathparagraph$, $*$ indicate statistically significant improvements over DRMM$^{\dagger}$, CDSSM$^{\ddagger}$, MP$^{\mathsection}$, K-NRM$^{\mathparagraph}$ and Conv-KNRM$^{*}$ respectively.}\label{table.overall_acc}
 \resizebox{\textwidth}{23mm}{
 \begin{tabular}{l|lr|lr|lr|lr|lr}
  \hline
  &	\multicolumn{4}{c|}{\textbf{Testing-SAME}}	&	\multicolumn{4}{c|}{\textbf{Testing-DIFF}} & \multicolumn{2}{c}{\textbf{Testing-RAW}} \\ \hline
    \textbf{Method}	&	\multicolumn{2}{c|}{\textbf{NDCG@1}}	&	\multicolumn{2}{c|}{\textbf{NDCG@10}}   &	\multicolumn{2}{c|}{\textbf{NDCG@1}}	&	\multicolumn{2}{c|}{\textbf{NDCG@10}} & \multicolumn{2}{c}{\textbf{MRR}}\\ \hline
\texttt{BM25}	
& ${0.1422}$ & $ -46.24\%  $
& ${0.2868}$ & $ -31.67\%  $
& ${0.1631}$ & $ -45.63\%  $
& ${0.3254}$ & $ -23.04\%  $
& ${0.2280}$ & $ -33.86\%  $  \\

\texttt{RankSVM}	
& ${0.1457}$ & $ -44.91\%  $
& ${0.3087}$ & $ -26.45\%  $
& ${0.1700}$ & $ -43.33\%  $
& ${0.3519}$ & $ -16.77\%  $
& ${0.2241}$ & $ -34.99\%  $  \\


\texttt{Coor-Ascent}	
& ${0.1594}$ & $ -39.74\%  $
& ${0.3547}$ & $ -15.49\%  $
& ${0.2089}$ & $ -30.37\%  $
& ${0.3775}$ & $ -10.71\%  $
& ${0.2415}$ & $ -29.94\%  $  \\ \hline



\texttt{DRMM}
& ${0.1367}$ & $ -48.34\%  $
& ${0.3134}$ & $ -25.34\%  $
& ${0.2126}^{\ddagger }$ & $ -29.14\%  $
& ${0.3592}^{\mathsection }$ & $ -15.05\%  $
& ${0.2335}$ & $ -32.26\%  $  \\


\texttt{CDSSM}	
& ${0.1441}$ & $ -45.53\%  $
& ${0.3329}$ & $ -20.69\%  $
& ${0.1834}$ & $ -38.86\%  $
& ${0.3534}$ & $ -16.41\%  $
& ${0.2310}$ & $ -33.00\%  $  \\



\texttt{MP}	
& ${0.2184}^{\dagger \ddagger }$ & $ -17.44\%  $
& ${0.3792}^{\dagger \ddagger }$ & $ -9.67\%  $
& ${0.1969}$ & $ -34.37\%  $
& ${0.3450}$ & $ -18.40\%  $
& ${0.2404}$ & $ -30.27\%  $ \\

\texttt{K-NRM}	
& $0.2645$ & --  
& $0.4197$ & --
& $0.3000$ & --  
& $0.4228$ & --
& $0.3447$ & --   \\


\texttt{Conv-KNRM}	
& ${0.3357}^{\dagger \ddagger \mathsection \mathparagraph }$ & $ +26.90\%  $
& ${0.4810}^{\dagger \ddagger \mathsection \mathparagraph }$ & $ +14.59\%  $
& ${0.3384}^{\dagger \ddagger \mathsection \mathparagraph }$ & $ +12.81\%  $
& ${0.4318}^{\dagger \ddagger \mathsection }$ & $ +2.14\%  $
& ${0.3582}^{\dagger \ddagger \mathsection }$ & $ +3.91\%  $\\
\hline

\texttt{EDRM-KNRM}	
& ${0.3096}^{\dagger \ddagger \mathsection \mathparagraph }$ & $ +17.04\%  $
& ${0.4547}^{\dagger \ddagger \mathsection \mathparagraph }$ & $ +8.32\%  $
& ${0.3327}^{\dagger \ddagger \mathsection \mathparagraph }$ & $ +10.92\%  $
& ${0.4341}^{\dagger \ddagger \mathsection \mathparagraph }$ & $ +2.68\%  $
& ${0.3616}^{\dagger \ddagger \mathsection \mathparagraph }$ & $ +4.90\% $  \\


\texttt{EDRM-CKNRM}	
& $\textbf{0.3397}^{\dagger \ddagger \mathsection \mathparagraph }$ & $ +28.42\%  $
& $\textbf{0.4821}^{\dagger \ddagger \mathsection \mathparagraph }$ & $ +14.86\%  $
& $\textbf{0.3708}^{\dagger \ddagger \mathsection \mathparagraph * }$ & $ +23.60\%  $
& $\textbf{0.4513}^{\dagger \ddagger \mathsection \mathparagraph * }$ & $ +6.74\%  $
& $\textbf{0.3892}^{\dagger \ddagger \mathsection \mathparagraph * }$ & $ +12.90\%  $ \\

\hline 
\end{tabular}}
\end{table*}

## Citation
```
Entity-Duet Neural Ranking: Understanding the Role of Knowledge Graph Semantics in Neural Information Retrieval. Zhenghao Liu, Chenyan Xiong, Maosong Sun and Zhiyuan Liu.
```

## Copyright

All Rights Reserved.


## Contact
If you have questions, suggestions and bug reports, please email liuzhenghao0819@gmail.com.
