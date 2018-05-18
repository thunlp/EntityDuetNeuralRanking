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



| |Testing-SAME | Testing-DIFF | Testing-RAW|
| --------   | --------   | --------   | --------  |
|K-NRM|0.2645|0.4197|0.3000|0.4228|0.3447|


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
