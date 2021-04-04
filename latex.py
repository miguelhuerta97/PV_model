def latex(PVModule,metrics,errorPV):
  f = open(PVModule+".tex", "w")
  f.write(r"""\documentclass[conference]{IEEEtran}
\usepackage[utf8]{inputenc}
\usepackage{babel}
\IEEEoverridecommandlockouts
%% The preceding line is only needed to identify funding in the first footnote. If that is unneeded, please comment it out.
\usepackage{cite}
\usepackage{url}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{algorithmic}
\usepackage{float}
\usepackage{xcolor}
\usepackage{enumitem}
\usepackage{graphicx}
\usepackage{textcomp}
\usepackage{dsfont}
\usepackage{empheq}

\usepackage[TABBOTCAP, tight]{subfigure}
\usepackage{multirow}

\setlength{\marginparwidth}{2cm}
\usepackage[textsize=scriptsize,shadow]{todonotes}
\newcommand{\tde}[1]{\todo[inline,color=green!40]{#1}}
\newcommand{\txcomment}[1]{{\color{red!80}{[#1]}}}

\def\BibTeX{{\rm B\kern-.05em{\sc i\kern-.025em b}\kern-.08em
    T\kern-.1667em\lower.7ex\hbox{E}\kern-.125emX}}

\graphicspath{{./}}
\newcommand{\unit}[1]{\,\text{\texttt{#1}}}
\newcommand{\unita}[1]{[\texttt{#1}]}
\newcommand{\units}[1]{\,#1}
\newcommand{\uno}{\mathds{1}}

\newcommand{\addfig}[1]{Fig.~\ref{#1}}
\newcommand{\addtab}[1]{Tab.~\ref{#1}}
\newcommand{\addeq}[1]{eq.~\ref{#1}}

\DeclareMathOperator*{\argmin}{arg\,min}
\DeclareMathOperator*{\midoper}{mid}

\graphicspath{{FIgs/Results/%s/}}

\begin{document}

\begin{figure}[!ht]
    \centering
    \includegraphics[height=.82\paperheight]{pv.png}
\end{figure}

\newpage

\begin{figure}[!ht]
    \centering
    \includegraphics[height=.82\paperheight]{iv.png}
\end{figure}

\newpage

\begin{figure*}[!ht]
    \centering
    \subfigure[Error curvas PV]{
    \includegraphics[width=0.8\linewidth]{pvError.png}}
    \subfigure[Error curvas IV]{
    \includegraphics[width=0.8\linewidth]{ivError.png}}
\end{figure*}

\newpage

\begin{figure*}[!ht]
    \centering
    \subfigure[Traking]{
    \includegraphics[width=0.95\linewidth]{tracking.png}}
\end{figure*}

\newpage

\begin{figure*}[!ht]
    \centering
    \begin{tabular}{|c|p{0.11\linewidth}|p{0.11\linewidth}|}
        \hline\hline
        \multicolumn{1}{|c|}{Variable} 
        &\multicolumn{1}{c|}{Mean}
        &\multicolumn{1}{c|}{Std}\\\hline\hline
        S (W/m$^2$)              & %6.3f & %6.3f \\
        T (°C)                   & %6.3f & %6.3f \\
        b (1/V)                  & %6.3f & %6.3f \\
        I$_\text{L}$ (A)         & %6.3f & %6.3f \\
        log(I$_\text{0}$)        & %6.3f & %6.3f \\
        R$_\text{s}$ ($\Omega$)  & %6.3f & %6.3f \\
        G$_\text{p}$ (S)         & %6.3f & %6.3f \\
        \hline\hline
      \end{tabular}
    
    \vspace{1em}
    \subfigure[Params]{
    \includegraphics[width=0.9\linewidth]{params.png}}

\end{figure*}

\newpage

\begin{figure*}[!ht]
    \centering
    \subfigure[Error en el conjunto de testeo]{
    \includegraphics[width=0.9\linewidth]{TestError1.png}}

  \vspace{1em}

  \begin{tabular}{|c|p{0.07\linewidth}|p{0.07\linewidth}|p{0.07\linewidth}|p{0.07\linewidth}|p{0.07\linewidth}|p{0.07\linewidth}|p{0.07\linewidth}|p{0.07\linewidth}|}
    \hline\hline
    \multirow{2}{*}{Variable} 
    &\multicolumn{2}{c|}{De Soto}
    &\multicolumn{2}{c|}{Dobos}
    &\multicolumn{2}{c|}{Boyd}  
    &\multicolumn{2}{c|}{Proposed}\\\cline{2-9}
    &\multicolumn{1}{c|}{Mean}
    &\multicolumn{1}{c|}{Std}
    &\multicolumn{1}{c|}{Mean}
    &\multicolumn{1}{c|}{Std}
    &\multicolumn{1}{c|}{Mean}
    &\multicolumn{1}{c|}{Std}
    &\multicolumn{1}{c|}{Mean}
    &\multicolumn{1}{c|}{Std}\\\hline\hline
    I$_\text{sc}$ & %6.3f & %6.3f & %6.3f & %6.3f & %6.3f & %6.3f & %6.3f& %6.3f\\
    P$_\text{mp}$ & %6.3f & %6.3f & %6.3f & %6.3f & %6.3f & %6.3f & %6.3f& %6.3f\\
    I$_\text{mp}$ & %6.3f & %6.3f & %6.3f & %6.3f & %6.3f & %6.3f & %6.3f& %6.3f\\
    V$_\text{mp}$ & %6.3f & %6.3f & %6.3f & %6.3f & %6.3f & %6.3f & %6.3f& %6.3f\\
    V$_\text{oc}$ & %6.3f & %6.3f & %6.3f & %6.3f & %6.3f & %6.3f & %6.3f& %6.3f\\
    \hline\hline
  \end{tabular}
\end{figure*}

\newpage

\begin{figure*}[!ht]
  \centering

  \subfigure[Error en el conjunto de testeo]{
    \includegraphics[width=0.9\linewidth]{TestError2.png}}

  \vspace{1em}

  \begin{tabular}{|c|p{0.06\linewidth}|p{0.06\linewidth}|p{0.06\linewidth}|p{0.06\linewidth}|}
  \hline\hline
  \multirow{2}{*}{Variable} &\multicolumn{4}{c|}{MAE}\\\cline{2-5}
      &\multicolumn{1}{c|}{De Soto}
      &\multicolumn{1}{c|}{Dobos}
      &\multicolumn{1}{c|}{Boyd}  
      &\multicolumn{1}{c|}{Proposed}\\\hline\hline
      Error         & %6.3f & %6.3f & %6.3f& %6.3f\\
      I$_\text{sc}$ & %6.3f & %6.3f & %6.3f& %6.3f\\
      P$_\text{mp}$ & %6.3f & %6.3f & %6.3f& %6.3f\\
      I$_\text{mp}$ & %6.3f & %6.3f & %6.3f& %6.3f\\
      V$_\text{mp}$ & %6.3f & %6.3f & %6.3f& %6.3f\\
      V$_\text{oc}$ & %6.3f & %6.3f & %6.3f& %6.3f\\
  \hline\end{tabular}\hspace{1em}
  \begin{tabular}{|c|p{0.06\linewidth}|p{0.06\linewidth}|p{0.06\linewidth}|p{0.06\linewidth}|}
  \hline\hline
  \multirow{2}{*}{Variable} &\multicolumn{4}{c|}{MSE}\\\cline{2-5}
      &\multicolumn{1}{c|}{De Soto}
      &\multicolumn{1}{c|}{Dobos}
      &\multicolumn{1}{c|}{Boyd}  
      &\multicolumn{1}{c|}{Proposed}\\\hline\hline
      Error         & %6.3f & %6.3f & %6.3f& %6.3f\\
      I$_\text{sc}$ & %6.3f & %6.3f & %6.3f& %6.3f\\
      P$_\text{mp}$ & %6.3f & %6.3f & %6.3f& %6.3f\\
      I$_\text{mp}$ & %6.3f & %6.3f & %6.3f& %6.3f\\
      V$_\text{mp}$ & %6.3f & %6.3f & %6.3f& %6.3f\\
      V$_\text{oc}$ & %6.3f & %6.3f & %6.3f& %6.3f\\
  \hline\end{tabular}\vspace{2em}
  \begin{tabular}{|c|p{0.06\linewidth}|p{0.06\linewidth}|p{0.06\linewidth}|p{0.06\linewidth}|}
  \hline\hline
  \multirow{2}{*}{Variable} &\multicolumn{4}{c|}{MSLE}\\\cline{2-5}
      &\multicolumn{1}{c|}{De Soto}
      &\multicolumn{1}{c|}{Dobos}
      &\multicolumn{1}{c|}{Boyd}  
      &\multicolumn{1}{c|}{Proposed}\\\hline\hline
      Error         & %6.3f & %6.3f & %6.3f& %6.3f\\
      I$_\text{sc}$ & %6.3f & %6.3f & %6.3f& %6.3f\\
      P$_\text{mp}$ & %6.3f & %6.3f & %6.3f& %6.3f\\
      I$_\text{mp}$ & %6.3f & %6.3f & %6.3f& %6.3f\\
      V$_\text{mp}$ & %6.3f & %6.3f & %6.3f& %6.3f\\
      V$_\text{oc}$ & %6.3f & %6.3f & %6.3f& %6.3f\\
  \hline\end{tabular}\hspace{1em}
  \begin{tabular}{|c|p{0.06\linewidth}|p{0.06\linewidth}|p{0.06\linewidth}|p{0.06\linewidth}|}
  \hline\hline
  \multirow{2}{*}{Variable} &\multicolumn{4}{c|}{MAPE}\\\cline{2-5}
      &\multicolumn{1}{c|}{De Soto}
      &\multicolumn{1}{c|}{Dobos}
      &\multicolumn{1}{c|}{Boyd}  
      &\multicolumn{1}{c|}{Proposed}\\\hline\hline
      Error         & %6.3f & %6.3f & %6.3f& %6.3f\\
      I$_\text{sc}$ & %6.3f & %6.3f & %6.3f& %6.3f\\
      P$_\text{mp}$ & %6.3f & %6.3f & %6.3f& %6.3f\\
      I$_\text{mp}$ & %6.3f & %6.3f & %6.3f& %6.3f\\
      V$_\text{mp}$ & %6.3f & %6.3f & %6.3f& %6.3f\\
      V$_\text{oc}$ & %6.3f & %6.3f & %6.3f& %6.3f\\
  \hline\end{tabular}
\end{figure*}

\end{document}
  """%(PVModule,
      *[k1 for k in ['S','T','b','IL','I0','Rs','Gp'] for k1 in ParamsMeanStd[k]],
      *[k2 for k1 in ['Isc', 'Pmp', 'Imp', 'Vmp', 'Voc'] for k in ['De Soto', 'Dobos', 'Boyd','Neural Network'] for k2 in metrics[k][k1]],
      *[errorPV[k][k1][k2] for k2 in ['MAE', 'MSE', 'MSLE', 'MAPE'] for k1 in ['Total', 'Isc', 'Pmp', 'Imp', 'Vmp', 'Voc']for k in ['De Soto', 'Dobos', 'Boyd','Neural Network']],
       ))
  f.close()
