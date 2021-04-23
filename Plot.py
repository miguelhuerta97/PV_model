class FunctionPlotting:
  def __init__(self, df, idxTest, SList=[100, 200, 400, 600, 800, 1000]): 
    plt.rcParams.update({
        'lines.linewidth' : 2,
        'legend.fontsize' : 10,
        'font.size'       : 10,
        'axes.titlesize'  : 14, 
        'axes.labelsize'  : 14,
        'xtick.labelsize' : 12,
        'ytick.labelsize' : 12,
        'figure.dpi'      : 80,
        'figure.facecolor': 'w',
        'figure.edgecolor': 'k',
        'figure.figsize'  : [15,4],
        })
    self.idxTest = idxTest
    self.SList, self.TList, self.eps = SList, [25, 50], 1e-18
    self.TimeSeries = df.drop(df.columns[np.arange(0, 41)], axis=1)
    self.df = df.drop(df.columns[np.append([2, 4, 6,  8, 10, 12], np.arange(14, len(df.columns)))], axis=1)
    self.X, self.Y = [self.df[self.df.columns[k]].to_numpy(dtype='float32') for k in [[1,2],[3,4,5,6,7]]]
    aux = self.df[self.df.columns[[0]][0]].str.split("T", n=1, expand=True)
    self.df.insert(0, "yyyy-mm-dd", aux[0])
    self.df[self.df.columns[1]] = aux[1].str.split(":", n=-1, expand = True).apply(lambda x: x[0]+':'+x[1], axis=1)
    self.df.rename(columns={self.df.columns[n+1]:k for n,k in enumerate(['hh:mm', 'S', 'T'])}, inplace=True)
    self.MAE  = tf.keras.losses.MeanAbsoluteError()
    self.MAPE = tf.keras.losses.MeanAbsolutePercentageError()
    self.MSE  = tf.keras.losses.MeanSquaredError()
    self.MSLE = tf.keras.losses.MeanSquaredLogarithmicError()
  def ModelParams(self, x, params, m):
    try:
      return PVModel(x=x, data=params, model=m)
    except:
      pass
  def DNNParams(self, x, model):   
    try:
      return modelPV.DNNParams(x, model)
    except:
      pass

  def PlotLoss(self, loss, ylabel='Mean square error', LegendPos=[0.8, 0.8]):
    if len(loss[0])!=0:
      fig = plt.figure(figsize=(10, 15))
      gs  = gridspec.GridSpec(3, ncols=3, width_ratios=[5, 1, 5], wspace=0.03, hspace=0.3)
      for num, [label, conty, contx] in enumerate([['Custom Loss', 0, 0], ['Isc', 2, 0], ['Pmp', 0, 2], 
                                                   ['Imp', 1, 0], ['Vmp', 1, 2], ['Voc', 2, 2],]):
        ax = fig.add_subplot(gs[conty, contx])
        ax.plot(loss[0][:, num], ".b", label='Training')
        ax.plot(loss[1][:, num], ".g", label='Validation')
        ax.axhline(y=loss[2][num], color="k", linestyle="-.", label="Test")
        ax.set_ylabel(ylabel)
        ax.set_xlabel('epoch (-)')
        ax.set_xlim([0, len(loss[0][:,num])])
        ax.set_title(label)
      handles, labels = ax.get_legend_handles_labels()
      fig.legend(handles, labels, loc=1, bbox_to_anchor=LegendPos)
      plt.show() 
  def Boxplot(self, X, Y, N=False):
    np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
    fig = plt.figure(figsize=(15, 10), dpi=80, facecolor='w', edgecolor='k')
    gs  = gridspec.GridSpec(nrows=3, ncols=9, width_ratios=[5, 1, 5, 1, 5, 1, 5, 1, 5], height_ratios=[3,1,9], wspace=0.1, hspace=0.1)
    for k in range(2):
      if k:
        ax = plt.subplot(gs[0, 5:9 ])
      else:
        ax = plt.subplot(gs[0, 0:4 ])
      ax.boxplot([x[:,k] for x in X], vert=False)
      ax.yaxis.set_ticklabels(['total', 'train', 'val', 'test'])
      if N:
        ax.set_title(['Irradiance (-)', 'Temperature (-)'][k])
      else:
        ax.set_title(['Irradiance (W/m$^2$)', 'Temperature (°C)'][k])
    for k in range(5):
      ax = plt.subplot(gs[2, 2*k])
      ax.boxplot([y[:,k] for y in Y])
      ax.xaxis.set_ticklabels(['total', 'train', 'val', 'test'], rotation=70)
      ax.set_title(['Isc (A)', 'Pmp (W)','Imp (A)', 'Vmp (V)', 'Voc (V)'][k])
    plt.show() 

  def SearchCurve(self, T1, S1, Filter=[False, 100]):
    df = self.df.iloc[self.idxTest]
    TimeSeries = self.TimeSeries.iloc[self.idxTest]
    # idx = np.where((df[df.columns[[3]]]==[T1]).to_numpy()==True)[0]
    idx = (np.where(( df[df.columns[[3]]]>=[T1*0.9]) & (df[df.columns[[3]]]<=[T1*1.1])))[0]
    aux = np.square(df[df.columns[[2]]].iloc[idx].to_numpy()-S1)
    idx = idx[np.where(aux==aux.min())[0]][0]
    current = TimeSeries.iloc[idx][np.arange(1, TimeSeries.iloc[idx][0]+1, dtype=int)].to_numpy()
    voltage = TimeSeries.iloc[idx][np.arange(TimeSeries.iloc[idx][0]+1, 2*TimeSeries.iloc[idx][0]+1, dtype=int)].to_numpy()
    return [current, voltage, df.iloc[idx]['S'], df.iloc[idx]['T']]

  
  
  def TestError1(self, xTest, yTest, params={}, model=None, LegendPos=[0.6, 0.89], 
              labelDNN='Neural Network', save=True):
    yData, metrics = [], {}
    for m in params: # Models
      Rs, Gp, IL, I0, b = [tf.reshape(k, [k.shape[0], 1]).numpy().astype('float64') for k in self.ModelParams(xTest, params, m)]
      [Isc, Vsc, Imp, Vmp, Ioc, Voc] = np.hsplit(PVPredict().predict(Rs, Gp, IL, I0, b).numpy().astype('float64'), 6)
      yData.append(np.hstack([Isc, Imp*Vmp, Imp, Vmp, Voc]))
    try: # Neural network
      Rs, Gp, IL, I0, b  = self.DNNParams(xTest, model)      
      [IscDNN, VscDNN, ImpDNN, VmpDNN, IocDNN, VocDNN] = np.hsplit(PVPredict().predict(Rs, Gp, IL, I0, b).numpy().astype('float64'), 6)
      yDNN = np.hstack([IscDNN, ImpDNN*VmpDNN, ImpDNN, VmpDNN, VocDNN])
    except:
      [IscDNN, VscDNN, ImpDNN, VmpDNN, IocDNN, VocDNN] = np.ones((6,1))*np.nan
      pass
    fig = plt.figure(figsize=(15, 10))
    gs  = gridspec.GridSpec(2, ncols=11, figure=fig, hspace=0.3, wspace=0.03)    
    positionsBox, labelBox, colors = 0, [], ["#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    labels, idx = ['Isc (A)', 'Pmp (W)', 'Imp (A)','Vmp (V)','Voc (V)'], 0
    for [ax1, ax2], label, DNN, in [[[fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1:3])],  'Pmp (W)', ImpDNN*VmpDNN],
                                    [[fig.add_subplot(gs[0, 4]), fig.add_subplot(gs[0, 5:7])],  'Imp (A)', ImpDNN],
                                    [[fig.add_subplot(gs[0, 8]), fig.add_subplot(gs[0, 9:11])], 'Vmp (V)', VmpDNN],
                                    [[fig.add_subplot(gs[1, 2]), fig.add_subplot(gs[1, 3:5])],  'Isc (A)', IscDNN],
                                    [[fig.add_subplot(gs[1, 6]), fig.add_subplot(gs[1, 7:9])],  'Voc (V)', VocDNN] ]:
      idx= labels.index(label)
      for model in params:
        error = ((yTest[:,idx]-yData[model][:, idx])/yTest[:,idx])*100
        try: P0 = metrics[params[model]['name']]
        except: P0 = {}
        P0[label[:-4]]=[error.mean(), error.std()]
        metrics[params[model]['name']] = P0
        box = ax1.boxplot(error, vert=True, positions=[positionsBox], patch_artist=True, widths=0.4)
        plt.setp(box["boxes"], facecolor=colors[positionsBox])
        plt.setp(box['medians'], color='black')
        labelBox.append('\n\n'+params[model]['label'])
        ax2.hist(error*100, 50, weights=np.ones(len(error))/len(error), density=False, alpha=0.75, 
                 orientation="horizontal", label=params[model]['name'], color=colors[positionsBox])
        positionsBox+=1
      if not(np.all(np.isnan(DNN))):
        error = ((yTest[:,idx]-DNN[:,0])/yTest[:,idx])*100
        try: P0 = metrics[labelDNN]
        except: P0 = {}
        P0[label[:-4]]=[error.mean(), error.std()]
        metrics[labelDNN] = P0
        box = ax1.boxplot(error, vert=True, positions=[positionsBox], patch_artist=True, widths=0.4)
        plt.setp(box["boxes"], facecolor=colors[positionsBox])
        plt.setp(box['medians'], color='black')
        labelBox.append('NN')
        ax2.hist(error*100, 50, weights=np.ones(len(error))/len(error), density=False, alpha=0.75, orientation="horizontal", 
                 label=labelDNN, color=colors[positionsBox])
        positionsBox+=1
      positionsBox=0
      ax1.set_xticklabels(labelBox), 
      ax1.set_ylabel('Prediction error (%)'), 
      ax1.xaxis.tick_top()
      ax2.xaxis.set_major_formatter(PercentFormatter(1))      
      ax2.yaxis.set_ticks([]), 
      ax2.set_xlabel('Frequency (%)'), 
      ax2.set_title(label)
    handles, labels = ax2.get_legend_handles_labels()
    fig.legend(handles, labels, loc=1, bbox_to_anchor=LegendPos, ncol=len(labels))
    if save: plt.savefig('TestError1.png', bbox_inches = 'tight')
    plt.show()
    return metrics




  def TestError2(self, xTest, yTest, params={}, model=None, LegendPos=[0.58, 1.18], plotBar=True, save=True, labelDNN='Neural Network', ShowTable=True):  
    ErrorData, yData = {}, []
    for m in params: # Models
      Rs, Gp, IL, I0, b = [tf.reshape(k, [k.shape[0], 1]).numpy().astype('float64') for k in self.ModelParams(xTest, params, m)]
      [Isc, Vsc, Imp, Vmp, Ioc, Voc] = np.hsplit(PVPredict().predict(Rs, Gp, IL, I0, b).numpy().astype('float64'), 6)
      yData.append(np.hstack([Isc, Imp*Vmp, Imp, Vmp, Voc]))
      for idx1, var in enumerate(['Isc', 'Pmp','Imp', 'Vmp', 'Voc']):
        try: P0 = ErrorData[params[m]['name']]
        except: P0 = {}
        P0[var]={'MAE' : self.MAE( yTest[:, idx1], yData[m][:, idx1]).numpy(),
                 'MSE' : self.MSE( yTest[:, idx1], yData[m][:, idx1]).numpy(),
                 'MAPE': self.MAPE(yTest[:, idx1], yData[m][:, idx1]).numpy(),
                 'MSLE': self.MSLE(yTest[:, idx1], yData[m][:, idx1]).numpy()}
        ErrorData[params[m]['name']] = P0
      try: P0 = ErrorData[params[m]['name']]
      except: P0 = {}
      P0['Total']={'MAE' : self.MAE( yTest, yData[m]).numpy(),
                   'MSE' : self.MSE( yTest, yData[m]).numpy(),
                   'MAPE': self.MAPE(yTest, yData[m]).numpy(),
                   'MSLE': self.MSLE(yTest, yData[m]).numpy()}
      ErrorData[params[m]['name']] = P0
    try: # Neural network
      Rs, Gp, IL, I0, b  = self.DNNParams(xTest, model)      
      [IscDNN, VscDNN, ImpDNN, VmpDNN, IocDNN, VocDNN] = np.hsplit(PVPredict().predict(Rs, Gp, IL, I0, b).numpy().astype('float64'), 6)
      yDNN = np.hstack([IscDNN, ImpDNN*VmpDNN, ImpDNN, VmpDNN, VocDNN])
      for idx1, var in enumerate(['Isc', 'Pmp','Imp', 'Vmp', 'Voc']):
        try: P0 = ErrorData[labelDNN]
        except: P0 = {}
        P0[var]={'MAE' : self.MAE( yTest[:, idx1], yDNN[:, idx1]).numpy(),
                 'MSE' : self.MSE( yTest[:, idx1], yDNN[:, idx1]).numpy(),
                 'MAPE': self.MAPE(yTest[:, idx1], yDNN[:, idx1]).numpy(),
                 'MSLE': self.MSLE(yTest[:, idx1], yDNN[:, idx1]).numpy()}
        ErrorData[labelDNN] = P0
      try: P0 = ErrorData[labelDNN]
      except: P0 = {}
      P0['Total']={'MAE' : self.MAE( yTest, yDNN).numpy(),
                   'MSE' : self.MSE( yTest, yDNN).numpy(),
                   'MAPE': self.MAPE(yTest, yDNN).numpy(),
                   'MSLE': self.MSLE(yTest, yDNN).numpy()}
      ErrorData[labelDNN] = P0                   
    except: [IscDNN, VscDNN, ImpDNN, VmpDNN, IocDNN, VocDNN] = np.ones((6,1))*np.nan
    if plotBar:
      width =  0.35/2  
      x = np.arange(6)
      fig = plt.figure(figsize=(15, 4))
      gs  = gridspec.GridSpec(1, ncols=4, figure=fig, wspace=0.4, hspace=0.55)
      for contx, metric in enumerate(['MAE', 'MSE', 'MSLE', 'MAPE']):
        ax = fig.add_subplot(gs[0, contx])
        ax.bar(x, x*np.nan,  width)
        for num, model in enumerate(ErrorData):
          plotData = [ErrorData[model][var][metric] for var in ['Total','Isc', 'Pmp','Imp', 'Vmp', 'Voc']]
          ax.bar(x + width*[-3/2, -1/2, 1/2, 3/2][num], plotData,  width, label=model)
        ax.set_title('\n'+metric)
        ax.set_xlabel('Targets')
        ax.set_xticks(x)
        ax.set_xticklabels(['Total','Isc', 'Pmp','Imp', 'Vmp', 'Voc'], rotation=45)
      handles, labels = ax.get_legend_handles_labels()
      fig.legend(handles, labels, loc=1, bbox_to_anchor=LegendPos, ncol=len(labels))
      fig.suptitle('\n')
      if save: plt.savefig('TestError2.png', bbox_inches = 'tight')
      plt.show()
    if ShowTable:
      print("""
===================================================================================================================================================================================================
        |                                                                                  Error                                                                                     
        |--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
   Var  |                    MAE                    |                   MSE                     |                     MSLE                      |                          MAPE                      
        |--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        | De Soto  |  Dobos   |   Boyd   |    NN    | De Soto  |  Dobos   |   Boyd   |    NN    | De Soto   |  Dobos    |   Boyd    |    NN     |  De Soto   |   Dobos    |    Boyd    |    NN    
===================================================================================================================================================================================================""")
      for var in ['Total','Isc', 'Pmp','Imp', 'Vmp', 'Voc']:
        plotData = []
        for metric in ['MAE', 'MSE', 'MSLE', 'MAPE']:
          for model in ErrorData:
            plotData += [ErrorData[model][var][metric]]
          if labelDNN not in list(ErrorData.keys()):
            plotData += [np.nan]
        print("{:6s}| {:>8.3f} | {:>8.3f} | {:>8.3f} | {:>8.3f} | {:>8.3f} | {:>8.3f} | {:>8.3f} | {:>8.3f} | {:>8.3e} | {:>8.3e} | {:>8.3e} | {:>8.3e} | {:>10.3f} | {:>10.3f} | {:>10.3f} | {:>10.3f}".format(var, *plotData))
    return ErrorData
	
	
	
	
	
	
  def Curves(self, curve, params={}, model=None, Filter=[False, 100], LegendPos=[0.8, 0.88], 
            addElement=True, showE=True, yPos=[0.95, 0.15], labelDNN='Neural Network', save=True):
    fig = plt.figure(figsize=(10, len(self.SList)*5))
    gs  = gridspec.GridSpec(nrows=len(self.SList), ncols=2, figure=fig, width_ratios=[1, 1], wspace=0.1, hspace=0.1)
    for contx, T1 in enumerate(self.TList):
      for conty, S1 in enumerate(self.SList):
        ax =  plt.subplot(gs[conty, contx])
        try:
          [current, voltage, S, T] = self.SearchCurve(T1, S1, Filter=Filter)
          if curve=='pv': # Experimental curve
            yData = voltage*current 
            if contx==0: ax.set_ylabel('$p_{pv}$ (W)')
          elif curve=='iv':
            yData=current
            if contx==0:
              ax.set_ylabel('$i_{pv}$ (A)')
          if showE:
            ax.plot(voltage, yData, label='Experimental curve')
          if addElement:
            voltage = np.hstack((voltage, voltage[-1]*np.linspace(1, 1.3, 10)))
          for m in params:# Models      
            Rs, Gp, IL, I0, b = self.ModelParams(np.array([[S, T]]), params, m)
            Ipv = PVPredict().fun_Ipv(Rs, Gp, IL, I0, b, voltage).numpy()
            if curve=='pv':   yData=voltage*Ipv
            elif curve=='iv': yData=Ipv
            ax.plot(voltage, yData, label=params[m]['name'], ls='--')
          try: # Neural network
            Rs, Gp, IL, I0, b  = self.DNNParams(np.array([[S, T]])  , model)
            Ipv_DNN = PVPredict().fun_Ipv(Rs, Gp, IL, I0, b, voltage).numpy()[0,:]
            if curve=='pv':   yDNN=voltage*Ipv_DNN
            elif curve=='iv': yDNN=Ipv_DNN
            ax.plot(voltage, yDNN, label=labelDNN, ls='--')
          except: pass
          if curve=='pv': yx = yPos[0]
          else: yx = yPos[1]
          ax.text(0.05, yx, "$S=%.2f$(W/m$^2$)\n$T=%.2f$(°C)" % (S, T), transform=ax.transAxes, 
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        except: ax.axis('off')
        ax.grid(alpha=0.75)
        if contx: ax.axes.yaxis.set_ticklabels([])
        if conty//(len(self.SList)-1): ax.set_xlabel('$v_{pv}$(V)')
        else: ax.axes.xaxis.set_ticklabels([])   
    for k in range(len(self.SList)):
      Ymax = np.max([k1.get_ylim()[1] for k1 in [fig.axes[k], fig.axes[k+6]]])
      for ax in [fig.axes[k], fig.axes[k+6]]:
        ax.set_ylim([0, Ymax])
    for k in range(len(self.TList)):
      Xmax = np.ceil(np.max([np.max([line.get_data()[0][np.argmin(np.abs(line.get_data()[1])[5:])]*1.03 for line in k1.get_lines()]) for k1 in fig.axes[k*6:(k+1)*6]]))
      for ax in fig.axes[k*6:(k+1)*6]:
        ax.set_xlim([0, Xmax])
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc=1, bbox_to_anchor=LegendPos, ncol=len(labels))
    fig.suptitle('\n')
    plt.subplots_adjust(top=0.97)
    if save: plt.savefig(curve+'.png', bbox_inches = 'tight')
    plt.show()
	
	
	
	
	
  def CurveError(self, curve, params={}, model=None, Filter=[False, 100], 
                 LegendPos=[0.565, 0.96], ShowTable=True, plotBar=True,
                 labelDNN='Neural Network', save=True):  
    ErrorData = {}
    for T1 in self.TList:
      for S1 in self.SList:
        try:
          current, voltage, S, T = self.SearchCurve(T1, S1, Filter=Filter)
          # Experimental curve
          if curve=='pv': yReal = voltage*current
          elif curve=='iv': yReal=current
          for m in params:# Models
            Rs, Gp, IL, I0, b = self.ModelParams(np.array([[S, T]]), params, m)
            Ipv = PVPredict().fun_Ipv(Rs, Gp, IL, I0, b, voltage).numpy()
            if curve=='pv': yData=voltage*Ipv
            elif curve=='iv': yData=Ipv
            try: P0 = ErrorData[params[m]['name']]
            except: P0 = {}
            P0[(S,T)]={'MAE' : self.MAE(yReal,  yData).numpy(),
                       'MSE' : self.MSE(yReal,  yData).numpy(),
                       'MAPE': self.MAPE(yReal, yData).numpy(),
                       'MSLE': self.MSLE(yReal, yData).numpy()} 
            ErrorData[params[m]['name']] = P0     
          try:# Neural network
            Rs, Gp, IL, I0, b  = self.DNNParams(np.array([[S, T]]), model)
            Ipv_DNN = PVPredict().fun_Ipv(Rs, Gp, IL, I0, b, voltage).numpy()[0,:]
            if curve=='pv': yDNN=voltage*Ipv_DNN
            elif curve=='iv': yDNN=Ipv_DNN
            try: P0 = ErrorData[labelDNN]
            except: P0 = {}
            P0[(S,T)]={'MAE' : self.MAE(yReal,  yDNN).numpy(),
                       'MSE' : self.MSE(yReal,  yDNN).numpy(),                                                           
                       'MAPE': self.MAPE(yReal, yDNN).numpy(),
                       'MSLE': self.MSLE(yReal, yDNN).numpy()} 
            ErrorData[labelDNN] = P0    
          except:pass
        except: pass
      
    if plotBar:
      width = 0.35/2
      fig = plt.figure(figsize=(15, 9))
      gs  = gridspec.GridSpec(nrows=2, ncols=4, figure=fig, wspace=0.4, hspace=0.55)
      for conty, Tref in enumerate(self.TList):
        for contx, metric in enumerate(['MAE', 'MSE', 'MSLE', 'MAPE']):
          ax = fig.add_subplot(gs[conty, contx])
          for num, model in enumerate(ErrorData.keys()):
            keys = [k for k in ErrorData[model].keys() if k[1]>0.9*Tref and k[1]<1.1*Tref]
            plotData = [ErrorData[model][key][metric] for key in keys]
            x = np.arange(len(keys))
            if not num: ax.bar(x, x*np.nan,  width)
            ax.bar(x + width*[-3/2, -1/2, 1/2, 3/2][num], plotData,  width, label=model)
            ax.set_title('\n'+metric)
            ax.set_xticks(x)
            ax.set_xlabel('Irradiancia (W/m$^2$)')
            ax.set_xticklabels([key[0] for key in keys], rotation=45)
      handles, labels = ax.get_legend_handles_labels()
      fig.legend(handles, labels, loc=1, bbox_to_anchor=LegendPos, ncol=len(labels))
      fig.suptitle('\n')
      if save: plt.savefig(curve+'Error.png', bbox_inches = 'tight')
      plt.show()
    if ShowTable:
      print("""
===================================================================================================================================================================================================================
          |       |                                                                                      Error                                                                                     
          |       |--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
     S    |   T   |                    MAE                    |                   MSE                     |                     MSLE                      |                            MAPE                    
          |       |--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
          |       | De Soto  |  Dobos   |   Boyd   |    NN    | De Soto  |  Dobos   |   Boyd   |    NN    | De Soto   |  Dobos    |   Boyd    |    NN     |   De Soto    |    Dobos     |     Boyd     |      NN        
===================================================================================================================================================================================================================""")
      for model in ErrorData: break 
      for key in ErrorData[model].keys():
        plotData=list(key)
        for metric in ['MAE', 'MSE', 'MSLE', 'MAPE']:
          plotData+= [ErrorData[model][key][metric] for model in ErrorData]
          if labelDNN not in list(ErrorData.keys()):
            plotData += [np.nan]
        print("{:>7.2f} |  {:>4.1f} | {:>8.3f} | {:>8.3f} | {:>8.3f} | {:>8.3f} | {:>8.3f} | {:>8.3f} | {:>8.3f} | {:>8.3f} | {:>8.3e} | {:>8.3e} | {:>8.3e} | {:>8.3e} | {:>12.3f} | {:>12.3f} | {:>12.3f} | {:>12.3f}".format(*plotData))
    return ErrorData
	
	
	
	


  def TrackingPlot(self, params, dayView, model=None, LegendPos=[0.9, 1], Xticks=8, 
                 showE=True, labelDNN='Neural Network', save=True):
    import datetime 
    [NGxView, NGyView, Time], yData = dayView, []
    for m in params: # Models
      Rs, Gp, IL, I0, b = [tf.reshape(k, [k.shape[0], 1]).numpy().astype('float64') for k in self.ModelParams(NGxView, params, m)]
      yData.append(PVPredict().predict(Rs, Gp, IL, I0, b))
    try: # Neural network
      Rs, Gp, IL, I0, b  = self.DNNParams(NGxView, model)
      yDNN = PVPredict().predict(Rs, Gp, IL, I0, b).numpy().astype('float64')
      [IscDNN, VscDNN, ImpDNN, VmpDNN, IocDNN, VocDNN] = np.hsplit(yDNN, 6)
    except:
      [IscDNN, VscDNN, ImpDNN, VmpDNN, IocDNN, VocDNN] = np.ones((6,1))*np.nan
    fig = plt.figure(figsize=(10, 10), constrained_layout=True)
    gs  = gridspec.GridSpec(3, ncols=3, figure=fig, hspace=0.1)
    labels, idx = ['Isc (A)', 'Vsc (V)','Imp (A)','Vmp (V)','Ioc (A)','Voc (V)', 'Pmp (W)'], 0
    T0 = datetime.datetime.timestamp(datetime.datetime.strptime(Time[0],'%H:%M')) 
    T1 = datetime.datetime.timestamp(datetime.datetime.strptime(Time[-1],'%H:%M'))
    ticks = [datetime.datetime.fromtimestamp(k).strftime("%H:%M") for k in np.linspace(start=T0, stop=T1, num=Xticks, dtype=int)]
    for ax, k1, label, DNN, in [[fig.add_subplot(gs[0, 0]), 0,'Irradiance (W/m$^2$)', np.nan],
                                [fig.add_subplot(gs[0, 1]), 1,'Temperature (°C)', np.nan],
                                [fig.add_subplot(gs[0, 2]), 1,     'Pmp (W)', VmpDNN*ImpDNN],
                                [fig.add_subplot(gs[1, 0]), 0,     'Isc (A)', IscDNN],
                                [fig.add_subplot(gs[1, 1]), False, 'Ioc (A)', IocDNN],
                                [fig.add_subplot(gs[1, 2]), 2,     'Imp (A)', ImpDNN], 
                                [fig.add_subplot(gs[2, 0]), False, 'Vsc (V)', VscDNN],
                                [fig.add_subplot(gs[2, 1]), 4,     'Voc (V)', VocDNN],
                                [fig.add_subplot(gs[2, 2]), 3,     'Vmp (V)', VmpDNN], ]:
      if not(label in labels):
        ax.plot(NGxView[:, k1], label='Experimental curve')
      else:
        idx = labels.index(label)
        try:
          ax.plot(NGyView[:, k1], label='Experimental curve')
        except:
          ax.plot(np.zeros([NGyView.shape[0], 1]), label='Experimental curve')
        for model in params:
          try:
            ax.plot(yData[model][:, idx], label=params[model]['name'].split(' Model')[0], ls='--')
          except:
            ax.plot(yData[model][:, 2]*yData[model][:, 3], label=params[model]['name'].split(' Model')[0], ls='--')
        if not(np.all(np.isnan(DNN))):
            ax.plot(DNN, ls='--', label=labelDNN)
      a1 = np.linspace(start=1, stop=NGxView.shape[0], num=NGxView.shape[0])
      ax.set_xticks(np.where(a1%Xticks==1)[0])
      ax.set_xticks(np.linspace(start=0, stop=NGxView.shape[0]-1, num=Xticks))
      if idx in [1, 3, 5]:
        ax.set_xticklabels(ticks, rotation=70)
        ax.set_xlabel('Time (hh:mm)')
      else:
        ax.axes.xaxis.set_ticklabels([])
      ax.set_xlim([0, NGxView.shape[0]-1])
      ax.grid(color='black', ls = '-.', lw = 0.1)
      ax.set_ylabel(label)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc=1, bbox_to_anchor=LegendPos, ncol=len(labels))
    fig.suptitle('\n')
    if save: plt.savefig('tracking.png', bbox_inches = 'tight')
    plt.show()
	
	
	
	
	
	
  def TrackingParams(self, params, dayView,  model=None, LegendPos=[0.59, 0.95], Xticks=8,
                  labelDNN='Neural Network', save=True):
    import datetime 
    [NGxView, NGyView, Time], yData = dayView, []
    for m in params: # Models
      yData.append([tf.reshape(k, [k.shape[0], 1]).numpy().astype('float64') for k in self.ModelParams(NGxView, params, m)]) 
    try: # Neural network
      RsDNN, GpDNN, ILDNN, I0DNN, bDNN = self.DNNParams(NGxView, model)
    except:
      RsDNN, GpDNN, ILDNN, I0DNN, bDNN = np.ones((5,1))*np.nan
    fig = plt.figure(figsize=(15, 8))
    gs  = gridspec.GridSpec(2, ncols=11, figure=fig, hspace=0.5, wspace=0.5)
    labels = ['Rs ($\Omega$)', 'Gp (S)', 'IL (A)', 'I0 (A)', 'b (1/V)']
    T0 = datetime.datetime.timestamp(datetime.datetime.strptime(Time[0],'%H:%M')) 
    T1 = datetime.datetime.timestamp(datetime.datetime.strptime(Time[-1],'%H:%M'))
    ticks = [datetime.datetime.fromtimestamp(k).strftime("%H:%M") for k in np.linspace(start=T0, stop=T1, num=Xticks, dtype=int)]
    for ax, label, DNN, in [[fig.add_subplot(gs[0, 0:3]), 'Rs ($\Omega$)', RsDNN],
                            [fig.add_subplot(gs[0, 4:7]), 'Gp (S)',        GpDNN],
                            [fig.add_subplot(gs[0, 8:11]), 'IL (A)',        ILDNN],
                            [fig.add_subplot(gs[1, 2:5]), 'I0 (A)',        I0DNN],
                            [fig.add_subplot(gs[1, 6:9]), 'b (1/V)',        bDNN] ]:
      idx = labels.index(label)
      ax.plot([np.nan]*2)
      for model in params:
        ax.plot(yData[model][idx], label=params[model]['name'], ls='--')
      if not(np.all(np.isnan(DNN))):
        ax.plot(DNN, ls='--', label=labelDNN)
      ax.set_xticks(np.linspace(start=0, stop=NGxView.shape[0]-1, num=Xticks))
      ax.set_xlim([0, NGxView.shape[0]-1])
      ax.set_xticklabels(ticks, rotation=70)
      ax.set_ylabel(label)
      ax.grid(color='black', ls = '-.', lw = 0.1)
      ax.set_xlabel('Time (hh:mm)')
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc=1, bbox_to_anchor=LegendPos, ncol=len(labels))
    if save: plt.savefig('params.png', bbox_inches = 'tight')
    plt.show()

  def plot3D(self, model, zlabel, clabel, PVModule, save=False, gridpts=50, s=10):
    from scipy.interpolate import griddata
    import matplotlib.colors
    cmap     = plt.cm.get_cmap("jet", 20)
    outlabel = zlabel+'_'+clabel+'.png'
    xNorm=np.array([
                                [1100.,   25.],[1100.,   50.],[1100.,   75.],
                  [1000.,   15.],[1000.,   25.],[1000.,   50.],[1000.,   75.],
                  [ 800.,   15.],[ 800.,   25.],[ 800.,   50.],[ 800.,   75.],
                  [ 600.,   15.],[ 600.,   25.],[ 600.,   50.],[ 600.,   75.],
                  [ 400.,   15.],[ 400.,   25.],[ 400.,   50.],
                  [ 200.,   15.],[ 200.,   25.],[ 200.,   50.],
                  [ 100.,   15.],[ 100.,   25.],  
                  ])

    x, y = xNorm[:,0], xNorm[:,1]
    x2, y2 = np.meshgrid(np.linspace(x.min(), x.max(), gridpts), np.linspace(y.min(), y.max(), gridpts))

    Rs, Gp, IL, I0, b = modelPV.DNNParams(xNorm, model, False)
    Isc, Vsc, Imp, Vmp, Ioc, Voc = tf.split(PVPredict().predict(Rs, Gp, IL, I0, b, MaxIterations=200, alpha=0, beta=0.8), axis=1, num_or_size_splits=6)

    Rs2, Gp2, IL2, I02, b2 = modelPV.DNNParams(np.array([x2, y2]).T.reshape(-1,2), model, False)
    Isc2, Vsc2, Imp2, Vmp2, Ioc2, Voc2 = tf.split(PVPredict().predict(Rs2, Gp2, IL2, I02, b2), axis=1, num_or_size_splits=6)

    if   'Rs'  in zlabel: 
      z, z2, view, zlabel = Rs, Rs2, 30, 0
    elif 'Gp'  in zlabel: 
      z, z2, view, zlabel = Gp, Gp2, -120, 1
    elif 'IL'  in zlabel: 
      z, z2, view, zlabel = IL, IL2, -120, 2
    elif 'I0'  in zlabel: 
      z, z2, view, zlabel = I0, I02, -30, 3
    elif 'b'   in zlabel: 
      z, z2, view, zlabel = b, b2, 135, 4
    elif 'Isc' in zlabel: 
      z, z2, view, zlabel = Isc, Isc2, -120, 5
    elif 'Voc' in zlabel: 
      z, z2, view, zlabel = Voc, Voc2, 120, 6
    elif 'Imp' in zlabel: 
      z, z2, view, zlabel = Imp, Imp2, -120, 7
    elif 'Vmp' in zlabel: 
      z, z2, view, zlabel = Vmp, Vmp2, 120, 8
    elif 'Pmp' in zlabel: 
      z, z2, view, zlabel = Imp*Vmp, Imp2*Vmp2, -120, 9

    if   'Rs'  in clabel: 
      c, c2, clabel = Rs, Rs2, 0
    elif 'Gp'  in clabel:
      c, c2, clabel = Gp, Gp2, 1
    elif 'IL'  in clabel:
      c, c2, clabel = IL, IL2, 2
    elif 'I0'  in clabel:
      c, c2, clabel = I0, I02, 3
    elif 'b'   in clabel:
      c, c2, clabel = b, b2, 4
    elif 'Isc' in clabel:
      c, c2, clabel = Isc, Isc2, 5
    elif 'Voc' in clabel:
      c, c2, clabel = Voc, Voc2, 6
    elif 'Imp' in clabel:
      c, c2, clabel = Imp, Imp2, 7
    elif 'Vmp' in clabel:
      c, c2, clabel = Vmp, Vmp2, 8
    elif 'Pmp' in clabel:
      c, c2, clabel = Imp*Vmp, Imp2*Vmp2, 9

    zlabel = ['Rs ($\Omega$)', 'Gp (S)', 'IL (A)', 'I0 (A)', 'b (1/V)', 'Isc (A)', 'Voc (V)', 'Imp (A)', 'Vmp (V)', 'Pmp (W)'][zlabel]
    clabel = ['Rs ($\Omega$)', 'Gp (S)', 'IL (A)', 'I0 (A)', 'b (1/V)', 'Isc (A)', 'Voc (V)', 'Imp (A)', 'Vmp (V)', 'Pmp (W)'][clabel]

    z, c = [k.numpy().flatten() for k in [z,c]]

    mask = np.isnan(griddata((x, y), z, (x2, y2), method='linear'))
    z2, c2 = [k.numpy().reshape(x2.shape).T for k in [z2,c2]]
    z2[np.where(mask==True)] = np.nan
    c2[np.where(mask==True)] = np.nan
    
    fig = plt.figure(figsize=(16, 12), dpi=80)
    gs = gridspec.GridSpec(7, 4, width_ratios=[10, 10, 10, 1], height_ratios=[1, 10, 1, 10, 1, 10, 1], hspace=0.3, wspace=0.1)
    ax1 = plt.subplot(gs[:, 1:3], projection='3d')
    ax2 = plt.subplot(gs[1, 0])
    ax3 = plt.subplot(gs[3, 0])
    ax4 = plt.subplot(gs[5, 0])
    axb = plt.subplot(gs[1:6, 3])
    
    ax1.view_init(30, view)

    norm = matplotlib.colors.Normalize(vmin=np.nanmin(c2), vmax=np.nanmax(c2))
    surf = ax1.plot_surface(x2, y2, z2, facecolors=cmap(norm(c2)), shade=False, antialiased=False, cstride=1, rstride=1, alpha=0.7, lw=0)
    ax2.pcolor(x2, y2, c2, cmap=cmap, norm=norm, antialiased=False, lw=1, vmin=np.nanmin(c2), vmax=np.nanmax(c2), alpha=0.7)
    ax3.pcolor(x2, z2, c2, cmap=cmap, norm=norm, antialiased=False, lw=1, vmin=np.nanmin(c2), vmax=np.nanmax(c2), alpha=0.7)
    ax4.pcolor(y2, z2, c2, cmap=cmap, norm=norm, antialiased=False, lw=1, vmin=np.nanmin(c2), vmax=np.nanmax(c2), alpha=0.7)


    x3, y3 = np.meshgrid(np.unique(x), np.unique(y))
    z3 = griddata((x, y), z, (x3, y3), method=inter)
    c3 = griddata((x, y), c, (x3, y3), method=inter)
    # ax1.plot_wireframe(x3, y3, z3, rstride=1, cstride=1, lw=0.3, color='black')
    
    m = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    m.set_array([])
    cbar = fig.colorbar(m, cax=axb, extend='both', alpha=0.7)
    cbar.mappable.set_clim([np.nanmin(c2), np.nanmax(c2)])
    axb.set_title(clabel)  

    
    ax1.scatter(x,y,z, s=s, color='black')
    ax1.set_xlabel('Irradiance (W/m$^2$)')
    ax1.set_ylabel('Temperature (°C)')
    ax1.set_zlabel(zlabel)

    ax2.scatter(x, y, s=s, color='black')
    ax2.set_xlabel('Irradiance (W/m$^2$)')
    ax2.set_ylabel('Temperature (°C)')

    ax3.scatter(x, z, s=s, color='black')
    ax3.set_xlabel('Irradiance (W/m$^2$)')
    ax3.set_ylabel(zlabel)

    ax4.scatter(y, z, s=s, color='black')
    ax4.set_xlabel('Temperature (°C)')
    ax4.set_ylabel(zlabel)
    ax1.set_title(PVModule)
    if save: plt.savefig(outlabel, bbox_inches = 'tight')
    plt.show()
	
