class FunctionPlotting:
  def __init__(self, df, SList=[100, 200, 400, 600, 800, 1000]): 
    plt.rcParams.update({
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
  def SearchDay(self, dayView='2014-01-20'):
    df1 = self.df.loc[self.df[self.df[self.df.columns[0]]==dayView][self.df.columns[0]].index]
    return [df1[df1.columns[[2,3]]].to_numpy(dtype='float32'), 
            df1[df1.columns[[4,5,6,7,8]]].to_numpy(dtype='float32'), 
            df1[df1.columns[1]].to_numpy()] 
  def SearchCurve(self, T1, S1, Filter=[False, 100]):
    idx = np.where((self.df[self.df.columns[[3]]]==[T1]).to_numpy()==True)[0]
    aux = np.square(self.df[self.df.columns[[2]]].iloc[idx].to_numpy()-S1)
    idx = idx[np.where(aux==aux.min())[0]][0]
    current = self.TimeSeries.iloc[idx][np.arange(1, self.TimeSeries.iloc[idx][0]+1, dtype=int)].to_numpy()
    voltage = self.TimeSeries.iloc[idx][np.arange(self.TimeSeries.iloc[idx][0]+1, 2*self.TimeSeries.iloc[idx][0]+1, dtype=int)].to_numpy()
    if Filter[0]:
      if (S1-self.df.iloc[idx]['S'])**2 < Filter[1]:
        return [current, voltage, self.df.iloc[idx]['S'], self.df.iloc[idx]['T']]
    else:
      return [current, voltage, self.df.iloc[idx]['S'], self.df.iloc[idx]['T']]
  def CurvesPV_IV(self, curve, params={}, model=None, showE=True, Filter=[False, 100], 
                  yPos=[0.7, 0.35], LegendPos=[0.865, 0.777]):
    fig = plt.figure(figsize=(10, len(self.SList)*5))
    gs  = gridspec.GridSpec(nrows=len(self.SList), ncols=2, figure=fig, width_ratios=[1, 1], wspace=0.1, hspace=0.1)
    contx, conty, maxVolt, yLim = 0, 0, 0, np.ones((len(self.SList), 1))
    for T1 in self.TList:
      for S1 in self.SList:
        try:
          [current, voltage,  S, T] = self.SearchCurve(T1, S1, Filter=Filter)
          if np.max(voltage)>maxVolt:
            maxVolt = np.max(voltage)
          if curve=='pv':# Experimental curve
            yData = voltage*current 
          elif curve=='iv':
            yData=current
          if yLim[conty, 0] < np.max(yData):
            yLim[conty, 0] = np.ceil(np.max(yData))
          for m in params:# Models      
            Rs, Gp, IL, I0, b = self.ModelParams(np.array([[S, T]]), params, m)
            Ipv = PVPredict().fun_Ipv(Rs, Gp, IL, I0, b, voltage).numpy()
            if curve=='pv': 
              yData=voltage*Ipv
            elif curve=='iv':
              yData=Ipv
            if yLim[conty, 0] < np.max(yData):
              yLim[conty, 0] = np.ceil(np.max(yData))
          try:# Neural network
            Rs, Gp, IL, I0, b  = self.DNNParams(np.array([[S, T]])  , model)
            Ipv_DNN = PVPredict().fun_Ipv(Rs, Gp, IL, I0, b, voltage).numpy()[0,:]
            if curve=='pv':
              yDNN=voltage*Ipv_DNN
            elif curve=='iv':
              yDNN=Ipv_DNN
            if yLim[conty, 0] < np.max(yDNN):
              yLim[conty, 0] = np.ceil(np.max(yDNN))
          except:
            pass
        except:
          pass
        conty+=1
        if conty//len(self.SList):
          conty=0
    conty=0
    for T1 in self.TList:
      for S1 in self.SList:
        ax =  plt.subplot(gs[conty, contx])
        try:
          [current, voltage,  S, T] = self.SearchCurve(T1, S1, Filter=Filter)
          if curve=='pv':# Experimental curve
            yData = voltage*current 
            if contx==0:
              ax.set_ylabel('$p_{pv}$ (W)')
          elif curve=='iv':
            yData=current
            if contx==0:
              ax.set_ylabel('$i_{pv}$ (A)')
          if showE:
            ax.plot(voltage, yData, label='Experimental curve')
          for m in params:# Models      
            Rs, Gp, IL, I0, b = self.ModelParams(np.array([[S, T]]), params, m)
            Ipv = PVPredict().fun_Ipv(Rs, Gp, IL, I0, b, voltage).numpy()
            if curve=='pv': 
              yData=voltage*Ipv
            elif curve=='iv':
              yData=Ipv
            ax.plot(voltage, yData, label=params[m]['name'], ls='--')
          try:# Neural network
            Rs, Gp, IL, I0, b  = self.DNNParams(np.array([[S, T]])  , model)
            Ipv_DNN = PVPredict().fun_Ipv(Rs, Gp, IL, I0, b, voltage).numpy()[0,:]
            if curve=='pv':
              yDNN=voltage*Ipv_DNN
            elif curve=='iv':
              yDNN=Ipv_DNN
            ax.plot(voltage, yDNN, label='DNN+Boyd', ls='--')
          except:
            pass
          if curve=='pv':
            ax.text(0.05, yPos[0], '\n'.join((r'$S=%.2f$(W/m$^2$)' % (S),)), transform=ax.transAxes, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
          else:
            ax.text(0.05, yPos[1], '\n'.join((r'$S=%.2f$(W/m$^2$)' % (S),)), transform=ax.transAxes, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        except:
          ax.axis('off')
          pass
        ax.grid(alpha=0.75), 
        ax.set_ylim([0, yLim[conty, 0]])
        ax.set_xlim([0, maxVolt])
        if contx:
          ax.axes.yaxis.set_ticklabels([])
        if conty//(len(self.SList)-1):
          ax.set_xlabel('$v_{pv}$(V)')
        else:
          ax.axes.xaxis.set_ticklabels([])
        conty+=1
        if conty//len(self.SList):
          conty=0
          contx+=1
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc=1, bbox_to_anchor=LegendPos, ncol=len(labels))
    fig.suptitle('\n')
    plt.subplots_adjust(top=0.97)
    plt.show()
  def ErrorPV_IV(self, curve, params={}, model=None, outliers=1e10, Filter=[False, 100], ShowTableError=False, LegendPos=[0.865, 0.777]):  
    fig = plt.figure(figsize=(10, len(self.SList)*5))
    gs  = gridspec.GridSpec(nrows=len(self.SList), ncols=5, figure=fig, width_ratios=[2, 5, 2, 2, 5], wspace=0.03, hspace=0.3)
    ErrorData, contx, conty, contData, positionsBox, labelBox = {}, 0, 0, 0, 0, []
    for T1 in self.TList:
      for S1 in self.SList:
        if contx == 0:
          ax1, ax2 = plt.subplot(gs[conty, 0]), plt.subplot(gs[conty, 1])
        elif contx == 2:
          ax1, ax2 = plt.subplot(gs[conty, 3]), plt.subplot(gs[conty, 4])
        try:
          current, voltage, S, T = self.SearchCurve(T1, S1, Filter=Filter)
          if curve=='pv':# Experimental curve
            yReal = voltage*current
          elif curve=='iv':
            yReal=current   
          for m in params:# Models   
            Rs, Gp, IL, I0, b = self.ModelParams(np.array([[S, T]]), params, m)
            Ipv = PVPredict().fun_Ipv(Rs, Gp, IL, I0, b, voltage).numpy()
            if curve=='pv': 
              yData=voltage*Ipv
            elif curve=='iv':
              yData=Ipv                  
            error = tf.math.reduce_mean(1-yData.reshape(yData.shape[0], 1)/(yReal.reshape(yReal.shape[0], 1)+self.eps), 1)*100
            error = error.numpy()[np.where(np.abs(error)<outliers)] ## Quita algunos outliers 
            ax1.boxplot(error, vert=True, positions=[positionsBox])
            positionsBox+=1
            labelBox.append(params[m]['label'])
            ax2.hist(error*100, 50, density=False, alpha=0.75, orientation="horizontal", label=params[m]['name'].split(' Model')[0])
            ErrorData[params[m]['name']+' - '+str(contData)] = {'MAE' : self.MAE(yReal,  yData).numpy(),
                                                                'MSE' : self.MSE(yReal,  yData).numpy(),
                                                                'MAPE': self.MAPE(yReal, yData).numpy(),
                                                                'MSLE': self.MSLE(yReal, yData).numpy(), 
                                                                'S':S, 'T':T} 
          # Neural network
          try:
            Rs, Gp, IL, I0, b  = self.DNNParams(np.array([[S, T]]), model)
            Ipv_DNN = PVPredict().fun_Ipv(Rs, Gp, IL, I0, b, voltage).numpy()[0,:]
            if curve=='pv':
              yDNN=voltage*Ipv_DNN
            elif curve=='iv':
              yDNN=Ipv_DNN      
            error = tf.math.reduce_mean(1-yDNN.reshape(yDNN.shape[0], 1)/(yReal.reshape(yReal.shape[0], 1)+self.eps), 1)*100
            error = error.numpy()[np.where(np.abs(error)<outliers)] ## Quita algunos outliers     
            ax1.boxplot(error, vert=True, positions=[positionsBox])
            positionsBox+=1
            labelBox.append('NN')
            ax2.hist(error*100, 50, density=False, alpha=0.75, orientation="horizontal", label='DNN+Boyd')
            ErrorData['Neural network - '+str(contData)] ={'MAE' : self.MAE(yReal,  yDNN).numpy(),
                                                           'MSE' : self.MSE(yReal,  yDNN).numpy(),                                                           
                                                           'MAPE': self.MAPE(yReal, yDNN).numpy(),
                                                           'MSLE': self.MSLE(yReal, yDNN).numpy(), 
                                                           'S':S, 'T':T} 
          except:
            pass
          ax1.xaxis.tick_top()          
          ax1.set_xticklabels(labelBox)
          ax2.xaxis.set_ticklabels(np.around(plt.xticks()[0]/error.shape[0]*100, 1))
          ax2.yaxis.set_ticks([])
          ax1.set_ylabel('Prediction error (%)')
          ax2.set_xlabel('Frequency (%)')    
        except:
          ax1.axis('off')
          ax2.axis('off')
        conty+=1
        if conty//len(self.SList):
          conty=0
          contx+=2
        contData+=1
    handles, labels = ax2.get_legend_handles_labels()
    fig.legend(handles, labels, loc=1, bbox_to_anchor=LegendPos, ncol=len(labels))
    fig.suptitle('\n')
    plt.subplots_adjust(top=0.94)
    plt.show()
    if ShowTableError:
      num50=True
      error25 = np.ones((len(self.SList), 20+2 ))*np.nan
      error50 = np.ones((len(self.SList), 20+2 ))*np.nan
      for num, k in enumerate(ErrorData):
        label = k.split('-')[0].split(' Model')[0].replace(' ','')
        if ErrorData[k]['T']==25:
          if not(num):
            Sref25=ErrorData[k]['S']
            conty25=0
          if Sref25!=ErrorData[k]['S']:
            Sref25=ErrorData[k]['S']
            conty25+=1
          error25[conty25, [0,1]] = [ErrorData[k][er] for er in ['S', 'T']]
          if label=='Neuralnetwork':
            error25[conty25, [2,7,12,17]] = [ErrorData[k][er] for er in ['MAE', 'MSE', 'MSLE', 'MAPE']]
          elif label=='Proposed':
            error25[conty25, [3,8,13,18]] = [ErrorData[k][er] for er in ['MAE', 'MSE', 'MSLE', 'MAPE']]
          elif label=='DeSoto':
            error25[conty25, [4,9,14,19]] = [ErrorData[k][er] for er in ['MAE', 'MSE', 'MSLE', 'MAPE']]
          elif label=='Dobos':
            error25[conty25, [5,10,15,20]] = [ErrorData[k][er] for er in ['MAE', 'MSE', 'MSLE', 'MAPE']]
          elif label=='Boyd':
            error25[conty25, [6,11,16,21]] = [ErrorData[k][er] for er in ['MAE', 'MSE', 'MSLE', 'MAPE']]
        elif ErrorData[k]['T']==50:
          if num50:
            num50 = False
            Sref50=ErrorData[k]['S']
            conty50=0
          if Sref50!=ErrorData[k]['S']:
            Sref50=ErrorData[k]['S']
            conty50+=1
          error50[conty50, [0,1]] = [ErrorData[k][er] for er in ['S', 'T']]
          if label=='Neuralnetwork':
            error50[conty50, [2,7,12,17]] = [ErrorData[k][er] for er in ['MAE', 'MSE', 'MSLE', 'MAPE']]
          elif label=='Proposed':
            error50[conty50, [3,8,13,18]] = [ErrorData[k][er] for er in ['MAE', 'MSE', 'MSLE', 'MAPE']]
          elif label=='DeSoto':
            error50[conty50, [4,9,14,19]] = [ErrorData[k][er] for er in ['MAE', 'MSE', 'MSLE', 'MAPE']]
          elif label=='Dobos':
            error50[conty50, [5,10,15,20]] = [ErrorData[k][er] for er in ['MAE', 'MSE', 'MSLE', 'MAPE']]
          elif label=='Boyd':
            error50[conty50, [6,11,16,21]] = [ErrorData[k][er] for er in ['MAE', 'MSE', 'MSLE', 'MAPE']]
      print("""
============================================================================================================================================================================================================================================
        |       |                                                                                      Error                                                                                     
        |       |---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    S   |   T   |                          MAE                         |                          MSE                         |                          MSLE                        |                          MAPE                         
        |       |---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        |       | DNN+Boyd | Proposed | De Soto  |  Dobos   |   Boyd   | DNN+Boyd | Proposed | De Soto  |  Dobos   |   Boyd   | DNN+Boyd | Proposed | De Soto  |  Dobos   |   Boyd   | DNN+Boyd | Proposed | De Soto  |  Dobos   |   Boyd    
============================================================================================================================================================================================================================================""")
      for k in range(len(self.SList)):
        if not(np.all(np.isnan(error25[k,:]))):
          print("{:>7.2f} |  {:>4.1f} | {:>8.3f} | {:>8.3f} | {:>8.3f} | {:>8.3f} | {:>8.3f} | {:>8.3f} | {:>8.3f} | {:>8.3f} | {:>8.3f} | {:>8.3f} | {:>8.3f} | {:>8.3f} | {:>8.3f} | {:>8.3f} | {:>8.3f} | {:>8.3f} | {:>8.3f} | {:>8.3f} | {:>8.3f} | {:>8.3f}".format(
        *[k[0] for k in np.hsplit(error25[k,:], 20+2)] ))
      for k in range(len(self.SList)):
        if not(np.all(np.isnan(error50[k,:]))):
          print("{:>7.2f} |  {:>4.1f} | {:>8.3f} | {:>8.3f} | {:>8.3f} | {:>8.3f} | {:>8.3f} | {:>8.3f} | {:>8.3f} | {:>8.3f} | {:>8.3f} | {:>8.3f} | {:>8.3f} | {:>8.3f} | {:>8.3f} | {:>8.3f} | {:>8.3f} | {:>8.3f} | {:>8.3f} | {:>8.3f} | {:>8.3f} | {:>8.3f}".format(
        *[k[0] for k in np.hsplit(error50[k,:], 20+2)] ))
    return ErrorData
  def Tracking(self, params, model=None, dayView='2014-01-20', showE=True, LegendPos=[0.865, 0.777]):
    [NGxView, NGyView, Time], yData = self.SearchDay(dayView=dayView), []
    for m in params: # Models
      Rs, Gp, IL, I0, b = [tf.reshape(k, [k.shape[0], 1]).numpy().astype('float64') for k in self.ModelParams(NGxView, params, m)]
      yData.append(PVPredict().predict(Rs, Gp, IL, I0, b))
    try: # Neural network
      Rs, Gp, IL, I0, b  = self.DNNParams(NGxView, model)
      yDNN = PVPredict().predict(Rs, Gp, IL, I0, b).numpy().astype('float64')
      [IscDNN, VscDNN, ImpDNN, VmpDNN, IocDNN, VocDNN] = np.hsplit(yDNN, 6)
    except:
      [IscDNN, VscDNN, ImpDNN, VmpDNN, IocDNN, VocDNN] = np.ones((6,1))*np.nan
      pass
    fig = plt.figure(figsize=(10, 10), constrained_layout=True)
    gs  = gridspec.GridSpec(3, ncols=3, figure=fig, hspace=0.1)
    labels, idx = ['Isc (A)', 'Vsc (V)','Imp (A)','Vmp (V)','Ioc (A)','Voc (V)', 'Pmp (W)'], 0
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
            ax.plot(yData[model][:, idx], label=params[model]['name'].split(' Model')[0])
          except:
            ax.plot(yData[model][:, 2]*yData[model][:, 3], label=params[model]['name'].split(' Model')[0])
        if not(np.all(np.isnan(DNN))):
            ax.plot(DNN, label='DNN+Boyd')
      a1 = np.linspace(start=1, stop=NGxView.shape[0], num=NGxView.shape[0])
      if idx in [1, 3, 5]:
        ax.set_xticks(np.where(a1%12==1)[0])
        ax.set_xticklabels(Time[np.where(a1%12==1)], rotation=70), 
      else:
        ax.set_xticks([])
      ax.set_xlim([a1.min(), a1.max()])
      ax.grid(color='black', ls = '-.', lw = 0.1)
      ax.set_ylabel(label)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc=1, bbox_to_anchor=LegendPos, ncol=len(labels))
    fig.suptitle('\n')
    plt.show()
  def TrackingParams(self, params, model=None, dayView='2014-01-20', LegendPos=[1,1]):
    [NGxView, NGyView, Time], yData = self.SearchDay(dayView=dayView), []
    for m in params: # Models
      yData.append([tf.reshape(k, [k.shape[0], 1]).numpy().astype('float64') for k in self.ModelParams(NGxView, params, m)]) 
    try: # Neural network
      RsDNN, GpDNN, ILDNN, I0DNN, bDNN = self.DNNParams(NGxView, model)
    except:
      RsDNN, GpDNN, ILDNN, I0DNN, bDNN = np.ones((5,1))*np.nan
      pass    
    fig = plt.figure(figsize=(10, 12))
    gs  = gridspec.GridSpec(3, ncols=2, figure=fig, hspace=0.1, wspace=0.33)
    labels = ['Rs ($\Omega$)', 'Gp (S)', 'IL (A)', 'I0 (A)', 'b (1/V)']
    for ax, label, DNN, in [[fig.add_subplot(gs[2, 1]), 'Rs ($\Omega$)', RsDNN],
                            [fig.add_subplot(gs[2, 0]), 'Gp (S)',        GpDNN],
                            [fig.add_subplot(gs[1, 0]), 'IL (A)',        ILDNN],
                            [fig.add_subplot(gs[1, 1]), 'I0 (A)',        I0DNN],
                            [fig.add_subplot(gs[0, 0]), 'b (1/V)',        bDNN] ]:
      idx = labels.index(label)
      for model in params:
        ax.plot(yData[model][idx], label=params[model]['name'])
      if not(np.all(np.isnan(DNN))):
         ax.plot(DNN, label='DNN+Boyd')
      a1  = np.linspace(start=1, stop=NGxView.shape[0], num=NGxView.shape[0])
      if idx in [0, 1]:
        ax.set_xticks(np.where(a1%12==1)[0])
        ax.set_xticklabels(Time[np.where(a1%12==1)], rotation=70), 
      else:
        ax.set_xticks([])
      ax.set_xlim([a1.min(), a1.max()])
      ax.set_ylabel(label)
      ax.grid(color='black', ls = '-.', lw = 0.1)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc=1, bbox_to_anchor=LegendPos)
    plt.show()
  def Error(self, xTest, yTest, params={}, model=None, outliers=1e10, LegendPos=[1,1], ShowTableError=False):
    ErrorData, yData = {}, []
    for m in params: # Models
      Rs, Gp, IL, I0, b = [tf.reshape(k, [k.shape[0], 1]).numpy().astype('float64') for k in self.ModelParams(xTest, params, m)]
      [Isc, Vsc, Imp, Vmp, Ioc, Voc] = np.hsplit(PVPredict().predict(Rs, Gp, IL, I0, b).numpy().astype('float64'), 6)
      yData.append(np.hstack([Isc, Imp*Vmp, Imp, Vmp, Voc]))
      ErrorData[params[m]['name']] = {'MAE' : self.MAE(yTest, yData[m]).numpy(),
                                      'MSE' : self.MSE(yTest, yData[m]).numpy(),
                                      'MAPE': self.MAPE(yTest, yData[m]).numpy(),
                                      'MSLE': self.MSLE(yTest, yData[m]).numpy()}
      for var in ['Isc', 'Pmp','Imp', 'Vmp', 'Voc']:
        idx1 = ['Isc', 'Pmp','Imp', 'Vmp', 'Voc'].index(var)
        ErrorData[params[m]['name']+'_'+var] = {'MAE' : self.MAE(yTest[:, idx1],  yData[m][:, idx1]).numpy(),
                                                'MSE' : self.MSE(yTest[:, idx1],  yData[m][:, idx1]).numpy(),
                                                'MAPE': self.MAPE(yTest[:, idx1], yData[m][:, idx1]).numpy(),
                                                'MSLE': self.MSLE(yTest[:, idx1], yData[m][:, idx1]).numpy()}
    try: # Neural network
      Rs, Gp, IL, I0, b  = self.DNNParams(xTest, model)      
      [IscDNN, VscDNN, ImpDNN, VmpDNN, IocDNN, VocDNN] = np.hsplit(PVPredict().predict(Rs, Gp, IL, I0, b).numpy().astype('float64'), 6)
      yDNN = np.hstack([IscDNN, ImpDNN*VmpDNN, ImpDNN, VmpDNN, VocDNN])
      ErrorData['Neural network'] = {'MAE' : self.MAE(yTest, yDNN).numpy(),
                                     'MSE' : self.MSE(yTest, yDNN).numpy(),
                                     'MAPE': self.MAPE(yTest, yDNN).numpy(),
                                     'MSLE': self.MSLE(yTest, yDNN).numpy()}
      for var in ['Isc', 'Pmp','Imp', 'Vmp', 'Voc']:
        idx1 = ['Isc', 'Pmp','Imp', 'Vmp', 'Voc'].index(var)
        ErrorData['Neural network_'+var] = {'MAE' : self.MAE(yTest[:, idx1],  yDNN[:, idx1]).numpy(),
                                            'MSE' : self.MSE(yTest[:, idx1],  yDNN[:, idx1]).numpy(),
                                            'MAPE': self.MAPE(yTest[:, idx1], yDNN[:, idx1]).numpy(),
                                            'MSLE': self.MSLE(yTest[:, idx1], yDNN[:, idx1]).numpy()}
    except:
      [IscDNN, VscDNN, ImpDNN, VmpDNN, IocDNN, VocDNN] = np.ones((6,1))*np.nan
      pass
    fig = plt.figure(figsize=(10, 15))
    gs  = gridspec.GridSpec(nrows=3, ncols=5, figure=fig, width_ratios=[2, 5, 2, 2, 5], wspace=0.03, hspace=0.3)
    contx, conty, positionsBox, labelBox = 0, 0, 0, []
    labels, idx = ['Isc (A)', 'Pmp (W)', 'Imp (A)','Vmp (V)','Voc (V)'], 0
    for [ax1, ax2], label, DNN, in [[[fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])], 'Pmp (W)', ImpDNN*VmpDNN],
                                    [[fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])], 'Imp (A)', ImpDNN],
                                    [[fig.add_subplot(gs[1, 3]), fig.add_subplot(gs[1, 4])], 'Vmp (V)', VmpDNN],
                                    [[fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1])], 'Isc (A)', IscDNN],
                                    [[fig.add_subplot(gs[2, 3]), fig.add_subplot(gs[2, 4])], 'Voc (V)', VocDNN] ]:
      idx = labels.index(label)
      for model in params:
        error = ((yTest[:,idx]-yData[model][:, idx])/yTest[:,idx])*100
        error = error[np.where(np.abs(error)<outliers)]
        ax1.boxplot(error, vert=True, positions=[positionsBox])
        positionsBox+=1
        labelBox.append(params[model]['label'])
        ax2.hist(error*100, 50, density=False, alpha=0.75, orientation="horizontal", label=params[model]['name'])
      if not(np.all(np.isnan(DNN))):
        error = ((yTest[:,idx]-DNN[:,0])/yTest[:,idx])*100
        error = error[np.where(np.abs(error)<outliers)]
        ax1.boxplot(error, vert=True, positions=[positionsBox])
        positionsBox+=1
        labelBox.append('NN')
        ax2.hist(error*100, 50, density=False, alpha=0.75, orientation="horizontal", label='DNN+Boyd')
      ax1.set_xticklabels(labelBox), 
      ax1.set_ylabel('Prediction error (%)'), 
      ax1.xaxis.tick_top()
      ax2.xaxis.set_ticklabels(np.around(ax2.get_xticks()/error.shape[0]*100, 1))
      ax2.yaxis.set_ticks([]), 
      ax2.set_xlabel('Frequency (%)'), 
      ax2.set_title(label)
    handles, labels = ax2.get_legend_handles_labels()
    fig.legend(handles, labels, loc=1, bbox_to_anchor=LegendPos)
    plt.show()
    if ShowTableError:
      error = np.ones((6, 20))*np.nan
      for num, k in enumerate(ErrorData):
        if 'Isc' in k.split('_'):
          conty=0
        elif 'Pmp' in k.split('_'):
          conty=1
        elif 'Imp' in k.split('_'):
          conty=2
        elif 'Vmp' in k.split('_'):
          conty=3
        elif 'Voc' in k.split('_'):
          conty=4
        else:
          conty=5
        if 'Neural network' in k.split('_'):
          error[conty, [0,5,10,15]] = [ErrorData[k][er] for er in ['MAE', 'MSE', 'MSLE', 'MAPE']]
        elif 'Proposed' in k.split('_'):
          error[conty, [1,6,11,16]] = [ErrorData[k][er] for er in ['MAE', 'MSE', 'MSLE', 'MAPE']]
        elif 'De Soto Model' in k.split('_'):
          error[conty, [2,7,12,17]] = [ErrorData[k][er] for er in ['MAE', 'MSE', 'MSLE', 'MAPE']]
        elif 'Dobos Model' in k.split('_'):
          error[conty, [3,8,13,18]] = [ErrorData[k][er] for er in ['MAE', 'MSE', 'MSLE', 'MAPE']]
        elif 'Boyd Model' in k.split('_'):
          error[conty, [4,9,14,19]] = [ErrorData[k][er] for er in ['MAE', 'MSE', 'MSLE', 'MAPE']]
      labels = ['Isc', 'Pmp', 'Imp','Vmp','Voc', 'Total']
      print("""
====================================================================================================================================================================================================================================
        |                                                                                  Error                                                                                     
        |---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
   Var  |                          MAE                         |                          MSE                         |                          MSLE                        |                          MAPE                         
        |---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        | DNN+Boyd | Proposed | De Soto  |  Dobos   |   Boyd   | DNN+Boyd | Proposed | De Soto  |  Dobos   |   Boyd   | DNN+Boyd | Proposed | De Soto  |  Dobos   |   Boyd   | DNN+Boyd | Proposed | De Soto  |  Dobos   |   Boyd    
====================================================================================================================================================================================================================================""")
      for k in range(6):
        print("{:6s}  | {:>8.3f} | {:>8.3f} | {:>8.3f} | {:>8.3f} | {:>8.3f} | {:>8.3f} | {:>8.3f} | {:>8.3f} | {:>8.3f} | {:>8.3f} | {:>8.3f} | {:>8.3f} | {:>8.3f} | {:>8.3f} | {:>8.3f} | {:>8.3f} | {:>8.3f} | {:>8.3f} | {:>8.3f} | {:>8.3f}".format(
        labels[k], *error[k, :]))
    return ErrorData
