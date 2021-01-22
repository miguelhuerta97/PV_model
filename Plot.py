class FunctionPlotting:
  def __init__(self, df, SList=[100, 200, 400, 600, 800, 1000, 1100]): 
    try:
      plt.rcParams.update({
        'font.size'       : 16,
        'axes.titlesize'  : 14, 
        'axes.labelsize'  : 14,
        'xtick.labelsize' : 14,
        'ytick.labelsize' : 14,
        'figure.dpi'      : 80,
        'figure.facecolor': 'w',
        'figure.edgecolor': 'k',
        'figure.figsize'  : [15,4],
        })
    except:
      pass
    self.SList, self.TList, self.eps = SList, [25, 50], 1e-18
    self.TimeSeries = df.drop(df.columns[np.arange(0, 41)], axis=1)
    self.df  = df.drop(df.columns[np.append([2, 4, 6, 7, 8, 10, 12], np.arange(14, len(df.columns)))], axis=1)
    self.X, self.Y = [self.df[self.df.columns[k]].to_numpy(dtype='float32') for k in [[1,2],[3,4,5,6]]]          
    aux = self.df[self.df.columns[[0]][0]].str.split("T", n=1, expand=True)
    self.df.insert(0, "yyyy-mm-dd", aux[0])
    self.df[self.df.columns[1]] = aux[1].str.split(":", n=-1, expand = True).apply(lambda x: x[0]+':'+x[1], axis=1)
    self.df.rename(columns={self.df.columns[n+1]:k for n,k in enumerate(['hh:mm', 'S', 'T'])}, inplace=True)
    self.MAE  = tf.keras.losses.MeanAbsoluteError(reduction=tf.keras.losses.Reduction.NONE)
    self.MAPE = tf.keras.losses.MeanAbsolutePercentageError(reduction=tf.keras.losses.Reduction.NONE)
    self.MSE  = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
    self.MSLE = tf.keras.losses.MeanSquaredLogarithmicError(reduction=tf.keras.losses.Reduction.NONE)

  def SearchDay(self, dayView='2014-01-20'):
    df1 = self.df.loc[self.df[self.df[self.df.columns[0]]==dayView][self.df.columns[0]].index]
    return [df1[df1.columns[[2,3]]].to_numpy(dtype='float32'), 
            df1[df1.columns[[4,5,6,7]]].to_numpy(dtype='float32'), 
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

  def PlotLoss(self, loss, label='Mean square error'):
    if len(loss[0])!=0:
      fig = plt.figure(figsize=(13, 15))
      gs = gridspec.GridSpec(4, ncols=3, width_ratios=[5, 1, 5], wspace=0.03, hspace=0.5)
      ax = fig.add_subplot(gs[0, :])
      contx, conty = 0, 1
      for _n, _label  in enumerate(['CustomLoss', 'Isc', 'Imp', 'Vmp', 'Voc']):
        if _n!=0:
          ax = fig.add_subplot(gs[conty, contx])
          contx+=2
          if contx//3: 
            contx=0
            conty+=1
        ax.plot(loss[0][:,_n], ".b", label='Training')
        ax.plot(loss[1][:,_n], ".g", label='Validation')
        ax.axhline(y=loss[2][_n], color="k", linestyle="-.",label="Test")
        ax.set_ylabel(label, fontsize=14)
        ax.set_xlabel('epoch (-)', fontsize=14)
        ax.set_xlim([0, len(loss[0][:,_n])])
        ax.legend(fontsize=14), ax.set_title(_label)
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

  def CurvesPV_IV(self, curve, params={}, model=None, showE=True, Filter=[False, 100]):
    fig = plt.figure(figsize=(10, len(self.SList)*5))
    gs  = gridspec.GridSpec(nrows=len(self.SList), ncols=2, figure=fig, width_ratios=[5, 5], wspace=0.1, hspace=0.1)
    contx, conty, maxVolt, yLim = 0, 0, 0, np.ones((len(self.SList), 1))
    for T1 in self.TList:
      for S1 in self.SList:
        try:
          [current, voltage,  S, T] = self.SearchCurve(T1, S1, Filter=Filter)
          if np.max(voltage)>maxVolt:
            maxVolt = np.max(voltage)
          # Experimental curve
          if curve=='pv':
            yData = voltage*current 
          elif curve=='iv':
            yData=current
          if yLim[conty, 0] < np.max(yData):
            yLim[conty, 0] = np.ceil(np.max(yData))
          # Models      
          for m in params:
            Rs, Gp, IL, I0, b = self.ModelParams(np.array([[S, T]]), params, m)
            Ipv = PVPredict().fun_Ipv(Rs, Gp, IL, I0, b, voltage).numpy()
            if curve=='pv': 
              yData=voltage*Ipv
            elif curve=='iv':
              yData=Ipv
            if yLim[conty, 0] < np.max(yData):
              yLim[conty, 0] = np.ceil(np.max(yData))
          # Neural network
          try:
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
          # Experimental curve
          if curve=='pv':
            loc=2
            yData = voltage*current 
            if contx==0:
              ax.set_ylabel('$p_{pv}$ (W)')
          elif curve=='iv':
            loc=3
            yData=current
            if contx==0:
              ax.set_ylabel('$i_{pv}$ (A)')
          if showE:
            ax.plot(voltage, yData, label='Experimental curve')
          # Models      
          for m in params:
            Rs, Gp, IL, I0, b = self.ModelParams(np.array([[S, T]]), params, m)
            Ipv = PVPredict().fun_Ipv(Rs, Gp, IL, I0, b, voltage).numpy()
            if curve=='pv': 
              yData=voltage*Ipv
            elif curve=='iv':
              yData=Ipv
            ax.plot(voltage, yData, label=params[m]['name'], Linestyle='--')
          # Neural network
          try:
            Rs, Gp, IL, I0, b  = self.DNNParams(np.array([[S, T]])  , model)
            Ipv_DNN = PVPredict().fun_Ipv(Rs, Gp, IL, I0, b, voltage).numpy()[0,:]
            if curve=='pv':
              yDNN=voltage*Ipv_DNN
            elif curve=='iv':
              yDNN=Ipv_DNN
            ax.plot(voltage, yDNN, label='Neural network', Linestyle='--')
          except:
            pass
          lg = ax.legend(fontsize=12, loc=loc)
          if curve=='pv':
            ax.text(0.05, 0.75, '\n'.join((r'$S=%.2f$(W/m$^2$)' % (S),)), 
                    transform=ax.transAxes, fontsize=12,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
          else:
            ax.text(0.05, 0.3, '\n'.join((r'$S=%.2f$(W/m$^2$)' % (S),)), 
                    transform=ax.transAxes, fontsize=12,
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
    plt.show()

  def Tracking(self, params, model=None, dayView='2014-01-20', showE=True):
    NGxView, NGyView, Time = self.SearchDay(dayView=dayView)
    a1  = np.linspace(start=1, stop=NGxView.shape[0], num=NGxView.shape[0])
    b1  = Time[np.where(a1%6==1)]
    fig = plt.figure(figsize=(13, 20), constrained_layout=True)
    gs  = gridspec.GridSpec(4, ncols=3, figure=fig, width_ratios=[5, 1, 5], hspace=0.3)
    contx, conty, contData, yData = 0, 0, 0, []
    # Models
    for m in params:
      Rs, Gp, IL, I0, b = [tf.reshape(k, [k.shape[0], 1]).numpy().astype('float64') for k in self.ModelParams(NGxView, params, m)]
      yData.append(PVPredict().predict(Rs, Gp, IL, I0, b))
    # Neural network
    try:       
      Rs, Gp, IL, I0, b  = self.DNNParams(NGxView, model)
      yDNN = PVPredict().predict(Rs, Gp, IL, I0, b).numpy().astype('float64')
    except:
      pass
    for value, var in enumerate(['Irradiance (W/m$^2$)', 'Temperature (°C)', 'Isc (A)', 
                                 'Vsc (V)', 'Imp (A)', 'Vmp (V)', 'Ioc (A)', 'Voc (V)']):
      ax = fig.add_subplot(gs[conty, contx])
      if conty==0:
        ax.plot(NGxView[:, value], label='Experimental curve')
      else:
        if var in ['Isc (A)', 'Imp (A)', 'Vmp (V)', 'Voc (V)']:
          if showE:
            ax.plot(NGyView[:, contData], label='Experimental curve')
          contData+=1
        else:
          if showE:
            ax.plot(np.zeros([NGyView.shape[0], 1]), label='Experimental curve')
        for model in params:
          ax.plot(yData[model][:, value-2], label=params[model]['name'])
        try: 
          ax.plot(yDNN[:, value-2], label='Neural network')
        except:
          pass
      contx+=2
      if contx//4:
        contx=0
        conty+=1 
      ax.legend(loc=1, fontsize=12)
      ax.set_xlim([a1.min(), a1.max()])
      ax.set_ylabel(var, fontsize=14), ax.set_xticks(np.where(a1%6==1)[0])
      ax.set_xticklabels(b1, rotation=70), ax.grid(color='black', ls = '-.', lw = 0.1)
    fig.suptitle('Day: '+dayView+'\n', fontsize=18)
    plt.show()

  def TrackingParams(self, params, model=None, dayView='2014-01-20'):
    NGxView, NGyView, Time = self.SearchDay(dayView=dayView)
    a1  = np.linspace(start=1, stop=NGxView.shape[0], num=NGxView.shape[0])
    b1  = Time[np.where(a1%6==1)]
    fig = plt.figure(figsize=(13, 20), constrained_layout=True)
    gs  = gridspec.GridSpec(4, ncols=3, figure=fig, width_ratios=[5, 1, 5], hspace=0.3)
    contx, conty, contData, yData = 0, 0, 0, []
    # Models
    for m in params:
      yData.append([tf.reshape(k, [k.shape[0], 1]).numpy().astype('float64') for k in self.ModelParams(NGxView, params, m)]) 
    # Neural network
    try:       
      yDNN = self.DNNParams(NGxView, model)
    except:
      pass
    for value, var in enumerate(['Rs ($\\Omega$)', 'Gp (S)', 'IL (A)', 'I0 (A)', 'b (1/V)']):
      ax = fig.add_subplot(gs[conty, contx])
      for model in params:
        ax.plot(yData[model][value], label=params[model]['name'])
      try:
        ax.plot(yDNN[value], label='Neural network')
      except:
          pass
      contx+=2
      if contx//4:
        contx=0
        conty+=1 
      ax.legend(loc=1, fontsize=12)
      ax.set_xlim([a1.min(), a1.max()])
      ax.set_ylabel(var, fontsize=14), ax.set_xticks(np.where(a1%6==1)[0])
      ax.set_xticklabels(b1, rotation=70), ax.grid(color='black', ls = '-.', lw = 0.1)
    fig.suptitle('Day: '+dayView+'\n', fontsize=18)
    plt.show()

  def Error(self, xTest, yTest, params={}, model=None, outliers=1e10):
    fig  = plt.figure(figsize=(15, 10))
    gs  = gridspec.GridSpec(nrows=2, ncols=5, width_ratios=[2, 5, 2, 2, 5], hspace=0.3, wspace=0.03)
    ErrorData, contx, conty, positionsBox, labelBox, yData = {}, 0, 0, 0, [], []
    # Models
    for m in params:
      Rs, Gp, IL, I0, b = [tf.reshape(k, [k.shape[0], 1]).numpy().astype('float64') for k in self.ModelParams(xTest, params, m)]
      yData.append(np.delete(PVPredict().predict(Rs, Gp, IL, I0, b), [1, 4], 1))
      ErrorData[params[m]['name']] = {'MAE' : self.MAE(yTest.T,  yData[m].T).numpy(),
                                      #'MAPE': self.MAPE(yTest.T, yData[m].T).numpy(),
                                      'MAPE': self.MAPE(yData[m].T, yTest.T,).numpy(),
                                      'MSE' : self.MSE(yTest.T,  yData[m].T).numpy(),
                                      'MSLE': self.MSLE(yTest.T, yData[m].T).numpy()}
    # Neural network
    try: 
      Rs, Gp, IL, I0, b  = self.DNNParams(xTest, model)
      yDNN = np.delete(PVPredict().predict(Rs, Gp, IL, I0, b).numpy().astype('float64'), [1, 4], 1)
      ErrorData['Neural network'] = {'MAE' : self.MAE(yTest.T,  yDNN.T).numpy(),
                                     #'MAPE': self.MAPE(yTest.T, yDNN.T).numpy(),
                                     'MAPE': self.MAPE(yDNN.T, yTest.T).numpy(),
                                     'MSE' : self.MSE(yTest.T,  yDNN.T).numpy(),
                                     'MSLE': self.MSLE(yTest.T, yDNN.T).numpy()}
    except:
      pass
    for n, label in enumerate(['Isc (A)', 'Imp (A)', 'Vmp (V)', 'Voc (V)']):
      if contx == 0:
        ax1, ax2 = plt.subplot(gs[conty, 0]), plt.subplot(gs[conty, 1])
      elif contx == 1:
        ax1, ax2 = plt.subplot(gs[conty, 3]), plt.subplot(gs[conty, 4])
      for m in params:
        error = tf.math.reduce_mean(1-np.hsplit(yData[m]/yTest.astype('float64'), 4)[n], 1)*100
        error = error.numpy()[np.where(np.abs(error)<outliers)] ## Quita algunos outliers 
        ax1.boxplot(error, vert=True, positions=[positionsBox])
        positionsBox+=1
        labelBox.append(params[m]['label'])
        ax2.hist(error*100, 50, density=False, alpha=0.75, orientation="horizontal", label=params[m]['name'])
      try: 
        error = tf.math.reduce_mean(1-np.hsplit(yDNN/yTest.astype('float64'), 4)[n], 1)*100
        error = error.numpy()[np.where(np.abs(error)<outliers)] ## Quita algunos outliers 
        ax1.boxplot(error, vert=True, positions=[positionsBox])
        positionsBox+=1
        labelBox.append('NN')
        ax2.hist(error*100, 50, density=False, alpha=0.75, orientation="horizontal", label='Neural network')
      except:
        pass
      ax1.set_xticklabels(labelBox, fontsize=12), ax2.legend(fontsize=12, loc=1)
      ax1.set_ylabel('Prediction error (\\%)', fontsize=14), ax1.xaxis.tick_top()
      ax2.xaxis.set_ticklabels(np.around(plt.xticks()[0]/error.shape[0]*100, 1))
      ax2.yaxis.set_ticks([]), ax2.set_xlabel('Frequency (\\%)', fontsize=14), 
      ax2.set_title(label), 
      contx+=1
      if contx//2:
        contx=0
        conty+=1    
    plt.show()
    return ErrorData

  def ErrorPV_IV(self, curve, params={}, model=None, outliers=1e10):  
    fig = plt.figure(figsize=(15, len(self.SList)*7))
    gs  = gridspec.GridSpec(nrows=len(self.SList), ncols=5, figure=fig, width_ratios=[2, 5, 2, 2, 5], wspace=0.03, hspace=0.25)
    ErrorData, contx, conty, contData, positionsBox, labelBox = {}, 0, 0, 0, 0, []
    for T1 in self.TList:
      for S1 in self.SList:
        current, voltage,  S, T = self.SearchCurve(T1, S1)
        if contx == 0:
          ax1, ax2 = plt.subplot(gs[conty, 0]), plt.subplot(gs[conty, 1])
        elif contx == 2:
          ax1, ax2 = plt.subplot(gs[conty, 3]), plt.subplot(gs[conty, 4])
        # Experimental curve
        if curve=='pv':
          yReal = voltage*current
        elif curve=='iv':
          yReal=current   
        # Models   
        for m in params:
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
          ax2.hist(error*100, 50, density=False, alpha=0.75, orientation="horizontal", label=params[m]['name'])
          ErrorData[params[m]['name']+' - '+str(contData)] = {'MAE' : self.MAE(yReal.T,  yData.T).numpy(),
                                                              #'MAPE': self.MAPE(yReal.T, yData.T).numpy(),
                                                              'MAPE': self.MAPE(yData.T, yReal.T).numpy(),
                                                              'MSE' : self.MSE(yReal.T,  yData.T).numpy(),
                                                              'MSLE': self.MSLE(yReal.T, yData.T).numpy(), 
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
          ax2.hist(error*100, 50, density=False, alpha=0.75, orientation="horizontal", label='Neural network')
          ErrorData['Neural network - '+str(contData)] ={'MAE' : self.MAE(yReal.T,  yDNN.T).numpy(),
                                                         #'MAPE': self.MAPE(yReal.T, yDNN.T).numpy(),
                                                         'MAPE': self.MAPE(yDNN.T, yReal.T).numpy(),
                                                         'MSE' : self.MSE(yReal.T,  yDNN.T).numpy(),
                                                         'MSLE': self.MSLE(yReal.T, yDNN.T).numpy(), 
                                                         'S':S, 'T':T} 
        except:
          pass
        ax1.xaxis.tick_top()
        ax2.legend(fontsize=12, loc=1), ax1.set_xticklabels(labelBox, fontsize=12)
        ax1.set_ylabel('Prediction error (\\%)', fontsize=14)
        ax2.xaxis.set_ticklabels(np.around(plt.xticks()[0]/error.shape[0]*100, 1))
        ax2.yaxis.set_ticks([]), ax2.set_xlabel('Frequency (\\%)', fontsize=14)
        ax2.set_title('S: '+str(S)+'(W/m$^2$) - T: '+str(T) +'(°C)')        
        conty+=1
        if conty//len(self.SList):
          conty=0
          contx+=2
        contData+=1
    plt.show()
    return ErrorData
    
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
