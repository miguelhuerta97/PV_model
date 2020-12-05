class FunctionPlotting:
  def __init__(self, df, SList=[100, 200, 400, 600, 800, 1000, 1100]): 
    try:
      plt.rcParams.update({
        'font.size'       : 16,
        'axes.titlesize'  : 14, 
        'axes.labelsize'  : 20,
        'xtick.labelsize' : 14,
        'ytick.labelsize' : 14,
        'figure.dpi'      : 80,
        'figure.facecolor': 'w',
        'figure.edgecolor': 'k',
        'figure.figsize'  : [15,4],
        'text.latex.preamble' : r'\usepackage{amsmath}',
        'text.usetex'     : True
        })
    except:
      pass
    self.SList, self.TList, self.eps = SList, [25, 50], 1e-18
    self.TimeSeries = df.drop(df.columns[np.arange(0, 41)], axis=1)
    self.df  = df.drop(df.columns[np.append([2, 4, 6, 8, 10, 12], np.arange(14, len(df.columns)))], axis=1)
    aux = self.df[self.df.columns[[0]][0]].str.split("T", n=1, expand=True) 
    self.df.insert(0, "yyyy-mm-dd", aux[0])
    self.df[self.df.columns[1]] = aux[1].str.split(":", n=-1, expand = True).apply(lambda x: x[0]+':'+x[1], axis=1)
    self.df.rename(columns={self.df.columns[1]:'hh:mm', 
                            self.df.columns[2]:'S', 
                            self.df.columns[3]:'T',}, inplace=True)    
    self.MAPE = tf.keras.losses.MeanAbsolutePercentageError()
    
  def PlotTraining(self, _loss):
    if len(_loss[0])!=0:
      fig = plt.figure(figsize=(13, 15))
      gs = gridspec.GridSpec(4, ncols=3, width_ratios=[5, 1, 5], wspace=0.03, hspace=0.3)
      ax = fig.add_subplot(gs[0, :])
      contx, conty = 0, 1
      for _n, _label  in enumerate(['CustomLoss', 'Imp', 'Vmp', 'Isc', 'Voc']):
        if _n!=0:
          ax = fig.add_subplot(gs[conty, contx])
          contx+=2
          if contx//3: 
            contx=0
            conty+=1
        ax.plot(_loss[0][:,_n], ".b", label='Training')
        ax.plot(_loss[1][:,_n], ".g", label='Validation')
        ax.axhline(y=_loss[2][_n], color="k", linestyle="-.",label="Test")
        ax.set_ylabel('Mean Absolute Error (\\%)', fontsize=14)
        ax.set_xlabel('epoch (-)', fontsize=14)
        ax.legend(fontsize=14), ax.set_title(_label)
      plt.show() 

  def Boxplot(self, X, X_train, X_val, X_test, X_view, y, y_train, y_val, y_test, y_view):
    fig = plt.figure(figsize=(15, 10), dpi=80, facecolor='w', edgecolor='k')
    gs  = gridspec.GridSpec(nrows=3, ncols=3, width_ratios=[5, 1, 5], wspace=0.03, hspace=0.5)
    contx, conty = 0, 0
    for num, label in enumerate(['Irradiance', 'Temperature', 'Isc (A)', 'Imp (A)', 'Vmp (V)', 'Voc (V)']):
      ax = plt.subplot(gs[conty, contx])
      if num in range(2):
        ax.boxplot([X[:,num], X_train[:,num], X_val[:,num], X_test[:,num]], vert=False)
      else:
        ax.boxplot([y[:,num-2], y_train[:,num-2], y_val[:,num-2], y_test[:,num-2]], vert=False)
      ax.yaxis.set_ticklabels(['total', 'train', 'val', 'test']), ax.set_title(label)
      contx+=2
      if contx==4:
        contx=0
        conty+=1
    plt.show() 

  def SearchCurve(self, T1, S1):
    idx = np.where((self.df[self.df.columns[[3]]]==[T1]).to_numpy()==True)[0]
    aux = np.sqrt(np.abs(self.df[self.df.columns[[2]]].iloc[idx].to_numpy()**2-S1**2))
    idx = idx[np.where(aux==aux.min())[0]][0]
    current = self.TimeSeries.iloc[idx][np.arange(1, self.TimeSeries.iloc[idx][0]+1, dtype=int)].to_numpy()
    voltage = self.TimeSeries.iloc[idx][np.arange(self.TimeSeries.iloc[idx][0]+1, 2*self.TimeSeries.iloc[idx][0]+1, dtype=int)].to_numpy()
    Sx, Tx = self.df.iloc[idx]['S'], self.df.iloc[idx]['T']
    S, T = Sx*np.ones(shape=(voltage.shape[0], 1)), Tx*np.ones(shape=(voltage.shape[0], 1))
    data = np.concatenate([S, T, voltage.reshape((voltage.shape[0], 1))], axis=1)
    return [voltage, current, S, T, Sx, Tx, data]

  
  def SearchDay(self, dayView='2014-01-20'):
    df1 = self.df.loc[self.df[ self.df[self.df.columns[0]]==dayView][self.df.columns[0]].index]
    return [df1[df1.columns[[2,3]]].to_numpy(dtype='float32'), 
            df1[df1.columns[[4,6,7,8]]].to_numpy(dtype='float32'), 
            df1[df1.columns[1]].to_numpy()]

  def CurvesPV_IV(self, curve, params={}, model=None):
    fig = plt.figure(figsize=(15, len(self.SList)*5))
    gs  = gridspec.GridSpec(nrows=len(self.SList), ncols=3, figure=fig, width_ratios=[5, 1, 5], wspace=0.03, hspace=0.5)
    contx, conty = 0, 0
    for T1 in self.TList:
      for S1 in self.SList:
        [voltage, current, S, T, Sx, Tx, data], ax = self.SearchCurve(T1, S1), plt.subplot(gs[conty, contx])
        # Experimental curve
        if curve=='pv':
          yData = current*voltage
          ax.set_ylabel('$p_{pv}$ (W)')
        elif curve=='iv':
          yData=current
          ax.set_ylabel('$i_{pv}$ (A)')
        ax.plot(voltage, yData, label='Experimental curve')
        maxVal = yData.max()
        # Models      
        for m in params:
          Rs, Gp, IL, I0, b = [PVModel(params, model=m).Rs(data[:,0], data[:,1]), 
                               PVModel(params, model=m).Gp(data[:,0]), 
                               PVModel(params, model=m).IL(data[:,0], data[:,1]), 
                               PVModel(params, model=m).I0(data[:,1]), 
                               PVModel(params, model=m).b(data[:,1])]
          Ipv = PVPredict().fun_Ipv(Rs, Gp, IL, I0, b, data[:,2]).numpy()
          if curve=='pv': 
            yData=data[:, 2]*Ipv
          elif curve=='iv':
            yData=Ipv
          ax.plot(data[:, 2], yData, label=params[m]['name'], Linestyle='--')
          maxVal = np.array([maxVal, np.max(yData)]).max()
        # Neural network
        try:
          Rs, Gp, IL, I0, b  = tf.split(model(np.concatenate([S/1000, (T-25)/25], axis=1)), axis=1, num_or_size_splits=5)
          Ipv_DNN = PVPredict().fun_Ipv(Rs, Gp, IL, I0, b, voltage.reshape((voltage.shape[0], 1))).numpy() 
          if curve=='pv':
            yDNN =Ipv_DNN*voltage.reshape((voltage.shape[0], 1))
          elif curve=='iv':
            yDNN=Ipv_DNN
          ax.plot(voltage, yDNN, label='Neural network', Linestyle='--')
          maxVal = np.array([maxVal, np.max(yDNN)]).max()
        except:
          pass
        ax.grid(alpha=0.75), ax.set_ylim([0, np.ceil(maxVal*1.05)])
        ax.set_xlabel('$v_{pv}$ (V)')
        ax.set_title('S: '+str(Sx)+'(W/m$^2$) - T: '+str(Tx) +'(°C)')
        ax.legend(fontsize=12, loc=2), ax.set_xlim([0, np.ceil(voltage.max())])
        conty+=1
        if conty//len(self.SList):
          conty=0
          contx+=2
    plt.show()

  def ErrorPV_IV(self, curve, params={}, model=None, outliers=1e2):  
    fig = plt.figure(figsize=(15, len(self.SList)*7))
    gs  = gridspec.GridSpec(nrows=len(self.SList), ncols=5, figure=fig, width_ratios=[2, 5, 2, 2, 5], wspace=0.03, hspace=0.25)
    MAPEData, contx, conty, contData, positionsBox, labelBox = {}, 0, 0, 0, 0, []
    for T1 in self.TList:
      for S1 in self.SList:
        voltage, current, S, T, Sx, Tx, data = self.SearchCurve(T1, S1)
        if contx == 0:
          ax1, ax2 = plt.subplot(gs[conty, 0]), plt.subplot(gs[conty, 1])
        elif contx == 2:
          ax1, ax2 = plt.subplot(gs[conty, 3]), plt.subplot(gs[conty, 4])
        # Experimental curve
        if curve=='pv':
          yReal = current*voltage
        elif curve=='iv':
          yReal=current       
        # Models
        for m in params:
          Rs, Gp, IL, I0, b = [PVModel(params, model=m).Rs(data[:,0], data[:,1]), 
                               PVModel(params, model=m).Gp(data[:,0]), 
                               PVModel(params, model=m).IL(data[:,0], data[:,1]), 
                               PVModel(params, model=m).I0(data[:,1]), 
                               PVModel(params, model=m).b(data[:,1])]
          Ipv = PVPredict().fun_Ipv(Rs, Gp, IL, I0, b, data[:,2]).numpy()
          if curve=='pv': 
            yData=data[:, 2]*Ipv
          elif curve=='iv':
            yData=Ipv
          error = np.mean(1-(yData/(yReal+self.eps)).reshape(yReal.shape[0], 1), 1)*100
          error = error[np.where(np.abs(error)<outliers)] ## Quita algunos outliers 
          ax1.boxplot(error, vert=True, positions=[positionsBox])
          positionsBox+=1
          labelBox.append(params[m]['label'])
          ax2.hist(error*100, 50, density=False, alpha=0.75, orientation="horizontal", label=params[m]['name'])
          MAPEData[params[m]['name']+' - '+str(contData)] = {'MAPE': np.around(self.MAPE(yReal, yData).numpy(),4),
                                                             'MeanPlot': np.around(error.mean(),4),
                                                             'Sx':Sx, 'Tx':Tx}
        # Neural network
        try:
          Rs, Gp, IL, I0, b  = tf.split(model(np.concatenate([S/1000, (T-25)/25], axis=1)), axis=1, num_or_size_splits=5)
          Ipv_DNN = PVPredict().fun_Ipv(Rs, Gp, IL, I0, b, voltage.reshape((voltage.shape[0], 1))).numpy() 
          if curve=='pv':
            yDNN = Ipv_DNN*voltage.reshape((voltage.shape[0], 1))
          elif curve=='iv':
            yDNN =Ipv_DNN
          error = np.mean(1-(yDNN/(yReal+self.eps)), 1)*100
          error = error[np.where(np.abs(error)<outliers)] ## Quita algunos outliers 
          ax1.boxplot(error, vert=True, positions=[positionsBox])
          positionsBox+=1
          labelBox.append('NN')
          ax2.hist(error*100, 50, density=False, alpha=0.75, orientation="horizontal", label='Neural network')
          MAPEData['Neural network - '+str(contData)] = {'MAPE': np.around(self.MAPE(yReal, yDNN).numpy(),4), 
                                                        #  'MeanPlot': np.around(error.mean(),4), # Activar cuando tenga una buena red
                                                         'Sx':Sx, 'Tx':Tx}
        except:
          pass
        ax1.xaxis.tick_top()
        ax2.legend(fontsize=12, loc=1), ax1.set_xticklabels(labelBox, fontsize=12)
        ax1.set_ylabel('Prediction error (\\%)', fontsize=14)
        ax2.xaxis.set_ticklabels(np.around(plt.xticks()[0]/error.shape[0]*100, 1))
        ax2.yaxis.set_ticks([]), ax2.set_xlabel('Frequency (\\%)', fontsize=14)
        ax2.set_title('S: '+str(Sx)+'(W/m$^2$) - T: '+str(Tx) +'(°C)')        
        conty+=1
        if conty//len(self.SList):
          conty=0
          contx+=2
        contData+=1
    plt.show()
    return MAPEData

  def Error(self, xTest, yTest, params={}, model=None, outliers=1e2):
    fig  = plt.figure(figsize=(15, 10))
    gs  = gridspec.GridSpec(nrows=2, ncols=5, width_ratios=[2, 5, 2, 2, 5], hspace=0.3, wspace=0.03)
    MAPEData, contx, conty, positionsBox, labelBox, yData = {}, 0, 0, 0, [], []
    try: 
      Rs, Gp, IL, I0, b  = tf.split(model(np.concatenate([xTest[:,0].reshape((xTest[:,0].shape[0], 1))/1000, (xTest[:,1].reshape((xTest[:,1].shape[0], 1))-25)/25], axis=1)), axis=1, num_or_size_splits=5)
      yDNN = np.delete(PVPredict().predict(Rs, Gp, IL, I0, b).numpy().astype('float64'), [1, 4], 1)
    except:
      pass
    for m in params:
      Rs, Gp, IL, I0, b = [tf.reshape(k, [k.shape[0], 1]).numpy().astype('float64')  for k in [PVModel(params, model=m).Rs(xTest[:,0], xTest[:,1]), 
                                                                                               PVModel(params, model=m).Gp(xTest[:,0]), 
                                                                                               PVModel(params, model=m).IL(xTest[:,0], xTest[:,1]), 
                                                                                               PVModel(params, model=m).I0(xTest[:,1]), 
                                                                                               PVModel(params, model=m).b(xTest[:,1])]]
      yData.append(np.delete(PVPredict().predict(Rs, Gp, IL, I0, b), [1, 4], 1))
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
        MAPEData[params[m]['name']+' - '+label] = {'MAPE':np.around(self.MAPE(yTest[:,n], yData[m][:,n]).numpy(),4),
                                                       'MeanPlot': np.around(error.mean(),4)}
      try: 
        error = tf.math.reduce_mean(1-np.hsplit(yDNN/yTest.astype('float64'), 4)[n], 1)*100
        error = error.numpy()[np.where(np.abs(error)<outliers)] ## Quita algunos outliers 
        ax1.boxplot(error, vert=True, positions=[positionsBox])
        positionsBox+=1
        labelBox.append('NN')
        ax2.hist(error*100, 50, density=False, alpha=0.75, orientation="horizontal", label='Neural network')
        MAPEData['Neural network - '+label] = {'MAPE':np.around(self.MAPE(yTest[:,n], yDNN[:,n]).numpy(),4),
                                               'MeanPlot': np.around(error.mean(),4)}
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
    return MAPEData

  def Tracking(self, params, model=None, dayView='2014-01-20'):
    xView, yView, Time = self.SearchDay(dayView=dayView)
    a1 = np.linspace(start=1, stop=xView.shape[0], num=xView.shape[0])
    b1 = Time[np.where(a1%6==1)]
    fig = plt.figure(figsize=(13, 20),  constrained_layout=True)
    gs  = gridspec.GridSpec(4, ncols=3, figure=fig, width_ratios=[5, 1, 5], hspace=0.3)
    contx, conty, contData, yData = 0, 0, 0, []
    # Neural network
    try:       
      Rs, Gp, IL, I0, b  = tf.split(model(np.concatenate([xView[:,0].reshape((xView[:,0].shape[0], 1))/1000, (xView[:,1].reshape((xView[:,1].shape[0], 1))-25)/25], axis=1)), axis=1, num_or_size_splits=5)
      yDNN = PVPredict().predict(Rs, Gp, IL, I0, b).numpy()
    except:
      pass
    # Models
    for m in params:
      Rs, Gp, IL, I0, b = [tf.reshape(k, [k.shape[0], 1]).numpy().astype('float64')  for k in [PVModel(params, model=m).Rs(xView[:,0], xView[:,1]), 
                                                                                               PVModel(params, model=m).Gp(xView[:,0]), 
                                                                                               PVModel(params, model=m).IL(xView[:,0], xView[:,1]), 
                                                                                               PVModel(params, model=m).I0(xView[:,1]), 
                                                                                               PVModel(params, model=m).b(xView[:,1])]]
      yData.append(PVPredict().predict(Rs, Gp, IL, I0, b))
    for value, var in enumerate(['Irradiance (W/m$^2$)', 'Temperature (°C)', 'Isc (A)', 
                                 'Vsc (V)', 'Imp (A)', 'Vmp (V)', 'Ioc (A)', 'Voc (V)']):
      ax = fig.add_subplot(gs[conty, contx])
      if conty==0:
        ax.plot(xView[:, value], label='Experimental curve')
      else:
        if var in ['Isc (A)', 'Imp (A)', 'Vmp (V)', 'Voc (V)']:
          ax.plot(yView[:, contData], label='Experimental curve')
          contData+=1
        else:
           ax.plot(np.zeros([yView.shape[0], 1]), label='Experimental curve')
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
