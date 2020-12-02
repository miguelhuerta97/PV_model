# Plotting module 
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import numpy as np
import pandas as pd 
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.model_selection import train_test_split

class FunctionPlotting:
  def __init__(self, df, SList=[100, 200, 400, 600, 800, 1000, 1100]): 
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
      })
    try:
      plt.rcParams.update({
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
        ax.axhline(y=_lossTest[_n].numpy(), color="k", linestyle="-.",label="Test")
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
          ax.set_ylabel('$p_{pv}$ $(W)$')
        elif curve=='iv':
          yData=current
          ax.set_ylabel('$i_{pv}$ $(A)$')
        ax.plot(voltage, yData, label='Experimental curve')
        maxVal = yData.max()
        # Neural network
        try:
          Ipv_DNN = modelPV.predict_Ipv(model(np.concatenate([S/1000, (T-25)/25], axis=1)), voltage.reshape((voltage.shape[0], 1)))
          if curve=='pv':
            yDNN =Ipv_DNN*voltage.reshape((voltage.shape[0], 1))
          elif curve=='iv':
            yDNN=Ipv_DNN
          ax.plot(voltage, yDNN, label='Neural network', Linestyle='--')
          maxVal = np.array([maxVal, np.max(yDNN)]).max()
        except:
          pass
        # Models
        for m in params:
          Ipv = PVModel(params, model=m).predict_Ipv(data)
          if curve=='pv': 
            yData=data[:, 2]*Ipv
          elif curve=='iv':
            yData=Ipv
          ax.plot(data[:, 2], yData, label=params[m]['name'], Linestyle='--')
          maxVal = np.array([maxVal, np.max(yData)]).max()
        ax.grid(alpha=0.75), ax.set_ylim([0, np.ceil(maxVal*1.05)])
        ax.set_xlabel('$v_{pv}$ $(V)$')
        ax.set_title('S: '+str(Sx)+'(W/m$^2$) - T: '+str(Tx) +'(°C)')
        ax.legend(), ax.set_xlim([0, np.ceil(voltage.max())])
        conty+=1
        if conty//len(self.SList):
          conty=0
          contx+=2
    plt.show()



  
  def SearchDay(self, dayView='2014-01-20'):
    df1 = self.df.loc[self.df[ self.df[self.df.columns[0]]==dayView][self.df.columns[0]].index]
    return [df1[df1.columns[[2,3]]].to_numpy(dtype='float32'), 
            df1[df1.columns[[4,6,7,8]]].to_numpy(dtype='float32'), 
            df1[df1.columns[1]].to_numpy()]
            
  def Tracking(self, params, model=None, dayView='2014-01-20'):
    xView, yView, Time = self.SearchDay( dayView=dayView)
    a = np.linspace(start=1, stop=xView.shape[0], num=xView.shape[0])
    b = Time[np.where(a%6==1)]
    fig = plt.figure(figsize=(13, 20),  constrained_layout=True)
    gs  = gridspec.GridSpec(4, ncols=3, figure=fig, width_ratios=[5, 1, 5], hspace=0.3)
    contx, conty, contData = 0, 0, 0
    # Neural network
    try:       
      yDNN = modelPV.predict(model(np.concatenate([xView[:,0].reshape((xView[:,0].shape[0], 1))/1000, (xView[:,1].reshape((xView[:,1].shape[0], 1))-25)/25], axis=1)))
    except:
      pass
    # Models
    yData = [PVModel(params, model=model).predict(xView)  for model in params]
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
        try: 
          ax.plot(yDNN[:, value-2], label='Neural network')
        except:
          pass
        for model in params:
          ax.plot(yData[model][:, value-2], label=params[model]['name'])
      contx+=2
      if contx//4:
        contx=0
        conty+=1 
      ax.legend(loc='upper right', fontsize=14)
      ax.set_xlim([a.min(), a.max()])
      ax.set_ylabel(var, fontsize=14), ax.set_xticks(np.where(a%6==1)[0])
      ax.set_xticklabels(b, rotation=70), ax.grid(color='black', ls = '-.', lw = 0.1)
    fig.suptitle('Day: '+dayView+'\n', fontsize=18)
    plt.show()

  
  def ErrorPV_IV(self, curve, params={}, model=None, outliers=1e2):  
    fig = plt.figure(figsize=(15, len(self.SList)*7))
    gs  = gridspec.GridSpec(nrows=len(self.SList), ncols=5, figure=fig, width_ratios=[1, 3, 1, 1, 3], wspace=0.03, hspace=0.25)
    MAPEData, contx, conty, contData= {}, 0, 0, 0
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
        # Neural network
        try:
          Ipv_DNN = modelPV.predict_Ipv(model(np.concatenate([S/1000, (T-25)/25], axis=1)), voltage.reshape((voltage.shape[0], 1)))
          if curve=='pv':
            yDNN = Ipv_DNN*voltage.reshape((voltage.shape[0], 1))
          elif curve=='iv':
            yDNN =Ipv_DNN
          error = np.mean(1-(yDNN/(yReal+self.eps)), 1)*100
          error = error[np.where(np.abs(error)<outliers)] ## Quita algunos outliers 
          ax1.boxplot(error, vert=True)
          ax2.hist(error*100, 50, density=False, alpha=0.75, orientation="horizontal", label='Neural network')
          MAPEData['Neural network - '+str(contData)] = {'MAPE': np.around(self.MAPE(yReal, yDNN).numpy(),4), 
                                                        #  'MeanPlot': np.around(error.mean(),4), # Activar cuando tenga una buena red
                                                         'Sx':Sx, 'Tx':Tx}
        except:
          pass
        # Models
        for m in params:
          Ipv = PVModel(params, model=m).predict_Ipv(data)
          if curve=='pv': 
            yData=data[:, 2]*Ipv
          elif curve=='iv':
            yData=Ipv
          error = np.mean(1-(yData/(yReal+self.eps)).reshape(yReal.shape[0], 1), 1)*100
          error = error[np.where(np.abs(error)<outliers)] ## Quita algunos outliers 
          ax1.boxplot(error, vert=True)
          ax2.hist(error*100, 50, density=False, alpha=0.75, orientation="horizontal", label=params[m]['name'])
          MAPEData[params[m]['name']+' - '+str(contData)] = {'MAPE': np.around(self.MAPE(yReal, yData).numpy(),4),
                                                             'MeanPlot': np.around(error.mean(),4),
                                                             'Sx':Sx, 'Tx':Tx}
        ax2.legend(fontsize=14)
        ax1.xaxis.set_ticks([]), ax1.set_ylabel('Prediction error (\\%)', fontsize=14)
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
    gs  = gridspec.GridSpec(nrows=2, ncols=5, width_ratios=[1, 3, 1, 1, 3], hspace=0.3, wspace=0.03)
    MAPEData, contx, conty = {}, 0, 0
    try: 
      yDNN = np.delete( modelPV.predict(model(np.concatenate([xTest[:,0].reshape((xTest[:,0].shape[0], 1))/1000, (xTest[:,1].reshape((xTest[:,1].shape[0], 1))-25)/25], axis=1))).numpy().astype('float64'), [1, 4], 1)
    except:
      pass
    yData = [np.delete(PVModel(params, model=model).predict(xTest), [1, 4], 1) for model in params]
    for n, label in enumerate(['Isc (A)', 'Imp (A)', 'Vmp (V)', 'Voc (V)']):
      if contx == 0:
        ax1, ax2 = plt.subplot(gs[conty, 0]), plt.subplot(gs[conty, 1])
      elif contx == 1:
        ax1, ax2 = plt.subplot(gs[conty, 3]), plt.subplot(gs[conty, 4])
      try: 
        error = tf.math.reduce_mean(1-np.hsplit(yDNN/yTest.astype('float64'), 4)[n], 1)*100
        error = error.numpy()[np.where(np.abs(error)<outliers)] ## Quita algunos outliers 
        ax1.boxplot(error, vert=True)
        ax2.hist(error*100, 50, density=False, alpha=0.75, orientation="horizontal", label='Neural network')
        MAPEData['Neural network - '+label] = {'MAPE':np.around(self.MAPE(yTest[:,n], yDNN[:,n]).numpy(),4),
                                               'MeanPlot': np.around(error.mean(),4)}
      except:
        pass
      for model in params:
        error = tf.math.reduce_mean(1-np.hsplit(yData[model]/yTest.astype('float64'), 4)[n], 1)*100
        error = error.numpy()[np.where(np.abs(error)<outliers)] ## Quita algunos outliers 
        ax1.boxplot(error, vert=True)
        ax2.hist(error*100, 50, density=False, alpha=0.75, orientation="horizontal", label=params[model]['name'])
        MAPEData[params[model]['name']+' - '+label] = {'MAPE':np.around(self.MAPE(yTest[:,n], yData[model][:,n]).numpy(),4),
                                                       'MeanPlot': np.around(error.mean(),4)}

      ax1.xaxis.set_ticks([]), ax1.set_ylabel('Prediction error (\\%)', fontsize=14)
      ax2.xaxis.set_ticklabels(np.around(plt.xticks()[0]/error.shape[0]*100, 1))
      ax2.yaxis.set_ticks([]), ax2.set_xlabel('Frequency (\\%)', fontsize=14), 
      ax2.set_title(label), 
      ax2.legend(loc='upper right', fontsize=14)
      contx+=1
      if contx//2:
        contx=0
        conty+=1    
    plt.show()
    return MAPEData
