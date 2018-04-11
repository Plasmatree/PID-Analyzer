#!/usr/bin/env python3
import argparse
import logging
import os
import subprocess
import numpy as np
from pandas import read_csv
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from matplotlib.gridspec import GridSpec
from scipy.ndimage.filters import gaussian_filter1d
# ----------------------------------------------------------------------------------
# "THE BEER-WARE LICENSE" (Revision 42):
# <florian.melsheimer@gmx.de> wrote this file. As long as you retain this notice you
# can do whatever you want with this stuff. If we meet some day, and you think
# this stuff is worth it, you can buy me a beer in return. Florian Melsheimer
# ----------------------------------------------------------------------------------
#
#

Version = 'PID-Analyzer 0.2 '

LOG_MIN_BYTES = 500000

class Trace:
    framelen = 2.
    resplen = 0.5
    smooth = 1.         #
    tuk_alpha = 1.0     # alpha of tukey window, if used
    superpos = 32       # sub windowing (superpos windows in framelen)
    threshold = 500.    # threshold for 'high input rate'

    def __init__(self, data):
        self.name = data['name']
        self.time, self.output = self.equalize(data['time'], data['output'])
        self.input = self.equalize(data['time'], self.pid_in(data['input'], data['output'], data['P']))[1]
        self.pidsum = self.equalize(data['time'],data['PIDsum'])[1]#/(self.time[1]-self.time[0])
        self.throttle = self.equalize(data['time'],data['throttle'])[1]
        #self.time_eq, self.input_eq, self.output_eq, self.pidsum_eq = self.equalize()
        self.flen, self.rlen = self.stepcalc(self.time)
        self.time_resp = self.time[0:self.rlen]-self.time[0]
        self.stacks = self.winmaker(self.flen)               # [[time, input, output],]
        self.window = np.hanning(self.flen)#self.tukeywin(self.flen, self.tuk_alpha)
        self.spec_sm, self.avr_t, self.avr_in, self.max_in = self.stack_response(self.stacks)
        self.low_mask, self.high_mask = self.low_high_mask(self.max_in, self.threshold)       #calcs masks for high and low inputs according to threshold
        self.toolow_mask = self.low_high_mask(self.max_in, 20)[1]          #mask for ignoring noisy low input
        self.resp_sm = self.weighted_mode_avr(self.spec_sm, self.toolow_mask, [-0.5,2.5], 600)
        self.resp_low = self.weighted_mode_avr(self.spec_sm, self.low_mask*self.toolow_mask, [-0.5,2.5], 600)
        if self.high_mask.sum()>0:
            self.resp_high = self.weighted_mode_avr(self.spec_sm, self.high_mask*self.toolow_mask, [-0.5,2.5], 600)

    def low_high_mask(self, signal, threshold):
        low = np.copy(signal)

        low[low <=threshold] = 1.
        low[low > threshold] = 0.
        high = -low+1.

        if high.sum<10:     # ignore high pinput that is too short
            high*=0.

        return low, high

    def pid_in(self, pval, gyro, pidp):
        pidin = gyro + pval / (3. * pidp)
        return pidin

    def rate_curve(self, rcin, inmax=500., outmax=800., rate=160.):
        expoin = (np.exp((rcin - inmax) / rate) - np.exp((-rcin - inmax) / rate)) * outmax
        return expoin

    ### makes tukey widow for envelopig
    def tukeywin(self, len, alpha=0.5):
        M = len
        n = np.arange(M - 1.)  #
        if alpha <= 0:
            return np.ones(M)  # rectangular window
        elif alpha >= 1:
            return np.hanning(M)

        # Normal case
        x = np.linspace(0, 1, M)
        w = np.ones(x.shape)

        # first condition 0 <= x < alpha/2
        first_condition = x < alpha / 2
        w[first_condition] = 0.5 * (1 + np.cos(2 * np.pi / alpha * (x[first_condition] - alpha / 2)))

        # second condition already taken care of

        # third condition 1 - alpha / 2 <= x <= 1
        third_condition = x >= (1 - alpha / 2)
        w[third_condition] = 0.5 * (1 + np.cos(2 * np.pi / alpha * (x[third_condition] - 1 + alpha / 2)))

        return w

    ### equalizes time scale
    def equalize(self, time, data):
        data_f = interp1d(time, data)
        newtime = np.linspace(time[0], time[-1], len(time))
        return newtime, data_f(newtime)

    ### calculates frequency and resulting windowlength
    def stepcalc(self, time):
        tstep = (time[1]-time[0])
        freq = 1./tstep
        flen = Trace.framelen * freq
        rlen = Trace.resplen * freq
        return int(flen), int(rlen)

    ### makes stack of windows for deconvolution
    def winmaker(self, flen):
        tlen = len(self.time)
        shift = int(flen/Trace.superpos)
        wins = int(tlen/shift)-Trace.superpos
        stacks={'time':[],'input':[],'output':[],'throttle':[],'pidsum':[], 'gyrodiff':[]}
        for i in np.arange(wins):
            stacks['time'].append(self.time[i * shift:i * shift + flen])
            stacks['input'].append(self.input[i * shift:i * shift + flen])
            stacks['output'].append(self.output[i * shift:i * shift + flen])
            stacks['throttle'].append(self.throttle[i * shift:i * shift + flen])
            stacks['pidsum'].append(self.pidsum[i * shift:i * shift + flen])
        for k in stacks.keys():
            stacks[k]=np.array(stacks[k])
        return stacks

    def wiener_deconvolution(self, input, output, smooth):      # input/output are two-dimensional
        pad = len(input[0]) + (1024 - (len(input[0]) % 1024))   # padding to power of 2, increases transform speed
        input = np.pad(input, [[0,0],[0,pad]], mode='constant')
        output = np.pad(output, [[0, 0], [0, pad]], mode='constant')
        H = np.fft.fft(input, axis=-1, norm='ortho')
        G = np.fft.fft(output,axis=-1, norm='ortho')
        Hcon = np.conj(H)
        deconvolved_sm = np.real(np.fft.ifft(G * Hcon / (H * Hcon + smooth),axis=-1))
        return deconvolved_sm

    def stack_response(self, stacks):
        inp = stacks['input']* self.window
        outp = stacks['output']* self.window
        thr = stacks['throttle']* self.window

        deconvolved_sm = self.wiener_deconvolution(inp, outp, self.smooth)[:,:self.rlen]
        delta_resp = deconvolved_sm.cumsum(axis=1)

        avr_in = np.abs(np.abs(inp)).mean(axis=1)#(np.gradient(np.convolve(inp,[0.1,0.2,0.3,0.2,0.1], mode='valid'))).mean()
        #avr_thr = np.abs(thr).mean(axis=1)
        max_in = np.max(np.abs(inp),axis=1)
        avr_t = stacks['time'].mean(axis=1)
        #plt.show()
        #sortargs= np.argsort(avr_in)
        #plt.figure()
        #plt.pcolormesh(delta_resp, vmin=0,vmax=2)
        #plt.pcolormesh(avr_in[sortargs],self.time_resp,delta_resp[sortargs].transpose(), vmin=0,vmax=2)
        #plt.show()

        return delta_resp, avr_t, avr_in, max_in

    ### finds the most common trace and std
    def weighted_mode_avr(self, values, weights, vertrange, vertbins):
        threshold = 0.5  # threshold for std calculation
        filt_width = 7 # width of gaussian smoothing for hist data

        resp_y = np.linspace(vertrange[0],vertrange[-1],vertbins)
        times = np.repeat(np.array([self.time_resp]), len(values), axis=0)
        weights = np.repeat(weights, len(values[0]))
        hist2d = np.histogram2d(times.flatten(), values.flatten(),
                                range=[[self.time_resp[0],self.time_resp[-1]], vertrange],
                                bins=[len(times[0]), vertbins], weights=weights)[0].transpose()  # , weights=weights.flatten()
        hist2d /= hist2d.max(0)
        hist2d = gaussian_filter1d(hist2d, filt_width, axis=0, mode='constant')
        hist2d /= np.max(hist2d, 0)

        hist2d_sm = np.copy(hist2d)
        pixelpos=np.repeat(resp_y.reshape(len(resp_y),1),len(times[0]),axis=1)
        #print pixelpos
        #avrmodes = np.argsort(hist2d, axis=0)[-avr_num:] / (vertbins/(vertrange[-1]-vertrange[0]))+vertrange[0]
        #avrmode = np.average(avrmodes, 0)
        avr = np.average(pixelpos, 0, weights=hist2d*hist2d*hist2d)

        # only used for monochrome error width
        hist2d[hist2d <= threshold] = 0.
        hist2d[hist2d > threshold] = 0.5/(vertbins/(vertrange[-1]-vertrange[0]))

        std = np.sum(hist2d, 0)

        return avr, std, [self.time_resp,resp_y,hist2d_sm]


    ### calculates weighted avverage and resulting errors
    def weighted_avg_and_std(self, values, weights):
        average = np.average(values, axis=0, weights=weights)
        variance = np.average((values - average) ** 2, axis=0, weights=weights)
        return (average, np.sqrt(variance))

class CSV_log:

    def __init__(self, fpath, name, headdict):
        self.file = fpath
        self.name = name
        self.headdict = headdict

        self.data = self.readcsv(self.file)

        logging.info('Processing:')
        self.traces = self.find_traces(self.data)
        self.roll, self.pitch, self.yaw = self.__analyze()
        self.fig = self.plot_all([self.roll, self.pitch, self.yaw])

    def plot_all(self, traces, style='fancy'): #style='fancy' gives 2d hist for response
        fig = plt.figure(self.headdict['logNum'], figsize=(18, 9))
        for i, tr in enumerate(traces):

            gs1 = GridSpec(24, 3*10, wspace=0.6, hspace=0.7, left=0.04, right=1., bottom=0.05, top=0.97)
            #gs1.update(left=float(i)/3., right=(i+1.)/3., wspace=0.0)

            ax0 = plt.subplot(gs1[0:7, i*10:i*10+9])
            plt.title(tr.name)
            plt.plot(tr.time, tr.output, label=tr.name + ' gyro')
            plt.plot(tr.time, tr.input, label=tr.name + ' loop input')
            # plt.plot(tr.avr_t, tr.weights*np.max(tr.input), label='weights')
            plt.ylabel('degrees/second')
            ax0.get_yaxis().set_label_coords(-0.1, 0.5)
            plt.grid()
            tracelim = np.max([np.abs(tr.output),np.abs(tr.input)])
            plt.ylim([-tracelim*1.1, tracelim*1.1])
            plt.legend(loc=1)
            plt.setp(ax0.get_xticklabels(), visible=False)

            ax1 = plt.subplot(gs1[7:9, i*10:i*10+9], sharex=ax0)
            plt.hlines(self.headdict['tpa_percent'], tr.avr_t[0], tr.avr_t[-1], label='tpa', colors='red', alpha=0.5)
            plt.fill_between(tr.time, 0., tr.throttle, label='throttle', color='grey', alpha=0.2)
            plt.ylabel('throttle %')
            ax1.get_yaxis().set_label_coords(-0.1, 0.5)
            plt.grid()
            plt.ylim([0, 100])
            plt.legend(loc=1)
            plt.setp(ax1.get_xticklabels(), visible=False)

            ax2 = plt.subplot(gs1[9:16, i*10:i*10+9], sharex=ax0)
            plt.pcolormesh(tr.avr_t, tr.time_resp, np.transpose(tr.spec_sm), vmin=0, vmax=2.)
            plt.ylabel('response time in s')
            ax2.get_yaxis().set_label_coords(-0.1, 0.5)
            plt.xlabel('log time in s')
            plt.xlim([tr.avr_t[0], tr.avr_t[-1]])

            if style=='fancy':
                theCM = plt.cm.get_cmap('Blues')
                theCM._init()
                alphas = np.abs(np.linspace(0., 0.5, theCM.N))
                theCM._lut[:-3,-1] = alphas
                ax3 = plt.subplot(gs1[17:, i*10:i*10+9])
                plt.contourf(*tr.resp_low[2], cmap=theCM, linestyles=None, antialiased=True, levels=np.linspace(tr.resp_low[2][2].min(),tr.resp_low[2][2].max(),20))
                #plt.pcolormesh(tr.time_resp, np.linspace(-0.5, 2.5,600), tr.resp_low[2], cmap=theCM, antialiased=True)
                #blue_patch = mpatches.Patch(color=theCM(0.9), label=tr.name + ' step response ' + '(<' + str(int(Trace.threshold)) + ') ' + ' PID ' +
                #                                                           self.headdict[tr.name + 'PID'])
                #plt.legend(handles=[blue_patch])
                plt.plot(tr.time_resp, tr.resp_low[0],
                         label=tr.name + ' step response ' + '(<' + str(int(Trace.threshold)) + ') '
                               + ' PID ' + self.headdict[tr.name + 'PID'])
                #plt.fill_between(tr.time_resp, tr.resp_low[0] - tr.resp_low[1], tr.resp_low[0] + tr.resp_low[1], alpha=0.1)


                if tr.high_mask.sum() > 0:
                    theCM = plt.cm.get_cmap('Oranges')
                    theCM._init()
                    alphas = np.abs(np.linspace(0., 0.5, theCM.N))
                    theCM._lut[:-3,-1] = alphas
                    plt.contourf(*tr.resp_high[2], cmap=theCM, linestyles=None, antialiased=True, levels=np.linspace(tr.resp_high[2][2].min(),tr.resp_high[2][2].max(),20))
                    #plt.pcolormesh(tr.time_resp,np.linspace(-0.5, 2.5,600), tr.resp_high[2], cmap=theCM, antialiased=True)
                    #orage_patch = mpatches.Patch(color=theCM(0.9), alpha=0.5, label=tr.name + ' step response ' + '(>' + str(int(Trace.threshold)) + ') ' + ' PID ' +
                    #           self.headdict[tr.name + 'PID'])
                    #plt.legend(handles=[blue_patch, orage_patch])
                    plt.plot(tr.time_resp, tr.resp_high[0],
                         label=tr.name + ' step response ' + '(<' + str(int(Trace.threshold)) + ') '
                               + ' PID ' + self.headdict[tr.name + 'PID'])
                    #plt.fill_between(tr.time_resp, tr.resp_high[0] - tr.resp_high[1], tr.resp_high[0] + tr.resp_high[1],
                    #                 alpha=0.1)
                plt.xlim([-0.001,0.501])


            else:
                ax3 = plt.subplot(gs1[17:, i*10:i*10+9])
                plt.plot(tr.time_resp, tr.resp_low[0],
                         label=tr.name + ' step response ' + '(<' + str(int(Trace.threshold)) + ') ' + ' PID ' +
                               self.headdict[tr.name + 'PID'])
                plt.fill_between(tr.time_resp, tr.resp_low[0] - tr.resp_low[1], tr.resp_low[0] + tr.resp_low[1], alpha=0.1)

                if tr.resp_high[0].sum() > 10:
                    plt.plot(tr.time_resp, tr.resp_high[0],
                             label=tr.name + ' step response ' + '(>' + str(int(Trace.threshold)) + ') ' + ' PID ' +
                                   self.headdict[tr.name + 'PID'])
                    plt.fill_between(tr.time_resp, tr.resp_high[0] - tr.resp_high[1], tr.resp_high[0] + tr.resp_high[1],
                                     alpha=0.1)
            plt.legend(loc=1)
            plt.ylim([0., 2])
            plt.ylabel('strength')
            ax3.get_yaxis().set_label_coords(-0.1, 0.5)
            plt.xlabel('response time in s')

            plt.grid()
        meanfreq = 1./(traces[0].time[1]-traces[0].time[0])
        ax4 = plt.subplot(gs1[:, -1])
        t = Version+"| Betaflight: Version "+self.headdict['version']+' | Craftname: '+self.headdict['craftName']+\
            ' | meanFreq: '+str(int(meanfreq))+' | rcRate/Expo: '+self.headdict['rcRate']+'/'+ self.headdict['rcExpo']+'\nrcYawRate/Expo: '+self.headdict['rcYawRate']+'/' \
            +self.headdict['rcYawExpo']+' | deadBand: '+self.headdict['deadBand']+' | yawDeadBand: '+self.headdict['yawDeadBand'] \
            +' | Throttle min/tpa/max: ' + self.headdict['minThrottle']+'/'+self.headdict['tpa_breakpoint']+'/'+self.headdict['maxThrottle'] \
            + ' | dynThrPID: ' + self.headdict['dynThrottle']+ '| D-TermSP: ' + self.headdict['dTermSetPoint']+'| vbatComp: ' + self.headdict['vbatComp']

        plt.text(0.5, 0, t, ha='left', rotation=90, color='grey', alpha=0.5, fontsize=8)
        ax4.axis('off')
        plt.savefig(self.file[:-13] + self.name + '_' + str(self.headdict['logNum'])+'.png')
        #plt.show()
        #plt.cla()
        #plt.clf()
        return fig

    def __analyze(self):
        analyzed = []
        for t in self.traces:
            logging.info(t['name'] + '...   ')
            analyzed.append(Trace(t))
            logging.info('\tDone!')
        return analyzed

    def readcsv(self, fpath):
        logging.info('Reading log '+str(self.headdict['logNum'][0])+'...   ')
        datdic = {}
        data = read_csv(fpath, header=0, skipinitialspace=1)
        datdic.update({'time_us': data['time (us)'].values * 1e-6})
        datdic.update({'throttle': data['rcCommand[3]'].values * 1e-6})         #### ???

        #acc = []
        for i in ['0', '1', '2']:
            #acc.append(data['accSmooth[' + i+']'].values)
            datdic.update({'PID sum' + i: data['axisP[' + i+']'].values+data['axisI[' + i+']'].values+data['axisD[' + i+']'].values})
            datdic.update({'PID loop in' + i: data['axisP[' + i+']'].values})
            if 'gyroADC[0]' in data.keys():
                datdic.update({'gyroData' + i: data['gyroADC[' + i+']'].values})
            elif 'gyroData[0]' in data.keys():
                datdic.update({'gyroData' + i: data['gyroData[' + i+']'].values})
            else:
                logging.warning('No gyro trace found!')

        #plt.figure()
        #for a in acc:
        #    plt.plot(a/2048.)
        #plt.plot(data['rcCommand[3]'].values-1000.)
        #thr =data['rcCommand[3]'].values-1000.
        #acce=data['accSmooth[2]'].values/2048.
        #print thr

        #H = np.fft.fft(np.array(thr), norm='ortho')
        #G = np.fft.fft(np.array(acce), norm='ortho')
        #Hcon = np.conj(H)
        # deconvolved = np.real(np.fft.ifft(G*Hcon / (H * Hcon + 0.), norm='ortho',axis=1))[:,:self.rlen]
        #deconvolved_sm = np.real(np.fft.ifft(G * Hcon / (H * Hcon + 100.)))
        #plt.figure()
        #plt.plot(deconvolved_sm.cumsum())
        #print deconvolved_sm

        #plt.show()
        logging.info('\tDone!')

        return datdic


    def find_traces(self, dat):
        time = self.data['time_us']
        throttle = dat['throttle']

        throt = (throttle-0.001)*(float(self.headdict['maxThrottle'])-float(self.headdict['minThrottle']))*100.
        self.headdict.update({'tpa_percent':100.*(float(self.headdict['tpa_breakpoint'])-float(self.headdict['minThrottle']))/
                                            (float(self.headdict['maxThrottle'])-float(self.headdict['minThrottle']))})

        traces = [{'name':'roll'},{'name':'pitch'},{'name':'yaw'}]

        for i, dic in enumerate(traces):
            dic.update({'time':time})
            dic.update({'input':dat['PID loop in'+str(i)]})
            dic.update({'output':dat['gyroData'+str(i)]})
            dic.update({'PIDsum':dat['PID sum'+str(i)]})
            dic.update({'P':float((self.headdict[dic['name']+'PID']).split(',')[0])*0.01})
            dic.update({'throttle':throt})

        return traces


class BB_log:
    def __init__(self, log_file_path, name, blackbox_decode):
        self.blackbox_decode_bin_path = blackbox_decode
        self.tmp_dir = os.path.join(os.path.dirname(log_file_path), name or 'tmp')
        if not os.path.isdir(self.tmp_dir):
            os.makedirs(self.tmp_dir)
        self.name = name

        self.loglist = self.decode(log_file_path)
        self.heads = self.beheader(self.loglist)
        self.figs = self._csv_iter(self.heads)

        self.deletejunk(self.loglist)

    def deletejunk(self, loglist):
        for l in loglist:
            os.remove(l)
            os.remove(l[:-3]+'01.csv')
            try:
                os.remove(l[:-3]+'01.event')
            except:
                logging.warning('No .event file of '+l+' found.')
        return

    def _csv_iter(self, heads):
        figs = []
        for h in heads:
            figs.append(CSV_log(h['tempFile'][:-3]+'01.csv', self.name, h).fig)
        return figs

    def beheader(self, loglist):
        heads = []
        for i, bblog in enumerate(loglist):
            log = open(os.path.join(self.tmp_dir, bblog), 'r')
            lines = log.readlines()

            headsdict = {'tempFile'     :'',
                         'dynThrottle' :'',
                         'craftName'   :'',
                         'version'     :'',
                         'dates'        :'',
                         'rcRate'      :'',
                         'rcExpo'       :'',
                         'rcYawExpo'    :'',
                         'rcYawRate'   :'',
                         'rates'        :'',
                         'rollPID'     :'',
                         'pitchPID'    :'',
                         'yawPID'      :'',
                         'deadBand'    :'',
                         'yawDeadBand' :'',
                         'logNum'       :'',
                         'tpa_breakpoint':'',
                         'minThrottle':'',
                         'maxThrottle': '',
                         'tpa_percent':'',
                         'dTermSetPoint':'',
                         'vbatComp':''}

            headsdict['tempFile'] = bblog

            for l in lines:
                #print l
                headsdict['logNum'] = str(i)
                if 'rcRate:' in l:
                    headsdict['rcRate'] = l[9:-1]
                elif 'rc_rate:' in l:
                    headsdict['rcRate'] = l[10:-1]

                elif 'rcExpo:' in l:
                    headsdict['rcExpo']=l[9:-1]
                elif 'rc_expo:' in l:
                    headsdict['rcExpo']=l[10:-1]

                elif 'rcYawRate:' in l:
                    headsdict['rcYawRate']=l[12:-1]
                elif 'rc_rate_yaw:' in l:
                    headsdict['rcYawRate']=l[14:-1]
                elif 'rcYawExpo:' in l:
                    headsdict['rcYawExpo']=l[12:-1]
                elif 'rc_expo_yaw:' in l:
                    headsdict['rcYawExpo']=l[14:-1]

                elif 'Firmware date:' in l:
                    headsdict['date']=l[16:-1]
                elif 'Firmware revision:' in l:
                    headsdict['version']=l[20:-1]
                elif 'Craft name:' in l:
                    headsdict['craftName']=l[13:-1]
                elif 'dynThrPID:' in l:
                    headsdict['dynThrottle']=l[12:-1]
                elif 'rates:' in l:
                    headsdict['rate']=l[8:-1]
                elif 'rollPID:' in l:
                    headsdict['rollPID']=l[10:-1]
                elif 'pitchPID:' in l:
                    headsdict['pitchPID']=l[11:-1]
                elif 'yawPID:' in l:
                    headsdict['yawPID']=l[9:-1]
                elif ' deadband:' in l:
                    headsdict['deadBand']=l[11:-1]
                elif 'yaw_deadband:' in l:
                    headsdict['yawDeadBand']=l[15:-1]
                elif 'tpa_breakpoint:' in l:
                    headsdict['tpa_breakpoint']=l[17:-1]
                elif 'minthrottle:' in l:
                    headsdict['minThrottle']=l[14:-1]
                elif 'maxthrottle:' in l:
                    headsdict['maxThrottle']=l[14:-1]

                elif 'dtermSetpointWeight:' in l:
                    headsdict['dTermSetPoint']=l[22:-1]
                elif 'dterm_setpoint_weight:' in l:
                    headsdict['dTermSetPoint']=l[24:-1]

                elif 'vbat_pid_compensation:' in l:
                    headsdict['vbatComp']=l[24:-1]
                elif 'vbat_pid_gain:' in l:
                    headsdict['vbatComp']=l[16:-1]

            heads.append(headsdict)
        return heads

    def decode(self, fpath):
        """Splits out one BBL per recorded session and converts each to CSV."""
        with open(fpath, 'r') as text_log_view:
            # The first line of the overall BBL file re-appears at the beginning
            # of each recorded session.
            firstline = text_log_view.readlines()[0]
        with open(fpath, 'rb') as binary_log_view:
            content = binary_log_view.read()

        split = content.split(firstline)
        bbl_sessions = []
        for i in range(len(split)):
            path_root, path_ext = os.path.splitext(os.path.basename(fpath))
            temp_path = os.path.join(
                self.tmp_dir, '%s_temp%d%s' % (path_root, i, path_ext))
            with open(temp_path, 'wb') as newfile:
                newfile.write(firstline+split[i])
            bbl_sessions.append(temp_path)

        loglist = []
        for bbl_session in bbl_sessions:
            size_bytes = os.path.getsize(os.path.join(self.tmp_dir, bbl_session))
            if size_bytes > LOG_MIN_BYTES:
                try:
                    msg = subprocess.check_call([self.blackbox_decode_bin_path, bbl_session])
                    loglist.append(bbl_session)
                except:
                    logging.error(
                        'Error in Blackbox_decode of %r' % bbl_session, exc_info=True)
            else:
                # There is often a small bogus session at the start of the file.
                logging.warning(
                    'Ignoring BBL session %r, %dB < %dB.'
                    % (bbl_session, size_bytes, LOG_MIN_BYTES))
                os.remove(bbl_session)
        return loglist


def run_analysis(log_file_path, plot_name, blackbox_decode):
    test = BB_log(log_file_path, plot_name, blackbox_decode)
    logging.info('Analysis complete, showing plot. (Close plot to exit.)')
    plt.show()


def strip_quotes(filepath):
    """Strips single or double quotes from a string."""
    if filepath[-1]=='"' or filepath[-1]=="'":
        return filepath[1:-1]
    else:
        return filepath


def clean_path(path):
    return os.path.abspath(os.path.expanduser(strip_quotes(path)))


if __name__ == "__main__":
    logging.basicConfig(
        format='%(levelname)s %(asctime)s %(filename)s:%(lineno)s: %(message)s',
        level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-l', '--log', action='append',
        help='BBL log file(s) to analyse. Omit for interactive prompt.')
    parser.add_argument('-n', '--name', default='', help='Plot name.')
    parser.add_argument(
        '--blackbox_decode',
        default=os.path.join(os.getcwd(), 'Blackbox_decode.exe'),
        help='Path to Blackbox_decode.exe.')
    args = parser.parse_args()

    blackbox_decode_path = clean_path(args.blackbox_decode)
    if not os.path.isfile(blackbox_decode_path):
        parser.error(
            ('Could not find Blackbox_decode.exe (used to generate CSVs from '
             'your BBL file) at %s. You may need to install it from '
             'https://github.com/cleanflight/blackbox-tools/releases.')
            % blackbox_decode_path)
    logging.info('Decoding with %r' % blackbox_decode_path)

    logging.info(Version)
    logging.info('Hello Pilot!')

    if args.log:
        for log_path in args.log:
            run_analysis(clean_path(log_path), args.name, args.blackbox_decode)
    else:
        while True:
            logging.info('Interactive mode: enter log file, or type ^C (control-C) when done.')
            try:
                raw_path = input('BBL log file path (type or drag in): ')
                name = input('Plot name [default = %r]: ' % args.name) or args.name
            except (EOFError, KeyboardInterrupt):
                logging.info('Goodbye!')
                break
            run_analysis(clean_path(raw_path), name, args.blackbox_decode)
