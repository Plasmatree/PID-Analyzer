import os
import numpy as np
from pandas import read_csv
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from matplotlib.gridspec import GridSpec


# ----------------------------------------------------------------------------------
# "THE BEER-WARE LICENSE" (Revision 42):
# <florian.melsheimer@gmx.de> wrote this file. As long as you retain this notice you
# can do whatever you want with this stuff. If we meet some day, and you think
# this stuff is worth it, you can buy me a beer in return. Florian Melsheimer
# ----------------------------------------------------------------------------------
#
#

Version = 'PID-Analyzer 0.12 '

class Trace:
    framelen = 1.5
    resplen = 0.5
    smooth = 1.
    tuk_alpha = 0.5
    superpos = 16
    threshold = 500.
    inweigh = 2.

    def __init__(self, data):
        self.name = data[0]
        self.time = data[1]
        self.input = self.pid_in(data[2], data[3], data[4])
        self.output = data[3]
        self.throttle = data[5]
        self.time_eq, self.input_eq, self.output_eq = self.equalize()
        self.flen, self.rlen = self.stepcalc(self.time_eq)
        self.time_resp = self.time_eq[0:self.rlen]-self.time_eq[0]
        self.stack = self.winmaker(self.flen)               # [[time, input, output],]
        self.tukey = self.tukeywin(self.flen, self.tuk_alpha)
        self.spec_sm, self.avr_t, self.avr_in, self.max_in = self.stack_deconvolve(self.stack)
        self.resp_sm, self.resp_low, self.resp_high = self.weight_response(self.spec_sm, self.avr_in, self.max_in)


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
    def equalize(self):
        inp_f = interp1d(self.time, self.input)
        outp_f = interp1d(self.time, self.output)
        newtime = np.linspace(self.time[0], self.time[-1], len(self.time))
        return newtime, inp_f(newtime), outp_f(newtime)

    ### calculates frequency and resulting windowlength
    def stepcalc(self, time):
        tstep = (time[1]-time[0])
        freq = 1./tstep
        flen = Trace.framelen * freq
        rlen = Trace.resplen * freq
        return int(flen), int(rlen)

    ### makes stack of windows for deconvolution
    def winmaker(self, flen):
        tlen = len(self.time_eq)
        stack = []
        shift = int(flen/Trace.superpos)
        wins = int(tlen/shift)-Trace.superpos
        for i in np.arange(wins):
            stack.append([self.time_eq[i * shift:i * shift + flen],
                          self.input_eq[i * shift:i * shift + flen],
                          self.output_eq[i * shift:i * shift + flen]])
        return np.array(stack)

    def stack_deconvolve(self, stack):
        inp = stack[:,1]* self.tukey
        outp = stack[:,2]* self.tukey
        pad = 8000 - len(inp[0])%8000
        inp = np.pad(inp, [[0,0],[0,pad]], mode='constant')
        outp = np.pad(outp, [[0, 0], [0, pad]], mode='constant')

        H = np.fft.fft(inp, axis=1, norm='ortho')
        G = np.fft.fft(outp,axis=1, norm='ortho')
        Hcon = np.conj(H)
        #deconvolved = np.real(np.fft.ifft(G*Hcon / (H * Hcon + 0.), norm='ortho',axis=1))[:,:self.rlen]
        deconvolved_sm = np.real(np.fft.ifft(G * Hcon / (H * Hcon + self.smooth),axis=1))[:,:self.rlen]

        delta_resp = deconvolved_sm.cumsum(axis=1)

        avr_in = np.abs(inp).mean(axis=1)#(np.gradient(np.convolve(inp,[0.1,0.2,0.3,0.2,0.1], mode='valid'))).mean()
        max_in = np.max(np.abs(inp),axis=1)
        avr_t = stack[:,0].mean(axis=1)

        return delta_resp, avr_t, avr_in, max_in

    ### calculates weighted avverage and resulting errors
    def weighted_avg_and_std(self, values, weights):
        average = np.average(values, axis=0, weights=weights)
        variance = np.average((values - average) ** 2, axis=0, weights=weights)
        return (average, np.sqrt(variance))

    ### wheight spectrogramm by rc_in and quality
    def weight_response(self, spec, avr_in, max_in):

        high_mask = np.clip(max_in - Trace.threshold, 0., 1.)
        low_mask = -high_mask + 1.
        ### average response
        resp_low = np.average(spec, axis=0, weights=avr_in*low_mask)

        if np.sum(high_mask) > 10:
            resp_high = np.average(spec, axis=0, weights=avr_in*high_mask)
        else:
            resp_high = np.zeros_like(resp_low)

        td = spec.std(axis=0)
        ### deviation from average per slice

        fitness_low = (spec - resp_low) ** 2
        fitness_high = (spec - resp_high) ** 2

        ### wheighting new average by quality of slice response

        weights_low = np.sqrt(avr_in*low_mask)**Trace.inweigh / np.sqrt(fitness_low.sum(axis=1))
        weights_high = np.sqrt(avr_in*high_mask) ** Trace.inweigh / np.sqrt(fitness_high.sum(axis=1))

        self.weights = weights_high+weights_low

        spec_w, std_w = self.weighted_avg_and_std(spec, self.weights)
        spec_low, std_low = self.weighted_avg_and_std(spec, weights_low)

        if np.sum(high_mask) > 10:
            spec_high, std_high = self.weighted_avg_and_std(spec, weights_high)
        else:
            spec_high, std_high = np.zeros_like(avr_in), np.zeros_like(avr_in)

        return (spec_w, std_w), (spec_low, std_low), (spec_high, std_high)

class CSV_log:

    def __init__(self, fpath, name, headdict):
        self.file = fpath
        self.name = name
        self.headdict = headdict

        self.data = self.readcsv(self.file)

        print 'Processing:'
        self.traces = self.find_traces(self.data)
        self.roll, self.pitch, self.yaw = self.__analyze()
        self.fig = self.plot_all([self.roll, self.pitch, self.yaw])

    def plot_all(self, traces):
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

            plt.ylim([0., 2])
            plt.ylabel('strength')
            ax3.get_yaxis().set_label_coords(-0.1, 0.5)
            plt.xlabel('response time in s')
            plt.legend(loc=1)
            plt.grid()
        meanfreq = 1./(traces[0].time_eq[1]-traces[0].time_eq[0])
        ax4 = plt.subplot(gs1[:, -1])
        t = Version+"| Betaflight: Version "+self.headdict['version']+' | Craftname: '+self.headdict['craftName']+\
            ' | meanFreq: '+str(int(meanfreq))+' | rcRate/Expo: '+self.headdict['rcRate']+'/'+ self.headdict['rcExpo']+'\nrcYawRate/Expo: '+self.headdict['rcYawRate']+'/' \
            +self.headdict['rcYawExpo']+' | deadBand: '+self.headdict['deadBand']+' | yawDeadBand: '+self.headdict['yawDeadBand'] \
            +' | Throttle min/tpa/max: ' + self.headdict['minThrottle']+'/'+self.headdict['tpa_breakpoint']+'/'+self.headdict['maxThrottle'] \
            + ' | dynThrPID: ' + self.headdict['dynThrottle']+ '| D-TermSP: ' + self.headdict['dTermSetPoint']+'| vbatComp: ' + self.headdict['vbatComp']

        plt.text(0.5, 0, t, ha='left', rotation=90, wrap=True, color='grey', alpha=0.5, fontsize=8)
        ax4.axis('off')
        plt.savefig(self.file[:-13] + self.name + '_' + str(self.headdict['logNum'])+'.png')
        #plt.cla()
        #plt.clf()
        return fig

    def __analyze(self):
        analyzed = []
        for t in self.traces:
            print t[0] + '...   ',
            analyzed.append(Trace(t))
            print 'Done!'
        return analyzed

    def readcsv(self, fpath):
        print 'Reading log '+str(self.headdict['logNum'][0])+'...   ',
        datdic = {}
        data = read_csv(fpath, header=0, skipinitialspace=1)
        datdic.update({'time_us': data['time (us)'].values * 1e-6})
        datdic.update({'throttle': data['rcCommand[3]'].values * 1e-6})

        for i in ['0', '1', '2']:
            datdic.update({'PID loop in' + i: data['axisP[' + i+']'].values})
            if 'gyroADC[0]' in data.keys():
                datdic.update({'gyroData' + i: data['gyroADC[' + i+']'].values})
            elif 'gyroData[0]' in data.keys():
                datdic.update({'gyroData' + i: data['gyroData[' + i+']'].values})
            else:
                print 'No gyro trace found!'

        print 'Done!'

        return datdic


    def find_traces(self, dat):
        time = self.data['time_us']
        throttle = dat['throttle']

        input0 = dat['PID loop in0']
        input1 = dat['PID loop in1']
        input2 = dat['PID loop in2']

        output0 = dat['gyroData0']
        output1 = dat['gyroData1']
        output2 = dat['gyroData2']

        throt = (throttle-0.001)*(float(self.headdict['maxThrottle'])-float(self.headdict['minThrottle']))*100.
        self.headdict.update({'tpa_percent':100.*(float(self.headdict['tpa_breakpoint'])-float(self.headdict['minThrottle']))/
                                            (float(self.headdict['maxThrottle'])-float(self.headdict['minThrottle']))})


        p0 = float((self.headdict['rollPID']).split(',')[0])*0.01
        p1 = float((self.headdict['pitchPID']).split(',')[0])*0.01
        p2 = float((self.headdict['yawPID']).split(',')[0])*0.01

        return [['roll', time, input0, output0, p0, throt],
                ['pitch', time, input1, output1, p1, throt],
                ['yaw', time, input2, output2, p2, throt]]



class BB_log:
    def __init__(self, filepath, name=''):
        self.fpath = filepath
        self.path, self.file = os.path.split(self.fpath)
        self.name = name

        #self.maketemp(self.path)
        self.loglist = self.decode(self.file)
        self.heads = self.beheader(self.loglist)
        self.figs = self._csv_iter(self.heads)

        self.deletejunk(self.loglist)

    def maketemp(self, path):
        os.mkdir(path+'/tmp')


    def deletejunk(self, loglist):
        for l in loglist:
            os.remove(l)
            os.remove(l[:-3]+'01.csv')
            try:
                os.remove(l[:-3]+'01.event')
            except:
                print 'No .event file of '+l+' found.'
        return

    def _csv_iter(self, heads):
        figs = []
        for h in heads:
            figs.append(CSV_log(h['tempFile'][:-3]+'01.csv', self.name, h).fig)
        return figs

    def beheader(self, loglist):
        heads = []
        for i, bblog in enumerate(loglist):
            log = open(self.path+'/'+bblog, 'r')
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

    def logfinder(self, fpath):
        path, file = os.path.split(fpath)
        name = file[:-3]
        flist = os.listdir(path)
        csvlist =   []
        csv_sizes = []
        eventlist =  []

        for i in range(1, 99, 1):
            csvname = name+'_temp'+str(i)+'.01.csv'
            eventname = name+'_temp'+str(i)+'.01.event'
            if csvname not in flist: break
            csvlist.append(csvname)
            eventlist.append(eventname)
        for csv in csvlist:
            csv_sizes.append(os.path.getsize(path+'/'+csv))
        return csvlist, csv_sizes, eventlist

    def decode(self, fpath):
        log = open(fpath, 'rb')
        log2 = open(fpath, 'r')
        firstline = log2.readlines()[0]
        content = log.read()

        split = content.split(str(firstline))
        temps = []
        for i in range(len(split)):
            newfile=open(fpath[:-4]+'_temp'+str(i)+fpath[-4:], 'wb')
            newfile.write(firstline+split[i])
            temps.append(fpath[:-4]+'_temp'+str(i)+fpath[-4:])
            newfile.close()

        loglist = []
        for t in temps:
            size = os.path.getsize(self.path+'/'+t)
            if size>500000:
                try:
                    msg = os.system('blackbox_decode.exe' + ' '+t)
                    loglist.append(t)
                except:
                    print 'Error in Blackbox_decode'
            else:
                os.remove(t)
        return loglist



def main():
    ### use here via:
    #test = BB_log('path', 'test')
    #plt.show()
    print Version +'\n\n'
    print 'Hello Pilot!'
    print 'This program uses Blackbox_decode:\n' \
          'https://github.com/cleanflight/blackbox-tools/releases\n' \
          'to generate .csv files from your log.\n' \
          'Please put logfiles, Blackbox_decode.exe and this program into a single folder.\n'

    while True:
        file = raw_input("Place your log here: \n-->")
        name = raw_input('\n Name for this plot: (optional)\n')
        test = BB_log(str(file)[1:-1], str(name))
        plt.show()

if __name__ == "__main__":
    main()
