import pycbc.noise
import pycbc.psd
import pylab
import inspect
import lalsimulation

# The color of the noise matches a PSD which you provide
flow = 30.0
delta_f = 1.0 / 16
flen = int(2048 / delta_f) + 1
psd = pycbc.psd.aLIGOZeroDetHighPower(flen, delta_f, flow)
print(len(psd))
# Generate 32 seconds of noise at 4096 Hz
delta_t = 1.0 / 4096
tsamples = int(32 / delta_t)
ts = pycbc.noise.noise_from_psd(tsamples, delta_t, psd, seed=127)
print(len(ts))
print(inspect.getsource(lalsimulation))
pylab.plot(ts.sample_times, ts)
pylab.ylabel('Strain')
pylab.xlabel('Time (s)')
pylab.show()
