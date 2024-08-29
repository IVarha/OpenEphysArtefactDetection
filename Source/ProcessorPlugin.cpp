/*
------------------------------------------------------------------

This file is part of the Open Ephys GUI
Copyright (C) 2022 Open Ephys

------------------------------------------------------------------

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "ProcessorPlugin.h"

#include "ProcessorPluginEditor.h"
#include <fftw3.h>
ProcessorPlugin::ProcessorPlugin()
    : GenericProcessor("STN Artefact Detection")
{
    addIntParameter(Parameter::GLOBAL_SCOPE,
                    "window_ms", "The size of the rolling average window in milliseconds",
                    1000, 10, 5000);

    addSelectedChannelsParameter(Parameter::STREAM_SCOPE,
                                 "Channels", "The input channels to analyze");
}

ProcessorPlugin::~ProcessorPlugin()
{
}

AudioProcessorEditor *ProcessorPlugin::createEditor()
{
    editor = std::make_unique<ProcessorPluginEditor>(this);
    return editor.get();
}

void ProcessorPlugin::updateSettings()
{
    int numInputsChange = getNumInputs() - currMean.size();

    if (numInputsChange > 0)
    {
        currMean.insertMultiple(-1, 0.0, numInputsChange);
        currVar.insertMultiple(-1, 0.0, numInputsChange);
        startingRunningMean.insertMultiple(-1, true, numInputsChange);
    }
    else if (numInputsChange < 0)
    {
        currMean.removeLast(-numInputsChange);
        currVar.removeLast(-numInputsChange);
        startingRunningMean.removeLast(-numInputsChange);
    }

    window_ms = getParameter("window_ms")->getValue();
}

void ProcessorPlugin::parameterValueChanged(Parameter *param)
{
    if (param->getName().equalsIgnoreCase("window_ms"))
    {
        window_ms = param->getValue();
    }
}
// Resample function
std::vector<double> resample(const std::vector<double>& signal, double targetFs, double currentFs) {
    std::vector<double> resampledSignal;
    // Calculate the resampling factor
    double resampleFactor = targetFs / currentFs;
    // Calculate the size of the resampled signal
    size_t resampledSize = static_cast<size_t>(signal.size() * resampleFactor);
    resampledSignal.resize(resampledSize);
    // Perform linear interpolation for resampling
    for (size_t i = 0; i < resampledSize; ++i) {
        double index = i / resampleFactor;
        size_t lowerIndex = static_cast<size_t>(index);
        size_t upperIndex = std::min(lowerIndex + 1, signal.size() - 1);
        double weight = index - lowerIndex;
        resampledSignal[i] = signal[lowerIndex] * (1.0 - weight) + signal[upperIndex] * weight;
    }
    return resampledSignal;
}

// Function to compute Welch's method for power spectral density estimation
std::vector<double> pwelch(const std::vector<double>& x, int window_size, int overlap, int fs) {
    int N = x.size(); // Length of input signal
    int nfft = window_size; // FFT length
    int nOverlap = overlap; // Overlap length
    int nWindows = (N - nfft) / (nfft - nOverlap) + 1; // Number of windows
    double M_PI = 3.141592653589793;
    // Initialize FFTW plans
    fftw_complex* fft_result = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * (nfft / 2 + 1));
    double* psd = new double[nfft / 2 + 1]; // Power Spectral Density
    fftw_plan forward_plan = fftw_plan_dft_r2c_1d(nfft, NULL, fft_result, FFTW_ESTIMATE);

    std::vector<double> result(nfft / 2 + 1, 0.0); // Initialize result

    // Iterate through windows
    for (int i = 0; i < nWindows; ++i) {
        // Extract current window
        std::vector<double> window(x.begin() + i * (nfft - nOverlap), x.begin() + i * (nfft - nOverlap) + nfft);
        
        // Apply windowing function (e.g., Hamming window)
        for (int j = 0; j < nfft; ++j) {
            window[j] *= 0.54 - 0.46 * cos(2 * M_PI * j / (nfft - 1)); // Hamming window
        }
        
        // Perform FFT
        fftw_plan forward_plan = fftw_plan_dft_r2c_1d(nfft, window.data(), fft_result, FFTW_ESTIMATE);
        fftw_execute(forward_plan);

        // Compute Power Spectral Density (PSD)
        for (int j = 0; j < nfft / 2 + 1; ++j) {
            psd[j] = (fft_result[j][0] * fft_result[j][0] + fft_result[j][1] * fft_result[j][1]) / nfft;
        }

        // Accumulate PSD
        for (int j = 0; j < nfft / 2 + 1; ++j) {
            result[j] += psd[j];
        }
    }

    // Normalize result by number of windows
    for (int j = 0; j < nfft / 2 + 1; ++j) {
        result[j] /= nWindows;
    }

    // Cleanup
    fftw_destroy_plan(forward_plan);
    fftw_free(fft_result);
    delete[] psd;

    return result;
}

double maxDiffPSD(const std::vector<double>& signal, double fs) 
{
    // Resample signal to 1000 Hz
    std::vector<double> resampledSignal = signal;
    if (fs != 24000) {
        resampledSignal = resample(signal, 24000, fs);
    }
        std::vector<double> meanClnPSD = {
        4.77562431760460e-05, 0.000166178838654735, 0.000227806114901376, 0.000146040306929677,
        0.000142911335033798, 0.000192389466398494, 0.000241298241799244, 0.000313328240515836,
        0.000551530514429587, 0.000676116046498073, 0.000589870009775301, 0.000666057276492933,
        0.000976870506074863, 0.00153008829774771, 0.00111734213373207, 0.00118978582786262,
        0.00138826592498130, 0.00183274813159608, 0.00166025979142410, 0.00172365391321772,
        0.00187407420905555, 0.00232046231406939, 0.00231718653722749, 0.00223308922838443,
        0.00229458106215795, 0.00249993684340288, 0.00265075493463280, 0.00270928586528559,
        0.00290230525509276, 0.00311006754409378, 0.00343268066344955, 0.00329037019721431,
        0.00334454943324877, 0.00344422750530643, 0.00341031274248160, 0.00341195297684773,
        0.00349486459313133, 0.00359357475245854, 0.00396922454483106, 0.00406929313602893,
        0.00411443190955744, 0.00417929900513955, 0.00399602431438698, 0.00411053910110060,
        0.00413623833438308, 0.00422796555655805, 0.00442006297486236, 0.00450079497340097,
        0.00449572150989367, 0.00459150757377476, 0.00480359422910404, 0.00477445815618844,
        0.00467302146324857, 0.00475741028179341, 0.00466343919152143, 0.00479140294729342,
        0.00490962350523759, 0.00495305358928402, 0.00492313075478619, 0.00494184122580270,
        0.00507271776515933, 0.00497804382483539, 0.00491620260712145, 0.00497576630307890,
        0.00504328104327651, 0.00500239299106261, 0.00505212342917972, 0.00500623040375661,
        0.00511808393864830, 0.00517149453283253, 0.00515532861107924, 0.00525727966706852,
        0.00532564371679703, 0.00524950588753595, 0.00513195437617655, 0.00515258273909534,
        0.00525297852364360, 0.00518068254349220, 0.00516886501298688, 0.00525421453752356,
        0.00521933405230834, 0.00522291129028569, 0.00515097660980339, 0.00513469256030222,
        0.00509685952940660, 0.00513033954767364, 0.00510540184120212, 0.00511239023673139,
        0.00510007133642861, 0.00515262337645407, 0.00509412434890254, 0.00499609041421214,
        0.00498431690873961, 0.00499109119680796, 0.00500241890073058, 0.00493661041288476,
        0.00483144683329681, 0.00475685578729588, 0.00477267259803039, 0.00468781002129009,
        0.00453504410066656, 0.00441554908554769, 0.00428693478964860, 0.00414824560076558,
        0.00402639994614453, 0.00396977628036308, 0.00381124811520073, 0.00358005585623452,
        0.00341173982172213, 0.00315776945179418, 0.00282596788939648, 0.00243328710170897,
        0.00194906339414642, 0.00143720209092989, 0.000984442619152012, 0.000708251619769507,
        0.000652203116917473, 0.000587308093124203, 0.000395889161122253, 0.000231961065082064,
        0.000153917206028356, 8.98882592299845e-05, 6.36917068230152e-05, 4.09800050910511e-05,
        2.64821890402850e-05, 1.78785650435613e-05, 1.31913155309165e-05, 1.02133156072875e-05,
        8.14636483140256e-06, 6.51860115623732e-06, 4.96845723093568e-06, 3.74540136466932e-06,
        3.40321984841199e-06, 3.14262291881277e-06, 2.54613862758588e-06, 1.98710511923399e-06,
        1.52063119274375e-06, 1.14453905787863e-06, 9.20127400809710e-07, 7.63438218168368e-07,
        6.62864135227592e-07, 5.68947395783192e-07, 4.74594419611450e-07, 3.78724515605050e-07,
        3.17605203658248e-07, 2.66544661094624e-07, 2.20445639478585e-07, 1.78235510202898e-07,
        1.45773392674388e-07, 1.25366031701202e-07, 1.01947623878542e-07, 8.42425921307846e-08,
        6.85905982134346e-08, 5.48727514314379e-08, 4.30786559666156e-08, 3.31608544672109e-08,
        2.49392722988317e-08, 1.82723356379005e-08, 1.29498268163043e-08, 8.79191746931985e-09,
        5.90097514537235e-09, 3.86611582416051e-09, 2.45331738489978e-09, 1.52496181053104e-09,
        9.40875107593720e-10, 5.68488698213966e-10, 3.37569641250794e-10, 1.97602426114367e-10,
        1.14005639079220e-10, 6.34692780030706e-11, 3.37180814533978e-11, 1.68002521503315e-11,
        7.74202505920239e-12, 3.13569668773717e-12, 1.07059776509576e-12, 3.00875734427033e-13,
        7.29548402921200e-14, 1.44583199225761e-14, 2.15520342090083e-15, 2.49998367830978e-16,
        2.24673102055617e-17, 1.55500219980910e-18, 8.07438395306694e-20, 3.11798944261579e-21,
        8.34826743235370e-23, 1.39920891336682e-24, 1.35407110108641e-26, 6.86073013814836e-29,
        1.50723854030200e-31, 1.31099619334005e-34, 3.80357972307392e-38
    };

        int NFFT = (meanClnPSD.size()-1)*2;

        std::vector<double> psd = pwelch(resampledSignal, NFFT, NFFT/2, NFFT);
        // compute sum psd
        double sum = 0;
        for (int i = 0; i < psd.size(); i++) {
            sum += psd[i];
        }
        // the max. absolute difference
        double maxDiff = 0;
        for (int i = 0; i < psd.size(); i++) {
            double diff = abs((psd[i]/sum) - meanClnPSD[i]);
            if (diff > maxDiff) {
                maxDiff = diff;
            }
        }
        return maxDiff;

}




void ProcessorPlugin::process(AudioBuffer<float> &buffer)
{
    for (auto stream : getDataStreams())
    {
        if ((*stream)["enable_stream"])
        {
            const uint16 streamId = stream->getStreamId();
            const uint32 nSamples = getNumSamplesInBlock(streamId);
            
            float sampleRate = getSampleRate(streamId);
            
            double samplesPerMs = getSampleRate(streamId) / 1000.0;


            int ch = 0;


            for (auto chan : *((*stream)["Channels"].getArray()))
            {
                
                double mean, var;
                int samp = 0;
                // print channel number
                //std::cout << "Channel1: " << int(chan) << std::endl;
                std::vector<double> bufferRecording = std::vector<double>();

                for (; samp < nSamples; ++samp)
                {
                    bufferRecording.push_back(buffer.getSample(chan, samp));

                }
                double md = maxDiffPSD(bufferRecording, sampleRate);
                // print to console
                //Estd::cout << "Max diff PSD: " << md << std::endl;
                if (md < 0.1) {
                    // send to output
                    for (samp = 0; samp < nSamples; ++samp)
                    {
                        buffer.setSample(chan, samp, bufferRecording[samp]);
                    }
                }
                else {
                    // send to output
                    for (samp = 0; samp < nSamples; ++samp)
                    {
                        buffer.setSample(chan, samp, std::numeric_limits<double>::quiet_NaN());
                    }
                }
                ch++;
            }
        }
    }
}
