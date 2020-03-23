/*
July 15, 2015
Iowa Hills Software LLC
http://www.iowahills.com

If you find a problem with this code, please leave us a note on:
http://www.iowahills.com/feedbackcomments.html

Source: ~Projects\Common\BasicFIRFilterCode.cpp

This generic FIR filter code is described in most textbooks.
e.g. Discrete Time Signal Processing, Oppenheim and Shafer

A nice paper on this topic is:
http://dea.brunel.ac.uk/cmsp/Home_Saeed_Vaseghi/Chapter05-DigitalFilters.pdf

This code first generates either a low pass, high pass, band pass, or notch
impulse response for a rectangular window. It then applies a window to this
impulse response.

There are several windows available, including the Kaiser, Sinc, Hanning,
Blackman, and Hamming. Of these, the Kaiser and Sinc are probably the most useful
for FIR filters because their sidelobe levels can be controlled with the Beta parameter.

This is a typical function call:
BasicFIR(FirCoeff, NumTaps, PassType, OmegaC, BW, wtKAISER, Beta);
BasicFIR(FirCoeff, 33, LPF, 0.2, 0.0, wtKAISER, 3.2);
33 tap, low pass, corner frequency at 0.2, BW=0 (ignored in the low pass code),
Kaiser window, Kaiser Beta = 3.2

These variables should be defined similar to this:
double FirCoeff[MAXNUMTAPS];
int NumTaps;                        NumTaps can be even or odd, but must be less than the FirCoeff array size.
TPassTypeName PassType;             PassType is an enum defined in the header file. LPF, HPF, BPF, or NOTCH
double OmegaC  0.0 < OmegaC < 1.0   The filters corner freq, or center freq if BPF or NOTCH
double BW      0.0 < BW < 1.0       The filters band width if BPF or NOTCH
TWindowType WindowType;             WindowType is an enum defined in the header to be one of these.
                                    wtNONE, wtKAISER, wtSINC, wtHANNING, .... and others.
double Beta;  0 <= Beta <= 10.0     Beta is used with the Kaiser, Sinc, and Sine windows only.
                                    It controls the transition BW and sidelobe level of the filters.


If you want to use it, Kaiser originally defined Beta as follows.
He derived its value based on the desired sidelobe level, dBAtten.
double dBAtten, Beta, Beta1=0.0, Beta2=0.0;
if(dBAtten < 21.0)dBAtten = 21.0;
if(dBAtten > 50.0)Beta1 = 0.1102 * (dBAtten - 8.7);
if(dBAtten >= 21.0 && dBAtten <= 50.0) Beta2 = 0.5842 * pow(dBAtten - 21.0, 0.4) + 0.07886 * (dBAtten - 21.0);
Beta = Beta1 + Beta2;

2015年7月15日
爱荷华山软件有限公司
http://www.iowahills.com

如果您发现此代码有问题，请给我们提供以下注意事项：
http://www.iowahills.com/feedbackcomments.html

来源：〜Projects \ Common \ BasicFIRFilterCode.cpp

大多数教科书都介绍了这种通用的FIR滤波器代码。
例如离散时间信号处理，Oppenheim和Shafer

关于此主题的一篇不错的论文是：
http://dea.brunel.ac.uk/cmsp/Home_Saeed_Vaseghi/Chapter05-DigitalFilters.pdf

此代码首先生成低通，高通，带通或陷波
矩形窗口的脉冲响应。然后将一个窗口应用于此
冲动反应。

有几个可用的窗口，包括Kaiser，Sinc，Hanning，
布莱克曼和汉明。其中，凯撒（Kaiser）和辛克（Sinc）可能是最有用的
用于FIR滤波器，因为可以通过Beta参数控制其旁瓣电平。

这是一个典型的函数调用：
BasicFIR（FirCoeff，NumTaps，PassType，OmegaC，BW，wtKAISER，Beta）;
BasicFIR（FirCoeff，33，LPF，0.2，0.0，wtKAISER，3.2）;
33抽头，低通，转折频率为0.2，BW = 0（在低通代码中忽略），
Kaiser窗口，Kaiser Beta = 3.2

这些变量的定义应与此类似：
双FirCoeff [MAXNUMTAPS];
int NumTaps； NumTaps可以是偶数或奇数，但必须小于FirCoeff数组的大小。
TPassTypeName PassType; PassType是在头文件中定义的枚举。 LPF，HPF，BPF或NOTCH
两倍OmegaC 0.0 <OmegaC <1.0过滤器拐角频率，或者如果BPF或NOTCH，则为中心频率
double BW 0.0 <BW <1.0 BPF或NOTCH时的滤波器带宽
TWindowType WindowType; WindowType是在标头中定义为其中之一的枚举。
                                    wtNONE，wtKAISER，wtSINC，wtHANNING等。
双Beta 0 <= Beta <= 10.0 Beta仅用于Kaiser，Sinc和Sine窗口。
                                    它控制滤波器的过渡带宽和旁瓣电平。


如果要使用它，Kaiser最初将Beta定义如下。
他根据所需的旁瓣电平dBAtten得出其值。
double dBAtten，Beta，Beta1 = 0.0，Beta2 = 0.0;
if（dBAtten <21.0）dBAtten = 21.0;
if（dBAtten> 50.0）Beta1 = 0.1102 *（dBAtten-8.7）;
if（dBAtten> = 21.0 && dBAtten <= 50.0）Beta2 = 0.5842 * pow（dBAtten-21.0，0.4）+ 0.07886 *（dBAtten-21.0）;
Beta = Beta1 + Beta2;
*/

//---------------------------------------------------------------------------


#pragma hdrstop     // for Embarcadero's C++ Builder用于Embarcadero的C ++ Builder

#include "WindowedFIRFilterWebCode.h"
#include <math.h>
#include <new.h>     // defines new(std::nothrow)定义new（std :: nothrow）
#include <vcl.h>     // for Embarcadero's ShowMessage function.用于Embarcadero的ShowMessage函数。

#pragma package(smart_init)  // for C++ Builder

#define M_2PI  6.28318530717958647692  // M_PI should be in the math.h file M_PI应该在math.h文件中
//---------------------------------------------------------------------------


// This first calculates the impulse response for a rectangular window.首先计算矩形窗口的脉冲响应。
// It then applies the windowing function of choice to the impulse response.然后将选择的开窗函数应用于脉冲响应。
void BasicFIR(double *FirCoeff, int NumTaps, TPassTypeName PassType, double OmegaC, double BW, TWindowType WindowType, double WinBeta)
{
 int j;
 double Arg, OmegaLow, OmegaHigh;

 switch(PassType)
  {
   case LPF:
	for(j=0; j<NumTaps; j++)
     {
      Arg = (double)j - (double)(NumTaps-1) / 2.0;
      FirCoeff[j] = OmegaC * Sinc(OmegaC * Arg * M_PI);
     }
    break;

   case HPF:
    if(NumTaps % 2 == 1) // Odd tap counts
     {
      for(j=0; j<NumTaps; j++)
       {
        Arg = (double)j - (double)(NumTaps-1) / 2.0;
        FirCoeff[j] = Sinc(Arg * M_PI) - OmegaC * Sinc(OmegaC * Arg * M_PI);
       }
     }

    else  // Even tap counts
      {
       for(j=0; j<NumTaps; j++)
        {
         Arg = (double)j - (double)(NumTaps-1) / 2.0;
         if(Arg == 0.0)FirCoeff[j] = 0.0;
         else FirCoeff[j] = cos(OmegaC * Arg * M_PI) / M_PI / Arg  + cos(Arg * M_PI);
        }
      }
   break;

   case BPF:
    OmegaLow  = OmegaC - BW/2.0;
    OmegaHigh = OmegaC + BW/2.0;
	for(j=0; j<NumTaps; j++)
     {
      Arg = (double)j - (double)(NumTaps-1) / 2.0;
      if(Arg == 0.0)FirCoeff[j] = 0.0;
      else FirCoeff[j] =  ( cos(OmegaLow * Arg * M_PI) - cos(OmegaHigh * Arg * M_PI) ) / M_PI / Arg ;
     }
   break;

   case NOTCH:  // If NumTaps is even for Notch filters, the response at Pi is attenuated.如果NumTaps甚至适用于陷波滤波器，则Pi处的响应会衰减。
    OmegaLow  = OmegaC - BW/2.0;
    OmegaHigh = OmegaC + BW/2.0;
	for(j=0; j<NumTaps; j++)
     {
      Arg = (double)j - (double)(NumTaps-1) / 2.0;
      FirCoeff[j] =  Sinc(Arg * M_PI) - OmegaHigh * Sinc(OmegaHigh * Arg * M_PI) - OmegaLow * Sinc(OmegaLow * Arg * M_PI);
     }
   break;
  }

 // WindowData can be used to window data before an FFT. When used for FIR filters we set 
 // Alpha = 0.0 to prevent a flat top on the window and 
 // set UnityGain = false to prevent the window gain from getting set to unity.
// WindowData可用于对FFT之前的数据进行窗口化。 当用于FIR滤波器时，我们将Alpha设置为0.0以防止窗口顶部平坦，而将UnityGain设置为false可以防止窗口增益设置为单位。

 WindowData(FirCoeff, NumTaps, WindowType, 0.0, WinBeta, false);

}

//---------------------------------------------------------------------------


// This gets used with the Kaiser window.这将与Kaiser窗口一起使用。
double Bessel(double x)
{
 double Sum=0.0, XtoIpower;
 int i, j, Factorial;
 for(i=1; i<10; i++)
  {
   XtoIpower = pow(x/2.0, (double)i);
   Factorial = 1;
   for(j=1; j<=i; j++)Factorial *= j;
   Sum += pow(XtoIpower / (double)Factorial, 2.0);
  }
 return(1.0 + Sum);
}

//-----------------------------------------------------------------------------

// This gets used with the Sinc window and various places in the BasicFIR function.这可用于Sinc窗口以及BasicFIR函数中的各个位置。
double Sinc(double x)
{
 if(x > -1.0E-5 && x < 1.0E-5)return(1.0);
 return(sin(x)/x);
}

//---------------------------------------------------------------------------

// These are the various windows definitions. These windows can be used for either
// FIR filter design or with an FFT for spectral analysis.
// Sourced verbatim from: ~MyDocs\Code\Common\FFTFunctions.cpp
// For definitions, see this article:  http://en.wikipedia.org/wiki/Window_function

// This function has 6 inputs
// Data is the array, of length N, containing the data to to be windowed. 
// This data is either a FIR filter sinc pulse, or the data to be analyzed by an fft.
 
// WindowType is an enum defined in the header file, which is at the bottom of this file.
// e.g. wtKAISER, wtSINC, wtHANNING, wtHAMMING, wtBLACKMAN, ...

// Alpha sets the width of the flat top.
// Windows such as the Tukey and Trapezoid are defined to have a variably wide flat top.
// As can be seen by its definition, the Tukey is just a Hanning window with a flat top.
// Alpha can be used to give any of these windows a partial flat top, except the Flattop and Kaiser.
// Alpha = 0 gives the original window. (i.e. no flat top)
// To generate a Tukey window, use a Hanning with 0 < Alpha < 1
// To generate a Bartlett window (triangular), use a Trapezoid window with Alpha = 0.
// Alpha = 1 generates a rectangular window in all cases. (except the Flattop and Kaiser)


// Beta is used with the Kaiser, Sinc, and Sine windows only.
// These three windows are primarily used for FIR filter design, not spectral analysis.
// In FIR filter design, Beta controls the filter's transition bandwidth and the sidelobe levels.
// The code ignores Beta except in the Kaiser, Sinc, and Sine window cases.

// UnityGain controls whether the gain of these windows is set to unity.
// Only the Flattop window has unity gain by design. The Hanning window, for example, has a gain
// of 1/2.  UnityGain = true will set the gain of all these windows to 1.
// Then, when the window is applied to a signal, the signal's energy content is preserved.
// Don't use this with FIR filter design however. Since most of the enegy in an FIR sinc pulse
// is in the middle of the window, the window needs a peak amplitude of one, not unity gain.
// Setting UnityGain = true will simply cause the resulting FIR filter to have excess gain.

// If using these windows for FIR filters, start with the Kaiser, Sinc, or Sine windows and
// adjust Beta for the desired transition BW and sidelobe levels (set Alpha = 0).
// While the FlatTop is an excellent window for spectral analysis, don't use it for FIR filter design.
// It has a peak amplitude of ~ 4.7 which causes the resulting FIR filter to have about this much gain.
// It works poorly for FIR filters even if you adjust its peak amplitude.
// The Trapezoid also works poorly for FIR filter design.

// If using these windows with an fft for spectral analysis, start with the Hanning, Gauss, or Flattop.
// When choosing a window for spectral analysis, you must trade off between resolution and amplitude accuracy.
// The Hanning has the best resolution while the Flatop has the best amplitude accuracy.
// The Gauss is midway between these two for both accuracy and resolution.
// These three were the only windows available in the HP 89410A Vector Signal Analyzer. Which is to say,
// unless you have specific windowing requirements, use one of these 3 for general purpose signal analysis.
// Set UnityGain = true when using any of these windows for spectral analysis to preserve the signal's enegy level.


//这些是各种Windows定义。这些窗口可用于
// FIR滤波器设计或带FFT的频谱分析。
//逐字记录来自：〜MyDocs \ Code \ Common \ FFTFunctions.cpp
//有关定义，请参见本文：http://en.wikipedia.org/wiki/Window_function

//此功能有6个输入
// Data是长度为N的数组，其中包含要窗口化的数据。
//此数据可以是FIR滤波器的正弦脉冲，也可以是fft要分析的数据。
 
// WindowType是在头文件中定义的枚举，该头文件位于此文件的底部。
//例如wtKAISER，wtSINC，wtHANNING，wtHAMMING，wtBLACKMAN，...

// Alpha设置平顶的宽度。
// Windows（例如Tukey和Trapezoid）定义为具有可变宽度的平顶。
//从其定义可以看出，Tukey只是具有平顶的Hanning窗口。
//除了Flattop和Kaiser之外，Alpha可用于为这些窗口中的任何一个提供部分平坦的顶部。
// Alpha = 0给出原始窗口。 （即无平顶）
//要生成Tukey窗口，请使用0 <Alpha <1的Hanning
//要生成Bartlett窗口（三角形），请使用Alpha = 0的梯形窗口。
// Alpha = 1会在所有情况下生成一个矩形窗口。 （Flattop和Kaiser除外）


// Beta仅与Kaiser，Sinc和Sine窗口一起使用。
//这三个窗口主要用于FIR滤波器设计，而不是频谱分析。
//在FIR滤波器设计中，Beta控制滤波器的过渡带宽和旁瓣电平。
//除了Kaiser，Sinc和Sine窗口情况外，代码均忽略Beta。

// UnityGain控制这些窗口的增益是否设置为单位。
//仅Flattop窗口在设计上具有单位增益。以汉宁窗为例
//的1/2。 UnityGain = true会将所有这些窗口的增益设置为1。
//然后，将窗口应用于信号时，将保留信号的能量含量。
//但是，请勿将此与FIR滤波器设计一起使用。由于大部分能量在FIR正弦脉冲中
//在窗口的中间，窗口需要的峰值幅度为1，而不是单位增益。
//设置UnityGain = true只会导致生成的FIR滤波器具有多余的增益。

//如果将这些窗口用于FIR滤波器，请从Kaiser，Sinc或Sine窗口开始，然后
//根据所需的过渡带宽和旁瓣电平（设置Alpha = 0）调整Beta。
//虽然FlatTop是进行光谱分析的绝佳窗口，但不要将其用于FIR滤波器设计。
//它的峰值幅度约为4.7，这使得所得的FIR滤波器具有大约这样的增益。
//即使您调整FIR滤波器的峰值幅度，它的效果也很差。
//梯形在FIR滤波器设计中也不能很好地工作。

//如果将这些窗口与fft一起用于频谱分析，请从Hanning，Gauss或Flattop开始。
//选择光谱分析窗口时，必须在分辨率和幅度精度之间进行权衡。
// Hanning具有最佳分辨率，而Flatop具有最佳幅度精度。
//就准确性和分辨率而言，高斯介于两者之间。
//这三个是HP 89410A矢量信号分析仪中唯一可用的窗口。就是说
//除非您有特定的窗口要求，否则请使用这3种之一进行通用信号分析。
//当使用这些窗口中的任何一个进行频谱分析以保留信号的能量水平时，将UnityGain = true设置。

void WindowData(double *Data, int N, TWindowType WindowType, double Alpha, double Beta, bool UnityGain)
{
 if(WindowType == wtNONE) return;

 int j, M, TopWidth;
 double dM, *WinCoeff;

 if(WindowType == wtKAISER ||  WindowType == wtFLATTOP )Alpha = 0.0;

 if(Alpha < 0.0)Alpha = 0.0;
 if(Alpha > 1.0)Alpha = 1.0;

 if(Beta < 0.0)Beta = 0.0;
 if(Beta > 10.0)Beta = 10.0;

 WinCoeff  = new(std::nothrow) double[N+2];
 if(WinCoeff == NULL)
  {
   ShowMessage("Failed to allocate memory in FFTFunctions::WindowFFTData() ");
   return;
  }

 TopWidth = (int)( Alpha * (double)N );
 if(TopWidth%2 != 0)TopWidth++;
 if(TopWidth > N)TopWidth = N;
 M = N - TopWidth;
 dM = M + 1;


 // Calculate the window for N/2 points, then fold the window over (at the bottom).计算N / 2点的窗口，然后将其折叠（在底部）。
 // TopWidth points will be set to 1.TopWidth点将设置为1。
 if(WindowType == wtKAISER)
  {
   double Arg;
   for(j=0; j<M; j++)
	{
	 Arg = Beta * sqrt(1.0 - pow( ((double)(2*j+2) - dM) / dM, 2.0) );
	 WinCoeff[j] = Bessel(Arg) / Bessel(Beta);
	}
  }

 else if(WindowType == wtSINC)  // Lanczos
  {
   for(j=0; j<M; j++)WinCoeff[j] = Sinc((double)(2*j+1-M)/dM * M_PI );
   for(j=0; j<M; j++)WinCoeff[j] = pow(WinCoeff[j], Beta);
  }

 else if(WindowType == wtSINE)  // Hanning if Beta = 2
  {
   for(j=0; j<M/2; j++)WinCoeff[j] = sin((double)(j+1) * M_PI / dM);
   for(j=0; j<M/2; j++)WinCoeff[j] = pow(WinCoeff[j], Beta);
  }

 else if(WindowType == wtHANNING)
  {
   for(j=0; j<M/2; j++)WinCoeff[j] = 0.5 - 0.5 * cos((double)(j+1) * M_2PI / dM);
  }

 else if(WindowType == wtHAMMING)
  {
   for(j=0; j<M/2; j++)
   WinCoeff[j] = 0.54 - 0.46 * cos((double)(j+1) * M_2PI / dM);
  }

 else if(WindowType == wtBLACKMAN)
  {
   for(j=0; j<M/2; j++)
	{
	 WinCoeff[j] = 0.42
	 - 0.50 * cos((double)(j+1) * M_2PI / dM)
	 + 0.08 * cos((double)(j+1) * M_2PI * 2.0 / dM);
	}
  }


 // See: http://www.bth.se/fou/forskinfo.nsf/0/130c0940c5e7ffcdc1256f7f0065ac60/$file/ICOTA_2004_ttr_icl_mdh.pdf
 else if(WindowType == wtFLATTOP)
  {
   for(j=0; j<=M/2; j++)
	{
	 WinCoeff[j] = 1.0
	 - 1.93293488969227 * cos((double)(j+1) * M_2PI / dM)
	 + 1.28349769674027 * cos((double)(j+1) * M_2PI * 2.0 / dM)
	 - 0.38130801681619 * cos((double)(j+1) * M_2PI * 3.0 / dM)
	 + 0.02929730258511 * cos((double)(j+1) * M_2PI * 4.0 / dM);
	}
  }


 else if(WindowType == wtBLACKMAN_HARRIS)
  {
   for(j=0; j<M/2; j++)
	{
	 WinCoeff[j] = 0.35875
	 - 0.48829 * cos((double)(j+1) * M_2PI / dM)
	 + 0.14128 * cos((double)(j+1) * M_2PI * 2.0 / dM)
	 - 0.01168 * cos((double)(j+1) * M_2PI * 3.0 / dM);
	}
  }

 else if(WindowType == wtBLACKMAN_NUTTALL)
  {
   for(j=0; j<M/2; j++)
	{
	 WinCoeff[j] = 0.3535819
	 - 0.4891775 * cos((double)(j+1) * M_2PI / dM)
	 + 0.1365995 * cos((double)(j+1) * M_2PI * 2.0 / dM)
	 - 0.0106411 * cos((double)(j+1) * M_2PI * 3.0 / dM);
	}
  }

 else if(WindowType == wtNUTTALL)
  {
   for(j=0; j<M/2; j++)
	{
	 WinCoeff[j] = 0.355768
	 - 0.487396 * cos((double)(j+1) * M_2PI / dM)
	 + 0.144232 * cos((double)(j+1) * M_2PI * 2.0 / dM)
	 - 0.012604 * cos((double)(j+1) * M_2PI * 3.0 / dM);
	}
  }

 else if(WindowType == wtKAISER_BESSEL)
  {
   for(j=0; j<=M/2; j++)
	{
	 WinCoeff[j] = 0.402
	 - 0.498 * cos(M_2PI * (double)(j+1) / dM)
	 + 0.098 * cos(2.0 * M_2PI * (double)(j+1) / dM)
	 + 0.001 * cos(3.0 * M_2PI * (double)(j+1) / dM);
	}
  }

 else if(WindowType == wtTRAPEZOID) // Rectangle for Alpha = 1  Triangle for Alpha = 0Alpha的矩形= 1 Alpha的矩形= 0
  {
   int K = M/2;
   if(M%2)K++;
   for(j=0; j<K; j++)WinCoeff[j] = (double)(j+1) / (double)K;
  }


 // This definition is from http://en.wikipedia.org/wiki/Window_function (Gauss Generalized normal window)
 // We set their p = 2, and use Alpha in the numerator, instead of Sigma in the denominator, as most others do.
 // Alpha = 2.718 puts the Gauss window response midway between the Hanning and the Flattop (basically what we want).
 // It also gives the same BW as the Gauss window used in the HP 89410A Vector Signal Analyzer.
 // Alpha = 1.8 puts it quite close to the Hanning.


//此定义来自http://en.wikipedia.org/wiki/Window_function（Gauss广义普通窗口）
  //我们将其p设置为2，并像大多数其他函数一样，在分子中使用Alpha而不是在分母中使用Sigma。
  // Alpha = 2.718将高斯窗口响应置于Hanning和Flattop之间（基本上是我们想要的）。
  //它的带宽与HP 89410A矢量信号分析仪中使用的高斯窗口相同。
  // Alpha = 1.8使其与汉宁相当接近。
 else if(WindowType == wtGAUSS)
  {
   for(j=0; j<M/2; j++)
    {
     WinCoeff[j] = ((double)(j+1) - dM/2.0) / (dM/2.0) * 2.7183;
     WinCoeff[j] *= WinCoeff[j];
     WinCoeff[j] = exp(-WinCoeff[j]);
    }
  }

 else // Error.
  {
   ShowMessage("Incorrect window type in WindowFFTData");
   delete[] WinCoeff;
   return;
  }

 // Fold the coefficients over.折系数。
 for(j=0; j<M/2; j++)WinCoeff[N-j-1] = WinCoeff[j];

 // This is the flat top if Alpha > 0. Cannot be applied to a Kaiser or Flat Top.如果Alpha> 0，则为平顶。不能应用于Kaiser或平顶。
 if(WindowType != wtKAISER &&  WindowType != wtFLATTOP)
  {
   for(j=M/2; j<N-M/2; j++)WinCoeff[j] = 1.0;
  }

 // This will set the gain of the window to 1. Only the Flattop window has unity gain by design. 这会将窗口的增益设置为1。仅Flattop窗口在设计上具有单位增益。
 if(UnityGain)
  {
   double Sum = 0.0;
   for(j=0; j<N; j++)Sum += WinCoeff[j];
   Sum /= (double)N;
   if(Sum != 0.0)for(j=0; j<N; j++)WinCoeff[j] /= Sum;
  }

 // Apply the window to the data.将窗口应用于数据。
 for(j=0; j<N; j++)Data[j] *= WinCoeff[j];

 delete[] WinCoeff;

}

//---------------------------------------------------------------------------

/*
The header file contents

#ifndef WindowedFIRFilterWebCodeH
#define WindowedFIRFilterWebCodeH


 enum TPassTypeName {LPF, HPF, BPF, NOTCH};
 enum TWindowType {wtNONE, wtKAISER, wtSINC, wtHANNING, wtHAMMING, wtBLACKMAN,
				   wtFLATTOP, wtBLACKMAN_HARRIS, wtBLACKMAN_NUTTALL, wtNUTTALL,
				   wtKAISER_BESSEL, wtTRAPEZOID, wtGAUSS, wtSINE, wtTEST };


 void BasicFIR(double *FirCoeff, int NumTaps, TPassTypeName PassType, double OmegaC, double BW, TWindowType WindowType, double WinBeta);
 void WindowData(double *Data, int N, TWindowType WindowType, double Alpha, double Beta, bool UnityGain);
 double Bessel(double x);
 double Sinc(double x);

#endif





头文件内容

#ifndef WindowedFIRFilterWebCodeH
#define WindowedFIRFilterWebCodeH


  枚举TPassTypeName {LPF，HPF，BPF，NOTCH};
  枚举TWindowType {wtNONE，wtKAISER，wtSINC，wtHANNING，wtHAMMING，wtBLACKMAN，
wtFLATTOP，wtBLACKMAN_HARRIS，wtBLACKMAN_NUTTALL，wtNUTTALL，
wtKAISER_BESSEL，wtTRAPEZOID，wtGAUSS，wtSINE，wtTEST}；


  void BasicFIR（double * FirCoeff，int NumTaps，TPassTypeName PassType，double OmegaC，double BW，TWindowType WindowType，double WinBeta）;
  void WindowData（double * Data，int N，TWindowType WindowType，double Alpha，double Beta，bool UnityGain）;
  双倍贝塞尔（双倍x）;
  双Sinc（双x）;

＃万一

*/