
#include <stdio.h>

void convolve(const double Signal[/* SignalLen */], 
              long SignalLen,
              const double Kernel[/* KernelLen */], 
              long KernelLen,
              double Result[/* SignalLen + KernelLen - 1 */]);