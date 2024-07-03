#pragma once

#ifndef NO_MANUAL_VECTORIZATION

  // Microsoft Visual C++ does not define __SSE__ or __SSE2__ but _M_IX86_FP instead
  // https://docs.microsoft.com/en-us/cpp/preprocessor/predefined-macros?view=msvc-170
  #ifdef _MSC_VER
    #if (defined(_M_AMD64) || defined(_M_X64) || defined(_M_IX86_FP) == 2)
      #define __SSE__
      #define __SSE2__
    #elif defined(_M_IX86_FP) == 1
      #define __SSE__
    #endif
  #endif

  #if defined(__AVX512F__)
    #define USE_AVX512

  #elif defined(__AVX__) || defined(__AVX2__)
    #define USE_AVX

  #elif defined(__SSE__) || defined(__SSE2__)
    #define USE_SSE

  #else
    #error "neither SSE, AVX nor AVX512 are defined"

  #endif
#endif

// TODO switch to only #include <immintrin.h>
// https://stackoverflow.com/questions/11228855/header-files-for-x86-simd-intrinsics
#if defined(USE_AVX) || defined(USE_SSE)
  #ifdef _MSC_VER
    #include <intrin.h>
    #include <stdexcept>
  #else
    #include <x86intrin.h>
  #endif
#endif
