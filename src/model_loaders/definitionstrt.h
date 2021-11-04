#ifndef DEFINITIONS_TRT_H
#define DEFINITIONS_TRT_H

/**
 * @brief structure d entete des ifchiers TensorRT.
 */
typedef struct SHeaderTRTFile
{
    /** Version du fichier. */
    uint32_t file_version;

    /** Version lib tensorrt. */
    uint32_t trt_version;

    /** Version CUDA. */
    uint32_t cuda_version;

    /** Version cudnn. */
    uint32_t cudnn_version;

    /** Version nvidia driver. */
    uint32_t nvidia_driver_version;

    /** Compute cabability du GPU. */
    uint32_t gpu_cc;

    /** CRC. */
    uint32_t crc;

    char reserve[128 - 7 * sizeof(uint32_t)];
} SHeaderTRTFile;

#endif