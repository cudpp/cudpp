/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and 
 * proprietary rights in and to this software and related documentation and 
 * any modifications thereto.  Any use, reproduction, disclosure, or distribution 
 * of this software and related documentation without an express license 
 * agreement from NVIDIA Corporation is strictly prohibited.
 * 
 */

#ifndef STREAM_TYPES_H
#define STREAM_TYPES_H


typedef char           S8;
typedef unsigned char  U8;
typedef short          S16;
typedef unsigned short U16;
typedef long           S32;
typedef unsigned long  U32;
typedef long long           S64;
typedef unsigned long long  U64;

typedef S64 NVTIME;
//typedef CRITICAL_SECTION NVCriticalSection;

// MPEG-1 PS: look for 2 consecutive MPEG-1 PS packs
enum PS_MPEG1_State {
    PS_MPEG1_Idle=0,
    PS_MPEG1_Pack,
    PS_MPEG1_Done
};

// MPEG-2 PS: look for 2 consecutive MPEG-2 PS packs
enum PS_MPEG2_State {
    PS_MPEG2_Idle=0,
    PS_MPEG2_Pack,
    PS_MPEG2_Done
};

enum {
    DEMUXCMD_NOP=0,
    DEMUXCMD_RUN,
    DEMUXCMD_RESET,
    DEMUXCMD_SEEK,
};

enum {DEMUXDETECT_BUFFER_SIZE = 64*1024 };  // 64KB buffer allocated during file-type detection


// MPEG-1 ES detection state
enum ES_MPEG1_State {
    ES_MPEG1_Idle=0,
    ES_MPEG1_Seq,
    ES_MPEG1_Gop,
    ES_MPEG1_Pic,
    ES_MPEG1_Done
};

// MPEG-2 ES detection state
enum ES_MPEG2_State {
    ES_MPEG2_Idle=0,
    ES_MPEG2_Seq,
    ES_MPEG2_SeqExt,
    ES_MPEG2_Gop,
    ES_MPEG2_Pic,
    ES_MPEG2_PicExt,
    ES_MPEG2_Done
};

// H.264 ES detection state
enum ES_H264_State {
    ES_H264_Idle=0,
    ES_H264_SPS,
    ES_H264_PPS,
    ES_H264_Slice,
    ES_H264_Done
};

// VC1 Advanced ES detection state
enum ES_VC1_State {
    ES_VC1_Idle=0,
    ES_VC1_Seq,
    ES_VC1_Gop,
    ES_VC1_Pic,
    ES_VC1_Done
};

// MPEG-4 ES detection state
enum ES_MPEG4_State {
    ES_MPEG4_Idle=0,
    ES_MPEG4_VOL,
    ES_MPEG4_Done
};

typedef struct _NVPackHeader
{
    S64 system_clock_reference;
    U32 scr_ext;
    S32 mux_rate;
    S32 system_header_length;       // 0 if no system header
} NVPackHeader;

// PES Packet
enum {
    PES_FLAGS_EXT           = 0x1,
    PES_FLAGS_CRC           = 0x2,
    PES_FLAGS_COPY          = 0x4,
    PES_FLAGS_DSM           = 0x8,
    PES_FLAGS_ES_RATE       = 0x10,
    PES_FLAGS_ESCR          = 0x20,
    PES_FLAGS_DTS           = 0x40,
    PES_FLAGS_PTS           = 0x80,
    PES_FLAGS_ORIGINAL      = 0x100,
    PES_FLAGS_COPYRIGHT     = 0x200,
    PES_FLAGS_DATAALIGN     = 0x400,
    PES_FLAGS_PRIORITY      = 0x800,
    PES_FLAGS_SCRAMBLING0   = 0x1000,
    PES_FLAGS_SCRAMBLING1   = 0x2000,
    PES_FLAGS_ZERO          = 0x4000,
    PES_FLAGS_MPEG2         = 0x8000,
};

typedef struct _NVPacketHeader
{
    U32 uFlags;     // PES_FLAGS_XXX
    U32 uReserved;
    S64 llPTS;
    S64 llDTS;
    const U8 *pPayload; // Payload data
    S32 lPayloadLength; // Payload data length
} NVPacketHeader;

// PSDemux definitions
enum {PS_BUFFER_SIZE=68*1024};
enum { // Start codes (Stream Ids)
    PACK_START_CODE          = 0xBA,
    SYSTEM_HEADER_CODE       = 0xBB,
    PROGRAM_STREAM_MAP       = 0xBC,
    PRIVATE_STREAM_1         = 0xBD,
    PADDING_STREAM           = 0xBE,
    PRIVATE_STREAM_2         = 0xBF,
    MPEG_AUDIO_STREAM        = 0xC0,
    MPEG2_EXT_AUDIO_STREAM   = 0xD0,
    MPEG_VIDEO_STREAM        = 0xE0,
    ECM_STREAM               = 0xF0,
    EMM_STREAM               = 0xF1,
    PROGRAM_STREAM_DIRECTORY = 0xFF
};
enum { // Private_Stream_1 SubStream Ids
    PCI_SUBSTREAM           = 0x00,
    DSI_SUBSTREAM           = 0x01,
    SUBPIC_SUBSTREAM        = 0x40,
    AC3_SUBSTREAM           = 0x80,
    DTS_SUBSTREAM           = 0x88,
    SDDS_SUBSTREAM          = 0x90,
    LPCM_SUBSTREAM          = 0xA0,
    AUDIO_SUBSTREAM_MASK    = 0xF8,
};

// TSDemux definitions
enum {TS_BUFFER_SIZE = 64*1024 };   // I/O buffer size
enum {TSP_SIZE  = 188};     // Fixed size of a transport packet
enum {SYNC_BYTE = 0x47};    // sync_byte
enum {TS_INVALID_VERSION = 0xfffffff0};
enum {TS_MAX_PROGRAMS = 32};   // Maximum number of programs (excluding reserved PIDs)
enum {TS_MAX_SECTIONS = 64};  // Maximum number of section PIDs
enum {TS_SECTION_SIZE = 1024};  // Maximum size of a section
enum { // 2nd byte fields
    TS1_ERROR       = 0x80, // transport_error_indicator
    TS1_START       = 0x40, // payload_unit_start_indicator
    TS1_PRIORITY    = 0x20, // transport_priority
    TS1_PID_HI      = 0x1F, // Upper 5 bits of PID
};
enum { TS2_PID_LO   = 0xFF }; // Lower 8-bits of PID
enum { // 4th byte fields
    TS3_SCTL_MASK   = 0xC0, // transport_scrambling_control
    TS3_ADAPT       = 0x20, // adaptation_field_control
    TS3_PAYLOAD     = 0x10, // payload present
    TS3_CONT_MASK   = 0x0F, // continuity_counter
};
enum { // adaptation_field_flags
    TS_AF_DISCONTINUITY = 0x80, // discontinuity_indicator
    TS_AF_RANDOMACCESS  = 0x40, // random_access_indicator
    TS_AF_ESPRIORITY    = 0x20, // elementary_stream_priority_indicator
    TS_AF_PCR           = 0x10, // PCR_flag
    TS_AF_OPCR          = 0x08, // OPCR_flag
    TS_AF_SPLICEPOINT   = 0x04, // splicing_point_flag
    TS_AF_PRIVATEDATA   = 0x02, // transport_private_data_flag
    TS_AF_EXTENSION     = 0x01, // adaptation_field_extension_flag
};
enum {
    TS_PID_PAT          = 0x0000,   // Program Association Table
    TS_PID_CAT          = 0x0001,   // Conditional Access Table
    TS_PID_TSTD         = 0x0002,   // Transport Stream Description Table
    TS_PID_MIN          = 0x0010,
    TS_PID_PSIP         = 0x1FFB,   // PSIP Table
    TS_PID_MAX          = 0x1FFE,
    TS_PID_NULL         = 0x1FFF,
    // Default PIDs
    TS_PID_DEFAULT_PROGRAM_MAP = 0x100,
};
// stream_type
enum {
    STRMTYPE_MPEG1VID    = 0x01,        // MPEG 1 VIDEO  ISO/IEC 11172-2 Video 
    STRMTYPE_MPEG2VID    = 0x02,        // MPEG 2 VIDEO  ISO/IEC 13818-2 Video 
    STRMTYPE_MPEG1AUD    = 0x03,        // MPEG 1 AUDIO  ISO/IEC 11172-3 Audio  
    STRMTYPE_MPEG2AUD    = 0x04,        // MPEG 2 AUDIO  ISO/IEC 13818-3 Audio 
    STRMTYPE_PRIVATE_PES = 0x06,        // PRIVATE PES STREAM
    STRMTYPE_H264VID     = 0x1B,        // MPEG 4 VIDEO  ISO/IEC 14496-10 Video
    STRMTYPE_ATSC_A53    = 0x81,        // ATSC A/53 (AC3) AUDIO
    STRMTYPE_LPCM        = 0x83,        // LPCM Audio
    STRMTYPE_VC1         = 0xEA,        // VC-1 Video
};

// RCV Demux information and detection
typedef struct _RCVFileHeader
{
    S32 lNumFrames;
    S32 bRCVIsV2Format;
    U32 uProfile;
    S32 lMaxCodedWidth;
    S32 lMaxCodedHeight;
    S32 lHrdBuffer;
    S32 lBitRate;
    S32 lFrameRate;
    S32 cbSeqHdr;       // Should always be 4 for simple/main
    U8 SeqHdrData[32];
} RCVFileHeader;

enum { RCV_VC1_TYPE = 0x85 };
enum { RCV_V2_MASK  = (1<<6) }; // Bit 6 of the type indicates V1 if 0, V2 if 1
enum { RCV_V2_FRAMESIZE_FLAGS = 0xf0000000 }; // Top nibble bits of frame size word are flags in V2
enum { RCV_V2_KEYFRAME_FLAG = 0x80000000 };
enum { RCV_V2_VBR_FLAG      = 0x10000000 }; // V2 extra information has a VBR flag
enum { RCV_MAX_FRAME_SIZE   = 2048*1024 };  // Maximum compressed frame size

// MuxFormats
enum NvMuxFormat
{
    NVMF_Invalid=-1,            // Invalid
    NVMF_ElementaryStream=0,    // Elementary stream (no system layer, ie: mux/demux bypass)
    NVMF_MPEG2_Program,         // MPEG-2 Program Stream
    NVMF_MPEG2_Transport,       // MPEG-2 Transport Stream
    NVMF_MPEG4,                 // MP4 File Format
    NVMF_AVI,                   // AVI
    NVMF_ASF,                   // ASF/WMV
    NVMF_RCV,                   // RCV
};

inline U32 _byteswap_ulong(U32 x)
{
   U8 temp[4];
   temp[0] = ((U8 *)&x)[3];
   temp[1] = ((U8 *)&x)[2];
   temp[2] = ((U8 *)&x)[1];
   temp[3] = ((U8 *)&x)[0];
   return *(U32 *)temp;
}

#define SWAP_U32(x) _byteswap_ulong(x)
#define U32_BE(u32) SWAP_U32(u32)
#define BYTEINDEX(pos)  ((pos)^3)

typedef struct _MP4Box
{
    U32 box_size;
    U32 box_type;
} MP4Box;

typedef struct _MP4LongBox
{
    U32 box_size_1;   // =1
    U32 box_type;
    U64 box_size;     // 64-bit size
} MP4LongBox;

// Stream category
enum NvMuxStreamCategory
{
    NVMux_InvalidCategory=-1,
    NVMux_Video=0,
    NVMux_Audio,
    NVMux_Subpic,
};


// Elementary stream types
enum NvMuxVideoStreamType
{
    NVMux_Video_Unknown=-1,     // Unknown video stream type
    NVMux_Video_NotPresent=0,   // Video stream is not present
    NVMux_Video_MPEG,           // MPEG-1/2 Video
    NVMux_Video_MPEG4,          // MPEG-4 Video
    NVMux_Video_H264,           // H.264 Video
    NVMux_Video_VC1,            // VC1 Video
};

enum NvMuxAudioStreamType
{
    NVMux_Audio_Unknown=-1,     // Unknown audio stream type
    NVMux_Audio_NotPresent=0,   // Audio stream is not present
    NVMux_Audio_MPEG,           // MPEG-1/2 Audio
    NVMux_Audio_AC3,            // AC3 Audio
    NVMux_Audio_LPCM,           // LPCM Audio
    NVMux_Audio_AAC,            // AAC Audio
};

typedef struct _NVMPEG2PictureData
{
//    IPicBuf *pForwardRef;           // Forward reference (P/B-frames)
//    IPicBuf *pBackwardRef;          // Backward reference (B-frames)
    int picture_coding_type;        // TYPE_?FRAME
    int full_pel_forward_vector;
    int full_pel_backward_vector;
    int f_code[2][2];
    int intra_dc_precision;
    int frame_pred_frame_dct;
    int concealment_motion_vectors;
    int q_scale_type;
    int intra_vlc_format;
    int alternate_scan;
    // Quantization matrices (raster order)
    unsigned char QuantMatrixIntra[64];
    unsigned char QuantMatrixInter[64];
} NVMPEG2PictureData;


typedef struct _NVH264DPBEntry
{
//    IPicBuf *pPicBuf;       // ptr to reference frame
    int FrameIdx;           // frame_num(short-term) or LongTermFrameIdx(long-term)
    int is_long_term;       // 0=short term reference, 1=long term reference
    int not_existing;       // non-existing reference frame (corresponding PicIdx should be set to -1)
    int used_for_reference; // 0=unused, 1=top_field, 2=bottom_field, 3=both_fields
    int FieldOrderCnt[2];   // field order count of top and bottom fields
} NVH264DPBEntry;


typedef struct _NVH264PictureData
{
    // SPS
    int log2_max_frame_num_minus4;
    int pic_order_cnt_type;
    int log2_max_pic_order_cnt_lsb_minus4;
    int delta_pic_order_always_zero_flag;
    int frame_mbs_only_flag;
    int direct_8x8_inference_flag;
    int num_ref_frames;
    int residual_colour_transform_flag;
    int qpprime_y_zero_transform_bypass_flag;
    // PPS
    int entropy_coding_mode_flag;
    int pic_order_present_flag;
    int num_ref_idx_l0_active_minus1;
    int num_ref_idx_l1_active_minus1;
    int weighted_pred_flag;
    int weighted_bipred_idc;
    int pic_init_qp_minus26;
    int deblocking_filter_control_present_flag;
    int redundant_pic_cnt_present_flag;
    int transform_8x8_mode_flag;
    int MbaffFrameFlag;
    int constrained_intra_pred_flag;
    int chroma_qp_index_offset;
    int second_chroma_qp_index_offset;
    int frame_num;
    int CurrFieldOrderCnt[2];
    // DPB
    NVH264DPBEntry dpb[16];          // List of reference frames within the DPB
    // Quantization Matrices (raster-order)
    unsigned char WeightScale4x4[6][16];
    unsigned char WeightScale8x8[2][64];
} NVH264PictureData;


typedef struct _NVVC1PictureData
{
//    IPicBuf *pForwardRef;   // Forward reference (P/B-frames)
//    IPicBuf *pBackwardRef;  // Backward reference (B-frames)
    int FrameWidth;         // Actual frame width
    int FrameHeight;        // Actual frame height
    // SEQUENCE
    int profile;
    int postprocflag;
    int pulldown;
    int interlace;
    int tfcntrflag;
    int finterpflag;
    int psf;
    int multires;
    int syncmarker;
    int rangered;
    int maxbframes;
    // ENTRYPOINT
    int panscan_flag;
    int refdist_flag;
    int extended_mv;
    int dquant;
    int vstransform;
    int loopfilter;
    int fastuvmc;
    int overlap;
    int quantizer;
    int extended_dmv;
    int range_mapy_flag;
    int range_mapy;
    int range_mapuv_flag;
    int range_mapuv;
} NVVC1PictureData;


typedef struct _NVDPictureData
{
    int PicWidthInMbs;      // Coded Frame Size
    int FrameHeightInMbs;   // Coded Frame Height
//    IPicBuf *pCurrPic;      // Current picture (output)
    int field_pic_flag;     // 0=frame picture, 1=field picture
    int bottom_field_flag;  // 0=top field, 1=bottom field (ignored if field_pic_flag=0)
    int second_field;       // Second field of a complementary field pair
    int progressive_frame;  // Frame is progressive
    int top_field_first;    // Frame pictures only
    int repeat_first_field; // For 3:2 pulldown (number of additional fields, 2=frame doubling, 4=frame tripling)
    int ref_pic_flag;       // Frame is a reference frame
    int intra_pic_flag;     // Frame is entirely intra coded (no temporal dependencies)
    int chroma_format;      // Chroma Format (should match sequence info)
    // Bitstream data
    unsigned int nBitstreamDataLen;        // Number of bytes in bitstream data buffer
    const unsigned char *pBitstreamData;   // Ptr to bitstream data for this picture (slice-layer)
    unsigned int nNumSlices;               // Number of slices in this picture
    const unsigned int *pSliceDataOffsets; // nNumSlices entries, contains offset of each slice within the bitstream data buffer
    // Codec-specific data
    union {
        NVMPEG2PictureData mpeg2;   // Also used for MPEG-1
        NVH264PictureData h264;
        NVVC1PictureData vc1;
    } CodecSpecific;
} NVDPictureData;


// Packet input for parsing
typedef struct _NVDBitstreamPacket
{
    const U8 *pByteStream;  // Ptr to byte stream data
    S32 nDataLength;        // Data length for this packet
    int bEOS;               // TRUE if this is an End-Of-Stream packet (flush everything)
    int bPTSValid;          // TRUE if llPTS is valid (also used to detect frame boundaries for VC1 SP/MP)
    S64 llPTS;              // Presentation Time Stamp for this packet (clock rate specified at initialization)
} NVDBitstreamPacket;

#if !USE_CUVID_SOURCE
// Compression Standard
enum NvVideoCompressionStd
{
    NVCS_Unknown=-1,
    NVCS_MPEG1=0,   // 11172
    NVCS_MPEG2,     // 13818
    NVCS_MPEG4,     // 14496-2
    NVCS_VC1,       // VC1
    NVCS_H264,      // 14496-10
};

enum NvFrameRate
{
    NV_FRAME_RATE_12 = 0,
    NV_FRAME_RATE_12_5,
    NV_FRAME_RATE_14_98,
    NV_FRAME_RATE_15,
    NV_FRAME_RATE_23_97,
    NV_FRAME_RATE_24,
    NV_FRAME_RATE_25,
    NV_FRAME_RATE_29_97,
    NV_FRAME_RATE_30,
    NV_FRAME_RATE_50,
    NV_FRAME_RATE_59_94,
    NV_FRAME_RATE_60,
    NV_NUM_FRAME_RATES,
    NV_FRAME_RATE_UNKNOWN    // Unknown/unspecified frame rate (or variable)
};

// Frame rate description (as a fraction and as an aproximate time per frame in 100ns units)
typedef struct _NVFrameRateDesc
{
    S32 lNumerator;
    S32 lDenominator;
    NVTIME llAvgTimePerFrame;
} NVFrameRateDesc;

// Definitions for video_format
enum {
    NVEVideoFormat_Component=0,
    NVEVideoFormat_PAL,
    NVEVideoFormat_NTSC,
    NVEVideoFormat_SECAM,
    NVEVideoFormat_MAC,
    NVEVideoFormat_Unspecified,
    NVEVideoFormat_Reserved6,
    NVEVideoFormat_Reserved7
};

enum {
    NVEColorPrimaries_Forbidden=0,
    NVEColorPrimaries_BT709,
    NVEColorPrimaries_Unspecified,
    NVEColorPrimaries_Reserved,
    NVEColorPrimaries_BT470M,
    NVEColorPrimaries_BT470BG,
    NVEColorPrimaries_SMPTE170M,
    NVEColorPrimaries_SMPTE240M,
    NVEColorPrimaries_GenericFilm
};

// Definitions for transfer_characteristics
enum {
    NVETransferCharacteristics_Forbidden=0,
    NVETransferCharacteristics_BT709,
    NVETransferCharacteristics_Unspecified,
    NVETransferCharacteristics_Reserved,
    NVETransferCharacteristics_BT470M,
    NVETransferCharacteristics_BT470BG,
    NVETransferCharacteristics_SMPTE170M,
    NVETransferCharacteristics_SMPTE240M,
    NVETransferCharacteristics_Linear,
    NVETransferCharacteristics_Log100,
    NVETransferCharacteristics_Log316
};

// Definitions for matrix_coefficients
enum {
    NVEMatrixCoefficients_Forbidden=0,
    NVEMatrixCoefficients_BT709,
    NVEMatrixCoefficients_Unspecified,
    NVEMatrixCoefficients_Reserved,
    NVEMatrixCoefficients_FCC,
    NVEMatrixCoefficients_BT470BG,
    NVEMatrixCoefficients_SMPTE170M,
    NVEMatrixCoefficients_SMPTE240M
};

static const NvFrameRate mpeg2_frame_rate_table[16] =
{
    NV_FRAME_RATE_UNKNOWN,
    NV_FRAME_RATE_23_97,
    NV_FRAME_RATE_24,
    NV_FRAME_RATE_25,
    NV_FRAME_RATE_29_97,
    NV_FRAME_RATE_30,
    NV_FRAME_RATE_50,
    NV_FRAME_RATE_59_94,
    NV_FRAME_RATE_60,
    NV_FRAME_RATE_UNKNOWN,
    NV_FRAME_RATE_UNKNOWN,
    NV_FRAME_RATE_UNKNOWN,
    NV_FRAME_RATE_UNKNOWN,
    NV_FRAME_RATE_UNKNOWN,
    NV_FRAME_RATE_UNKNOWN,
    NV_FRAME_RATE_UNKNOWN,
};

// MPEG-1 pel_aspect_ratio table (x10000)
static const U16 mpeg1_par_table[13] =
{
    6735, 7031,  7615,  8055,  8437,  8935, 9157, 9815, 10255, 10695, 10950, 11575, 12015
};

static const U16 mpeg2_dar_table[3][2] =
{
    {4,3}, {16,9}, {221,100}
};

const NVFrameRateDesc g_FrameRateDesc[NV_NUM_FRAME_RATES+2] =
{
    { 12000,  1000, 833333 },   // 12
    { 12500,  1000, 800000 },   // 12.5
    { 15000,  1001, 667333 },   // 14.985
    { 15000,  1000, 666666 },   // 15
    { 24000,  1001, 417083 },   // 23.976
    { 24000,  1000, 416666 },   // 24
    { 25000,  1000, 400000 },   // 25
    { 30000,  1001, 333666 },   // 29.97
    { 30000,  1000, 333333 },   // 30
    { 50000,  1000, 200000 },   // 50
    { 60000,  1001, 166833 },   // 59.94
    { 60000,  1000, 166666 },   // 60
    // 2 dummy entries in case someone attempts to index the array with NV_FRAME_RATE_UNKNOWN
    {     0,     0,      0 },
    {     0,     0,      0 },
};

// Picture type
#define TYPE_IFRAME     1
#define TYPE_PFRAME     2
#define TYPE_BFRAME     3

#define MAX_SEQ_HDR_LEN (512) // 512 bvytes

#define ES_BUFFER_SIZE      (256*1024)   // 256KB buffer

	//////////////////////////////////////////////////////////////////////////////////////////
	//
	// VC-1
	//

	// Sample Aspect Ratio table (6.1.14.3.1)
	static const unsigned char vc1_sar_tab[15][2] =
	{
		{0,0}, {1,1}, {12,11}, {10,11}, {16,11}, {40,33}, {24,11}, {20,11},
		{32,11}, {80,33}, {18,11}, {15,11}, {64,33}, {160,99}, {0,0}
	};
	// Frame rate nr table
	static const unsigned char vc1_frameratenr_tab[8] =
	{
		0, 24, 25, 30, 50, 60, 48, 72
	};
	// Frame rate dr table
	static const unsigned short vc1_frameratedr_tab[4] =
	{
		0, 1000, 1001, 0
	};


	// Sequence information
	typedef struct _NVDSequenceInfo
	{
		NvVideoCompressionStd eCodec;   // Compression Standard
		NvFrameRate eFrameRate;         // Frame Rate stored in the bitstream
		int bProgSeq;                   // Progressive Sequence
		int nDisplayWidth;              // Displayed Horizontal Size
		int nDisplayHeight;             // Displayed Vertical Size
		int nCodedWidth;                // Coded Picture Width
		int nCodedHeight;               // Coded Picture Height
		U8 nChromaFormat;               // Chroma Format (0=4:0:0, 1=4:2:0, 2=4:2:2, 3=4:4:4)
		U8 uBitDepthLumaMinus8;         // Luma bit depth (0=8bit)
		U8 uBitDepthChromaMinus8;       // Chroma bit depth (0=8bit)
		U8 uReserved1;                  // For alignment
		S32 lBitrate;                   // Video bitrate (bps)
		S32 lDARWidth, lDARHeight;      // Display Aspect Ratio = lDARWidth : lDARHeight
		S32 lVideoFormat;               // Video Format (NVEVideoFormat_XXX)
		S32 lColorPrimaries;            // Colour Primaries (NVEColorPrimaries_XXX)
		S32 lTransferCharacteristics;   // Transfer Characteristics
		S32 lMatrixCoefficients;        // Matrix Coefficients
		S32 cbSequenceHeader;           // Number of bytes in SequenceHeaderData
		U8 SequenceHeaderData[MAX_SEQ_HDR_LEN]; // Raw sequence header data (codec-specific)
	} NVDSequenceInfo;

	// Mux/Demux thread state
	enum NvDemuxState
	{
		NvDemuxState_Stopped=0,
		NvDemuxState_Starting,
		NvDemuxState_Stopping,
		NvDemuxState_Paused,
		NvDemuxState_Running,
	};

	#define NVDP_FLAGS_EOS              0x01    // End of stream (this is the last packet to be delivered)
	#define NVDP_FLAGS_PTSVALID         0x02    // packet has a PTS
	#define NVDP_FLAGS_DISCONTINUITY    0x04    // Data discontinuity flag (packet drop, seek operation, etc)
	#define NVDP_FLAGS_STREAMID(flags)  (((U32)(flags))>>24) // Upper 8-bits store the stream index

	typedef struct _NVDemuxPacket
	{
		const U8 *pbData;   // Ptr to raw data (temporary ptr)
		S32 lDataLen;       // length of data in bytes
		U32 uFlags;         // Packet flags (NVDP_FLAGS_XXX)
		S64 llPTS;          // PTS if any
	} NVDemuxPacket;


	enum {MAX_VIDEO_STREAMS=2};
	enum {MAX_AUDIO_STREAMS=8};

	typedef struct _NvDemuxVideoStream
	{
		NvMuxVideoStreamType vidType;
		U32 uPrivateID;     // Format-specific id
		S64 llStartTime;    // Stream start time (first DTS or PTS)
		bool bDiscontinuity;
		NVDSequenceInfo vidFormat;
	} NvDemuxVideoStream;

	typedef enum _NvAudioCodec
	{
		NVAC_UNKNOWN=0,
		NVAC_MPEG1,
		NVAC_MPEG2,
		NVAC_MP3,   // MPEG-1 Layer III
		NVAC_LPCM,
		NVAC_AC3,
		NVAC_DTS,
		NVAC_SDDS,
		NVAC_AAC,
	}NvAudioCodec;

	typedef struct _NVAudioFormat
	{
		NvAudioCodec eCodec;
		S32 lChannels;
		S32 lSamplesPerSec;
		S32 lBitrate;       // For uncompressed, can also be used to determine bits per sample
		S32 lFrameSize;
		U32 uHeader;        // Format-specific header
	}NVAudioFormat;

	typedef struct _NvDemuxAudioStream
	{
		NvMuxAudioStreamType audType;
		U32 uPrivateID;     // Format-specific id
		S64 llStartTime;    // Stream start time (first PTS)
		bool bDiscontinuity;
		NVAudioFormat audFormat;
	} NvDemuxAudioStream;

	// Simplify an aspect ratio fraction (both inputs must be positive)
	inline void SimplifyAspectRatio(S32 *pARWidth, S32 *pARHeight)
	{
		U32 a = abs(*pARWidth), b = abs(*pARHeight);
		while (a)
		{
			U32 tmp = a;
			a = b % tmp;
			b = tmp;
		}
		if (b)
		{
			*pARWidth /= (S32)b;
			*pARHeight /= (S32)b;
		}
	}

	// Convert AvgTimePerFrame to the closest frame rate code
	inline NvFrameRate FindClosestFrameRate(S64 llAvgTimePerFrame, S32 lUnits)
	{
		S32 llBestErr = (S32)g_FrameRateDesc[0].llAvgTimePerFrame;
		int nBestMatch = 0;
	    
		if ((lUnits != 10000000) && (lUnits > 0))
		{
			llAvgTimePerFrame = (llAvgTimePerFrame * 10000000LL) / lUnits;
		}
		for (int i=0; i<NV_NUM_FRAME_RATES; i++)
		{
			S32 llErr = abs((S32)(llAvgTimePerFrame - g_FrameRateDesc[i].llAvgTimePerFrame));
			if ((!i) || (llErr < llBestErr))
			{
				llBestErr = llErr;
				nBestMatch = i;
			}
		}
		return (NvFrameRate)nBestMatch;
	}

	// Return the number of bits necessary to represent n (n must be positive)
	static inline S32 Log2U31(S32 n)
	{
//		assert(n >= 0);
		S32 sz = 0;
		while (n)
		{
			sz++;
			n >>= 1;
		}
		return sz;
	}


#endif

#endif
