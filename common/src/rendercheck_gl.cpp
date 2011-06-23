/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and 
 * proprietary rights in and to this software and related documentation. 
 * Any use, reproduction, disclosure, or distribution of this software 
 * and related documentation without an express license agreement from
 * NVIDIA Corporation is strictly prohibited.
 *
 * Please refer to the applicable NVIDIA end user license agreement (EULA) 
 * associated with this source code for terms and conditions that govern 
 * your use of this NVIDIA software.
 * 
 */

//
// CheckBackBuffer/FBO verification Class
//
//  This piece of code is used to to allow us verify the contents from the frame bufffer
//  of graphics samples.  We can render to the CheckBackBuffer or FBO in OpenGL, and read the
//  contents back for the purpose of comparisons
//
//  http://www.nvidia.com/dev_content/nvopenglspecs/GL_EXT_framebuffer_object.txt
//
// Authors: Mark Harris, Evan Hart, Simon Green, and Eric Young
// Email: sdkfeedback@nvidia.com
//
// Copyright (c) NVIDIA Corporation. All rights reserved.
////////////////////////////////////////////////////////////////////////////////

#include <cutil.h>
#include "rendercheck_gl.h"


// CheckRender::Base class for CheckRender (there will be default functions all classes derived from this)
CheckRender::CheckRender(unsigned int width, unsigned int height, unsigned int Bpp,
                         bool bQAReadback, bool bUseFBO, bool bUsePBO) 
                         :
    m_Width(width), m_Height(height), m_Bpp(Bpp), m_bQAReadback(bQAReadback),
    m_bUseFBO(bUseFBO), m_bUsePBO(bUsePBO), m_PixelFormat(GL_BGRA), m_fThresholdCompare(0.0f)
{
    allocateMemory(width, height, Bpp, bQAReadback, bUseFBO, bUsePBO);
}

CheckRender::~CheckRender() 
{
	// Release PBO resources
    if (m_bUsePBO) {
        glDeleteBuffersARB(1, &m_pboReadback);
	    m_pboReadback = 0;            
    }

    free(m_pImageData);
}

void 
CheckRender::allocateMemory( unsigned int width, unsigned int height, unsigned int Bpp,
                            bool bQAReadback, bool bUseFBO, bool bUsePBO )
{
    // Create the PBO for readbacks
    if (bUsePBO) {
        glGenBuffersARB(1, &m_pboReadback);
        glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, m_pboReadback);
        glBufferDataARB(GL_PIXEL_UNPACK_BUFFER_ARB, width*height*Bpp, NULL, GL_STREAM_READ);
        glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
    }

    m_pImageData = (unsigned char *)malloc(width*height*Bpp);  // This is the image data stored in system memory
}

void 
CheckRender::savePGM(  const char *zfilename, bool bInvert, void **ppReadBuf )
{
    if (zfilename != NULL) {
        if (bInvert) {
            unsigned char *readBuf;
            unsigned char *writeBuf= (unsigned char *)malloc(m_Width * m_Height);

            for (unsigned int y=0; y < m_Height; y++) {
                if (ppReadBuf) {
                    readBuf = *(unsigned char **)ppReadBuf;
                } else {
                    readBuf = (unsigned char *)m_pImageData;
                }
                memcpy(&writeBuf[m_Width*m_Bpp*y], (readBuf+ m_Width*(m_Height-1-y)), m_Width);
            }
            // we copy the results back to original system buffer
            if (ppReadBuf) {
                memcpy(*ppReadBuf, writeBuf, m_Width*m_Height);
            } else {
                memcpy(m_pImageData, writeBuf, m_Width*m_Height);
            }
            free (writeBuf);
        }
        printf("> Saving PGM: <%s>\n", zfilename);
        if (ppReadBuf) {
		    cutSavePGMub(zfilename, *(unsigned char **)ppReadBuf, m_Width, m_Height);
        } else {
		    cutSavePGMub(zfilename, (unsigned char *)m_pImageData, m_Width, m_Height);
        }
    }
}

void 
CheckRender::savePPM(  const char *zfilename, bool bInvert, void **ppReadBuf )
{
    if (zfilename != NULL) {
        if (bInvert) {
            unsigned char *readBuf;
            unsigned char *writeBuf= (unsigned char *)malloc(m_Width * m_Height * m_Bpp);

            for (unsigned int y=0; y < m_Height; y++) {
                if (ppReadBuf) {
                    readBuf = *(unsigned char **)ppReadBuf;
                } else {
                    readBuf = (unsigned char *)m_pImageData;
                }
                memcpy(&writeBuf[m_Width*m_Bpp*y], (readBuf+ m_Width*m_Bpp*(m_Height-1-y)), m_Width*m_Bpp);
            }
            // we copy the results back to original system buffer
            if (ppReadBuf) {
                memcpy(*ppReadBuf, writeBuf, m_Width*m_Height*m_Bpp);
            } else {
                memcpy(m_pImageData, writeBuf, m_Width*m_Height*m_Bpp);
            }
            free (writeBuf);
        }
        printf("> Saving PPM: <%s>\n", zfilename);
        if (ppReadBuf) {
		    cutSavePPM4ub(zfilename, *(unsigned char **)ppReadBuf, m_Width, m_Height);
        } else {
		    cutSavePPM4ub(zfilename, (unsigned char *)m_pImageData, m_Width, m_Height);
        }
    }
}

bool 
CheckRender::PGMvsPGM( const char *src_file, const char *ref_file, const float epsilon, const float threshold )
{
    unsigned char *src_data = NULL, *ref_data = NULL;
    unsigned long error_count = 0;
    unsigned int width, height;

    char *ref_file_path = cutFindFilePath(ref_file, m_ExecPath);
    if (ref_file_path == NULL) {
        printf("CheckRender::PGMvsPGM unable to find <%s> in <%s> Aborting comparison!\n", ref_file, m_ExecPath);
        printf(">>> Check info.xml and [project//data] folder <%s> <<<\n", ref_file);
        printf("Aborting comparison!\n", ref_file, m_ExecPath);
        printf("  FAILED\n");
        error_count++;
    } else {

        if (src_file == NULL || ref_file_path == NULL) {
            printf("PGMvsPGM: Aborting comparison\n");
            return false;
        }
		printf("   src_file <%s>\n", src_file);
		printf("   ref_file <%s>\n", ref_file_path);

        if (cutLoadPGMub(ref_file_path, &ref_data, &width, &height) != CUTTrue) {
            printf("PGMvsPGM: unable to load ref image file: %s\n", ref_file_path);
            return false;
        }

        if (cutLoadPGMub(src_file, &src_data, &width, &height) != CUTTrue) {
            printf("PGMvsPGM: unable to load src image file: %s\n", src_file);
            return false;
        }

        printf("PGMvsPGM: comparing images size (%d,%d) epsilon(%2.4f), threshold(%4.2f%%)\n", m_Height, m_Width, epsilon, threshold*100);
        if (cutCompareubt( ref_data, src_data, m_Height*m_Width, epsilon, threshold ) == CUTFalse) {
            error_count = 1;
        }
    }

    if (error_count == 0) { 
        printf("  OK\n"); 
    } else {
		printf("  FAILURE: %d errors...\n", (unsigned int)error_count);
    }
    return (error_count == 0);  // returns true if all pixels pass
}

bool 
CheckRender::PPMvsPPM( const char *src_file, const char *ref_file, const float epsilon, const float threshold )
{
    unsigned char *src_data = NULL, *ref_data = NULL;
    unsigned long error_count = 0;

    char *ref_file_path = cutFindFilePath(ref_file, m_ExecPath);
    if (ref_file_path == NULL) {
        printf("CheckRender::PPMvsPPM unable to find <%s> in <%s> Aborting comparison!\n", ref_file, m_ExecPath);
        printf(">>> Check info.xml and [project//data] folder <%s> <<<\n", ref_file);
        printf("Aborting comparison!\n", ref_file, m_ExecPath);
        printf("  FAILED\n");
        error_count++;
    } 

    if (src_file == NULL || ref_file_path == NULL) {
        printf("PPMvsPPM: Aborting comparison\n");
        return false;
    }
	printf("   src_file <%s>\n", src_file);
	printf("   ref_file <%s>\n", ref_file_path);
    return (cutComparePPM( src_file, ref_file_path, epsilon, threshold, true ) == CUTTrue ? true : false);
}

void CheckRender::dumpBin(void *data, unsigned int bytes, const char *filename) 
{
    printf("CheckRender::dumpBin: <%s>\n", filename);
    FILE *fp = fopen(filename, "wb");
    fwrite(data, bytes, 1, fp);
    fflush(fp);
    fclose(fp);
}

bool CheckRender::compareBin2BinUint(const char *src_file, const char *ref_file, unsigned int nelements, const float epsilon, const float threshold)
{
    unsigned int *src_buffer, *ref_buffer;
    FILE *src_fp = NULL, *ref_fp = NULL;

    unsigned long error_count = 0;

    if ((src_fp = fopen(src_file, "rb")) == NULL) {
        printf("compareBin2Bin <unsigned int> unable to open src_file: %s\n", src_file);   
        error_count++;
    }
    char *ref_file_path = cutFindFilePath(ref_file, m_ExecPath);
    if (ref_file_path == NULL) {
        printf("compareBin2Bin <unsigned int>  unable to find <%s> in <%s>\n");
        printf(">>> Check info.xml and [project//data] folder <%s> <<<\n", ref_file);
        printf("Aborting comparison!\n", ref_file, m_ExecPath);
        printf("  FAILED\n");
        error_count++;

        if (src_fp) fclose(src_fp);
        if (ref_fp) fclose(ref_fp);
    } 
    else 
    {
        if ((ref_fp = fopen(ref_file_path, "rb")) == NULL) {
            printf("compareBin2Bin <unsigned int>  unable to open ref_file: %s\n", ref_file_path);   
            error_count++;
        }

        if (src_fp && ref_fp) {
            src_buffer = (unsigned int *)malloc(nelements*sizeof(unsigned int));
            ref_buffer = (unsigned int *)malloc(nelements*sizeof(unsigned int));

            fread(src_buffer, nelements, sizeof(unsigned int), src_fp);
            fread(ref_buffer, nelements, sizeof(unsigned int), ref_fp);

            printf("> compareBin2Bin <unsigned int> nelements=%d, epsilon=%4.2f, threshold=%4.2f\n", nelements, epsilon, threshold);
			printf("   src_file <%s>\n", src_file);
			printf("   ref_file <%s>\n", ref_file_path);
            if (!cutCompareuit( ref_buffer, src_buffer, nelements, epsilon, threshold))
                error_count++;

            fclose(src_fp);
            fclose(ref_fp);

            free(src_buffer);
            free(ref_buffer);
        } else {
            if (src_fp) fclose(src_fp);
            if (ref_fp) fclose(ref_fp);
        }
    }

    if (error_count == 0) { 
        printf("  OK\n"); 
    } else {
		printf("  FAILURE: %d errors...\n", (unsigned int)error_count);
    }

    return (error_count == 0);  // returns true if all pixels pass
}

bool CheckRender::compareBin2BinFloat(const char *src_file, const char *ref_file, unsigned int nelements, const float epsilon, const float threshold)
{
    float *src_buffer, *ref_buffer;
    FILE *src_fp = NULL, *ref_fp = NULL;

    unsigned long error_count = 0;

    if ((src_fp = fopen(src_file, "rb")) == NULL) {
        printf("compareBin2Bin <float> unable to open src_file: %s\n", src_file);   
        error_count = 1;
    }
    char *ref_file_path = cutFindFilePath(ref_file, m_ExecPath);
    if (ref_file_path == NULL) {
        printf("compareBin2Bin <float> unable to find <%s> in <%s>\n", ref_file, m_ExecPath);
        printf(">>> Check info.xml and [project//data] folder <%s> <<<\n", m_ExecPath);
        printf("Aborting comparison!\n");
        printf("  FAILED\n");
        error_count++;

        if (src_fp) fclose(src_fp);
        if (ref_fp) fclose(ref_fp);
    } 
    else 
    {
        if ((ref_fp = fopen(ref_file_path, "rb")) == NULL) {
            printf("compareBin2Bin <float> unable to open ref_file: %s\n", ref_file_path);   
            error_count = 1;
        }

        if (src_fp && ref_fp) {
            src_buffer = (float *)malloc(nelements*sizeof(float));
            ref_buffer = (float *)malloc(nelements*sizeof(float));

            fread(src_buffer, nelements, sizeof(float), src_fp);
            fread(ref_buffer, nelements, sizeof(float), ref_fp);

            printf("> compareBin2Bin <float> nelements=%d, epsilon=%4.2f, threshold=%4.2f\n", nelements, epsilon, threshold);
			printf("   src_file <%s>\n", src_file);
			printf("   ref_file <%s>\n", ref_file_path);

            if (!cutComparefet( ref_buffer, src_buffer, nelements, epsilon, threshold)) {
                error_count++;
            }

            fclose(src_fp);
            fclose(ref_fp);

            free(src_buffer);
            free(ref_buffer);
        } else {
            if (src_fp) fclose(src_fp);
            if (ref_fp) fclose(ref_fp);
        }
    }

    if (error_count == 0) { 
        printf("  OK\n"); 
    } else {
		printf("  FAILURE: %d errors...\n", (unsigned int)error_count);
    }

    return (error_count == 0);  // returns true if all pixels pass
}


void CheckRender::bindReadback() {
    if (!m_bQAReadback) {
        printf("CheckRender::bindReadback() uninitialized!\n");
        return;
    }
    if (m_bUsePBO) {
		glBindBufferARB(GL_PIXEL_PACK_BUFFER_ARB, m_pboReadback);	// Bind the PBO
    }
}

void CheckRender::unbindReadback() {
    if (!m_bQAReadback) {
        printf("CheckRender::unbindReadback() uninitialized!\n");
        return;
    }
    if (m_bUsePBO) {
		glBindBufferARB(GL_PIXEL_PACK_BUFFER_ARB, 0);	// Release the bind on the PBO
    }
}


// CheckBackBuffer::Class for rendering and verifying results from the CheckBackBuffer
CheckBackBuffer::CheckBackBuffer(unsigned int width, unsigned int height, unsigned int Bpp, bool bUseOpenGL) : 
			CheckRender(width, height, Bpp, false, false, bUseOpenGL)
{
}

CheckBackBuffer::~CheckBackBuffer()
{
}

bool CheckBackBuffer::checkStatus(const char *zfile, int line, bool silent)
{
    GLenum nErrorCode = glGetError();

    if (nErrorCode != GL_NO_ERROR)
    {
        if (!silent)
           printf("Assertion failed(%s,%d): %s\n", zfile, line, gluErrorString(nErrorCode));
    }
    return true;
}

//////////////////////////////////////////////////////////////////////
//  readback
//
//////////////////////////////////////////////////////////////////////
bool CheckBackBuffer::readback( GLuint width, GLuint height )
{
    bool ret = false;

    if (m_bUsePBO) 
    {
        // binds the PBO for readback
        bindReadback();

        // Initiate the readback BLT from BackBuffer->PBO->membuf
	    glReadPixels(0, 0, width, height, getPixelFormat(),      GL_UNSIGNED_BYTE, BUFFER_OFFSET(0));
        ret = checkStatus(__FILE__, __LINE__, true);
        if (!ret) printf("CheckBackBuffer::glReadPixels() checkStatus = %d\n", ret);

	    // map - unmap simulates readback without the copy
	    void *ioMem = glMapBufferARB(GL_PIXEL_PACK_BUFFER_ARB, GL_READ_ONLY_ARB);
        memcpy(m_pImageData,    ioMem, width*height*m_Bpp);

		glUnmapBufferARB(GL_PIXEL_PACK_BUFFER_ARB);

        // release the PBO
        unbindReadback();
    } else {
        // reading direct from the backbuffer
        glReadBuffer(GL_FRONT);
        glReadPixels(0, 0, width, height, getPixelFormat(), GL_UNSIGNED_BYTE, m_pImageData);
    }

    return ret; 
}

//////////////////////////////////////////////////////////////////////
//  readback 
//
//   Code to handle reading back of the FBO data (but with a specified FBO pointer)
//
//////////////////////////////////////////////////////////////////////
bool CheckBackBuffer::readback( GLuint width, GLuint height, GLuint bufObject )
{
    bool ret = false;

    if (m_bUseFBO) {
        if (m_bUsePBO) 
        {
            printf("CheckBackBuffer::readback() FBO->PBO->m_pImageData\n");
            // binds the PBO for readback
            bindReadback();

            // bind FBO buffer (we want to transfer FBO -> PBO)
            glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, bufObject );

            // Now initiate the readback to PBO
	        glReadPixels(0, 0, width, height, getPixelFormat(),      GL_UNSIGNED_BYTE, BUFFER_OFFSET(0));
            ret = checkStatus(__FILE__, __LINE__, true);
            if (!ret) printf("CheckBackBuffer::readback() FBO->PBO checkStatus = %d\n", ret);

	        // map - unmap simulates readback without the copy
	        void *ioMem = glMapBufferARB(GL_PIXEL_PACK_BUFFER_ARB, GL_READ_ONLY_ARB);
            memcpy(m_pImageData,    ioMem, width*height*m_Bpp);

		    glUnmapBufferARB(GL_PIXEL_PACK_BUFFER_ARB);

            // release the FBO
		    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0); 

            // release the PBO
            unbindReadback();
        } else {
            printf("CheckBackBuffer::readback() FBO->m_pImageData\n");
            // Reading direct to FBO using glReadPixels
            glBindFramebufferEXT( GL_FRAMEBUFFER_EXT, bufObject );
            ret = checkStatus(__FILE__, __LINE__, true);
            if (!ret) printf("CheckBackBuffer::readback::glBindFramebufferEXT() fbo=%d checkStatus = %d\n", bufObject, ret);

            glReadBuffer(static_cast<GLenum>(GL_COLOR_ATTACHMENT0_EXT));
            ret &= checkStatus(__FILE__, __LINE__, true);
            if (!ret) printf("CheckBackBuffer::readback::glReadBuffer() fbo=%d checkStatus = %d\n", bufObject, ret);

            glReadPixels(0, 0, width, height, getPixelFormat(), GL_UNSIGNED_BYTE, m_pImageData);

            glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
        }
    } else {
        
        printf("CheckBackBuffer::readback() PBO->m_pImageData\n");
        // read from bufObject (PBO) to system memorys image
		glBindBufferARB(GL_PIXEL_PACK_BUFFER_ARB, bufObject);	// Bind the PBO

        // map - unmap simulates readback without the copy
        void *ioMem = glMapBufferARB(GL_PIXEL_PACK_BUFFER_ARB, GL_READ_ONLY_ARB);

        // allocate a buffer so we can flip the image
        unsigned char * temp_buf = (unsigned char *)malloc(width*height*m_Bpp);
        memcpy( temp_buf, ioMem, width*height*m_Bpp );

        // let's flip the image as we copy
        for (unsigned int y = 0; y < height; y++) {
            memcpy( (void *)&(m_pImageData[(height-y)*width*m_Bpp]), (void *)&(temp_buf[y*width*m_Bpp]), width*m_Bpp);
        }
        free(temp_buf);

	    glUnmapBufferARB(GL_PIXEL_PACK_BUFFER_ARB);

        // read from bufObject (PBO) to system memory image
		glBindBufferARB(GL_PIXEL_PACK_BUFFER_ARB, 0);	// unBind the PBO
    }
	return CHECK_FBO;
}

//////////////////////////////////////////////////////////////////////
//  readback 
//
//   Code to handle reading back of a system memory buffer
//
//////////////////////////////////////////////////////////////////////
bool CheckBackBuffer::readback( GLuint width, GLuint height, unsigned char *memBuf )
{
    // let's flip the image as we copy
    for (unsigned int y = 0; y < height; y++) {
        memcpy( (void *)&(m_pImageData[(height-y)*width*m_Bpp]), (void *)&(memBuf[y*width*m_Bpp]), width*m_Bpp);
    }
	return true;
}


// End of class for rendering to CheckBackBuffer

#define NEW_FBO_CODE 1

// Class for CFrameBufferObject (container for FBO rendering)
CFrameBufferObject::CFrameBufferObject(unsigned int width, unsigned int height, 
                                       unsigned int Bpp, bool bUseFloat, GLenum eTarget) :
			m_bUseFloat(bUseFloat), 
			m_eGLTarget(eTarget)
{
#if NEW_FBO_CODE
    glGenFramebuffersEXT(1, &m_fboData.fb);

    m_fboData.colorTex = createTexture(m_eGLTarget, width, height, 
                                (bUseFloat ? GL_RGBA32F_ARB : GL_RGBA8), GL_RGBA);

    m_fboData.depthTex = createTexture( m_eGLTarget, width, height, 
                                        GL_DEPTH_COMPONENT24, GL_DEPTH_COMPONENT);

    attachTexture( m_eGLTarget, m_fboData.colorTex,   GL_COLOR_ATTACHMENT0_EXT );
    attachTexture( m_eGLTarget, m_fboData.depthTex,   GL_DEPTH_ATTACHMENT_EXT  );

//    bool ret = checkStatus(__FILE__, __LINE__, true);
#else
    // this is th eoriginal FBO path
	initialize(width, height, m_fboConfig, m_fboData);
    CHECK_FBO;
#endif
}


CFrameBufferObject::CFrameBufferObject(unsigned int width, unsigned int height, 
                                       unsigned int Bpp, fboData &data, fboConfig &config, 
                                       bool bUseFloat) :
                  m_fboData(data), 
                  m_fboConfig(config),
                  m_bUseFloat(bUseFloat),
                  m_eGLTarget(GL_TEXTURE_2D)
{
#if !NEW_FBO_CODE
    initialize(width, height, m_fboConfig, m_fboData);

    CHECK_FBO;
#endif
}

CFrameBufferObject::CFrameBufferObject(unsigned int width, unsigned int height, 
                                       unsigned int Bpp, fboData &data, fboConfig &config, 
                                       bool bUseFloat, GLenum eTarget) :
                  m_fboData(data), 
                  m_fboConfig(config),
                  m_bUseFloat(bUseFloat),
                  m_eGLTarget(eTarget)
{
#if !NEW_FBO_CODE
	initialize(width, height, m_fboConfig, m_fboData);

    CHECK_FBO;
#endif
}

CFrameBufferObject::~CFrameBufferObject()
{
//   freeResources();
}

void 
CFrameBufferObject::freeResources()
{
    if (m_fboData.fb)           glDeleteFramebuffersEXT( 1, &m_fboData.fb);
    if (m_fboData.resolveFB)    glDeleteFramebuffersEXT( 1, &m_fboData.resolveFB);
    if (m_fboData.colorRB)      glDeleteRenderbuffersEXT( 1, &m_fboData.colorRB);
    if (m_fboData.depthRB)      glDeleteRenderbuffersEXT( 1, &m_fboData.depthRB);
    if (m_fboData.colorTex)     glDeleteTextures( 1, &m_fboData.colorTex);
    if (m_fboData.depthTex)     glDeleteTextures( 1, &m_fboData.depthTex);

    glDeleteProgramsARB(1, &m_textureProgram);
    glDeleteProgramsARB(1, &m_overlayProgram);
}


// create an OpenGL texture
GLuint
CFrameBufferObject::createTexture(GLenum target, int w, int h, GLint internalformat, GLenum format)
{
    GLuint texid;
    glGenTextures(1, &texid);
    glBindTexture(target, texid);

    glTexParameteri(target, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(target, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(target, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(target, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glTexImage2D(target, 0, internalformat, w, h, 0, format, GL_FLOAT, 0);
    return texid;
}


// attach the texture to a FBO
void
CFrameBufferObject::attachTexture( GLenum texTarget, GLuint texId, 
                                   GLenum attachment, int mipLevel, int zSlice )
{
    bindRenderPath();
    switch (texTarget) {
        case GL_TEXTURE_1D:
            glFramebufferTexture1DEXT( GL_FRAMEBUFFER_EXT, attachment,
                                       GL_TEXTURE_1D, texId, mipLevel );
            break;
        case GL_TEXTURE_3D:
            glFramebufferTexture3DEXT( GL_FRAMEBUFFER_EXT, attachment,
                                       GL_TEXTURE_3D, texId, mipLevel, zSlice );
           break;
        default:
            // Default is GL_TEXTURE_2D, GL_TEXTURE_RECTANGLE_ARB, or cube faces
            glFramebufferTexture2DEXT( GL_FRAMEBUFFER_EXT, attachment,
                                       texTarget, texId, mipLevel );
            break;
    }
    unbindRenderPath();
}

//////////////////////////////////////////////////////////////////////
//  initialize
//
//   helper function to setup the FBO
//////////////////////////////////////////////////////////////////////
bool CFrameBufferObject::initialize(unsigned int width, unsigned int height, fboConfig & rConfigFBO, fboData & rActiveFBO)
{
    //Framebuffer config options
    vector<bufferConfig> colorConfigs;
    vector<bufferConfig> depthConfigs;
    bufferConfig temp;

    //add default color configs
	temp.name   = (m_bUseFloat ? "RGBA32F" : "RGBA8");
	temp.bits   = (m_bUseFloat ? 32 : 8);
	temp.format = (m_bUseFloat ? GL_RGBA32F_ARB : GL_RGBA8);
	colorConfigs.push_back( temp);

    //add default depth configs
    temp.name = "D24";
    temp.bits = 24;
    temp.format = GL_DEPTH_COMPONENT24;
    depthConfigs.push_back( temp );

    // If the FBO can be created, add it to the list of available configs, and make a menu entry
    string root = colorConfigs[0].name + " " + depthConfigs[0].name;

    rConfigFBO.colorFormat	= colorConfigs[0].format;
    rConfigFBO.depthFormat	= depthConfigs[0].format;
    rConfigFBO.redbits		= colorConfigs[0].bits;
    rConfigFBO.depthBits	= depthConfigs[0].bits;

    //single sample
    rConfigFBO.name				= root;
    rConfigFBO.coverageSamples	= 0;
    rConfigFBO.depthSamples		= 0;

    create( width, height, rConfigFBO, rActiveFBO );

    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);

    if (m_bUseFloat) {
	    // load fragment programs
	    const char* strTextureProgram2D = 
		    "!!ARBfp1.0\n"
		    "TEX result.color, fragment.texcoord[0], texture[0], 2D;\n"
		    "END\n";

        m_textureProgram = nv::CompileASMShader( GL_FRAGMENT_PROGRAM_ARB, strTextureProgram2D );

	    const char* strOverlayProgram = 
		    "!!ARBfp1.0\n"
		    "TEMP t;\n"
		    "TEX t, fragment.texcoord[0], texture[0], 2D;\n"
		    "MOV result.color, t;\n"
		    "END\n";
	    
        m_overlayProgram = nv::CompileASMShader( GL_FRAGMENT_PROGRAM_ARB, strOverlayProgram );
    }

    return CHECK_FBO;
}

//////////////////////////////////////////////////////////////////////
//  renderQuad (sourced from FBO, assume the binding has already taken place)
//
//////////////////////////////////////////////////////////////////////
void CFrameBufferObject::renderQuad(int width, int height, GLenum eTarget)
{
#if 1
    // Bind the FBO texture for the display path
    glBindTexture(eTarget, m_fboData.colorTex);

    glGenerateMipmapEXT( GL_TEXTURE_2D );
    glBindTexture(eTarget, 0);

    // now render to the full screen using this texture
	glClearColor(0.2, 0.2, 0.2, 0.0);
	glClear(GL_COLOR_BUFFER_BIT);

	glBindProgramARB(GL_FRAGMENT_PROGRAM_ARB, m_textureProgram );
	glEnable(GL_FRAGMENT_PROGRAM_ARB);
	glDisable(GL_DEPTH_TEST);

	glBegin(GL_QUADS);
	{
		glVertex2f(0, 0); glTexCoord2f(0, 0);
		glVertex2f(0, 1); glTexCoord2f(1, 0);
		glVertex2f(1, 1); glTexCoord2f(1, 1);
		glVertex2f(1, 0); glTexCoord2f(0, 1);
	}
	glEnd();

    // Release the FBO texture (finished rendering)
    glBindTexture(eTarget, 0);

#else
    // Bind the FBO texture for the display path
    glBindTexture(eTarget, m_fboData.colorTex);

    // render a screen sized quad
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_LIGHTING);
    glEnable(eTarget);
    glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);

    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glOrtho(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0);

    glMatrixMode( GL_MODELVIEW);
    glLoadIdentity();

    glViewport(0, 0, width, height);

	if (m_bUseFloat) {
		// fragment program is required to display floating point texture
		glBindProgramARB(GL_FRAGMENT_PROGRAM_ARB, m_textureProgram );
		glEnable(GL_FRAGMENT_PROGRAM_ARB);
		glDisable(GL_DEPTH_TEST);
	}

    glBegin(GL_QUADS);
	if (eTarget == GL_TEXTURE_2D) {
		glTexCoord2f(0.0f , 0.0f  ); glVertex3f(-1.0f, -1.0f, 0.5f);
		glTexCoord2f(1.0f , 0.0f  ); glVertex3f( 1.0f, -1.0f, 0.5f);
		glTexCoord2f(1.0f , 1.0f  ); glVertex3f( 1.0f,  1.0f, 0.5f);
		glTexCoord2f(0.0f , 1.0f  ); glVertex3f(-1.0f,  1.0f, 0.5f);
	} else {
		glTexCoord2f(0.0f , 0.0f  ); glVertex3f(-1.0f, -1.0f, 0.5f);
		glTexCoord2f(width, 0.0f  ); glVertex3f( 1.0f, -1.0f, 0.5f);
		glTexCoord2f(width, height); glVertex3f( 1.0f,  1.0f, 0.5f);
		glTexCoord2f(0.0f , height); glVertex3f(-1.0f,  1.0f, 0.5f);
	}

    glEnd();

    glMatrixMode(GL_PROJECTION);
    glPopMatrix();

    glDisable(eTarget);

    // Release the FBO texture (finished rendering)
    glBindTexture(eTarget, 0);
#endif

}

//////////////////////////////////////////////////////////////////////
//  checkStatus of FBO
//
//   Check the framebuffer status to ensure it is a supported
//    config.
//////////////////////////////////////////////////////////////////////
bool CFrameBufferObject::checkStatus(const char *zfile, int line, bool silent)
{
    GLenum status;
    status = (GLenum) glCheckFramebufferStatusEXT(GL_FRAMEBUFFER_EXT);
	if (status != GL_FRAMEBUFFER_COMPLETE_EXT) {
		printf("<%s : %d> - ", zfile, line);
	}

	switch(status) {
        case GL_FRAMEBUFFER_COMPLETE_EXT:
            break;
        case GL_FRAMEBUFFER_UNSUPPORTED_EXT:
			if (!silent) printf("Unsupported framebuffer format\n");
            return false;
        case GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT_EXT:
            if (!silent) printf("Framebuffer incomplete, missing attachment\n");
            return false;
        case GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT_EXT:
            if (!silent) printf("Framebuffer incomplete, duplicate attachment\n");
            return false;
        case GL_FRAMEBUFFER_INCOMPLETE_DIMENSIONS_EXT:
            if (!silent) printf("Framebuffer incomplete, attached images must have same dimensions\n");
            return false;
        case GL_FRAMEBUFFER_INCOMPLETE_FORMATS_EXT:
            if (!silent) printf("Framebuffer incomplete, attached images must have same format\n");
            return false;
        case GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER_EXT:
            if (!silent) printf("Framebuffer incomplete, missing draw buffer\n");
            return false;
        case GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER_EXT:
            if (!silent) printf("Framebuffer incomplete, missing read buffer\n");
            return false;
        default:
            assert(0);
            return false;
    }
    return true;
}

//////////////////////////////////////////////////////////////////////
//  create
//
//   Creates a FBO based on the given config, if the created FBO
// fails to validate, or the bits do not match the prescribed size,
// this it returns false. If the number of samples or coverage
// samples exceed the requested number, then the number of samples
// in the config is changed to match the number recieved.
//
//////////////////////////////////////////////////////////////////////
bool CFrameBufferObject::create( GLuint width, GLuint height, fboConfig &config, fboData &data) 
{
    bool multisample = config.depthSamples > 0;
    bool csaa = config.coverageSamples > config.depthSamples;
    bool ret = true;
    GLint query;

	printf("\nCreating FBO <%s> Float:%s\n", config.name.c_str(), (m_bUseFloat ? "Y":"N") );

    glGenFramebuffersEXT(1, &data.fb);
    glGenTextures(1, &data.colorTex);

    // init texture
    glBindTexture( m_eGLTarget, data.colorTex);
    glTexImage2D ( m_eGLTarget, 0, config.colorFormat, 
					width, height, 0, GL_RGBA, 
				   (m_bUseFloat ? GL_FLOAT : GL_UNSIGNED_BYTE), 
					NULL 
				  );

    glGenerateMipmapEXT( m_eGLTarget );
    
    glTexParameterf( m_eGLTarget, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameterf( m_eGLTarget, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameterf( m_eGLTarget, GL_TEXTURE_MIN_FILTER, GL_NEAREST); // GL_LINEAR_MIPMAP_LINEAR);
    glTexParameterf( m_eGLTarget, GL_TEXTURE_MAG_FILTER, GL_NEAREST); // GL_LINEAR);

    //
    // Handle multisample FBO's first
    //
    if (multisample) {
        if (csaa) {
			ret &= createCSAA(width, height, &config, &data);
        }
        else {
			ret &= createMSAA(width, height, &config, &data);
        }
       
        // attach the depth buffer
        glFramebufferRenderbufferEXT(GL_FRAMEBUFFER_EXT, GL_DEPTH_ATTACHMENT_EXT, GL_RENDERBUFFER_EXT, data.depthRB);
        ret &= checkStatus(__FILE__, __LINE__, true);
    } 
    else // case if not multisampled
	{ 
        glGenTextures( 1, &data.depthTex );
        data.depthRB = 0;
        data.colorRB = 0;
        data.resolveFB = 0;

        //non-multisample, so bind things directly to the FBO
        glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, data.fb); 
        glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, m_eGLTarget, data.colorTex, 0);

        glBindTexture(	m_eGLTarget, data.depthTex );
        glTexImage2D(	m_eGLTarget, 0, config.depthFormat, 
						width, height, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
    
        glTexParameterf( m_eGLTarget, GL_TEXTURE_MIN_FILTER, GL_NEAREST); // GL_LINEAR);
        glTexParameterf( m_eGLTarget, GL_TEXTURE_MAG_FILTER, GL_NEAREST); // GL_LINEAR);
        glTexParameterf( m_eGLTarget, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameterf( m_eGLTarget, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameterf( m_eGLTarget, GL_DEPTH_TEXTURE_MODE, GL_LUMINANCE);

        glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_DEPTH_ATTACHMENT_EXT, m_eGLTarget, data.depthTex, 0);

        ret &= checkStatus(__FILE__, __LINE__, true);
    }
    
    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, data.fb);
    glGetIntegerv( GL_RED_BITS, &query);
    if ( query != config.redbits) {
        ret = false;
    }

    glGetIntegerv( GL_DEPTH_BITS, &query);
    if ( query != config.depthBits) {
        ret = false;
    }

    if (multisample) {
        glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, data.resolveFB);
        glGetIntegerv( GL_RED_BITS, &query);
        if ( query != config.redbits) {
            ret = false;
        }
    }
    
    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);

    ret &= checkStatus(__FILE__, __LINE__, true);

    return ret;
}

bool CFrameBufferObject::createCSAA( GLuint width, GLuint height, fboConfig *p_config, fboData *p_data )
{
    GLint query;
    bool ret = false;

    // Step #1
    {
        glGenRenderbuffersEXT( 1, &p_data->depthRB);
        glGenRenderbuffersEXT( 1, &p_data->colorRB);
        glGenFramebuffersEXT( 1, &p_data->resolveFB);
        p_data->depthTex = 0; //no resolve of depth buffer for now
        
        //multisample, so we need to resolve from the FBO, bind the texture to the resolve FBO
        glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, p_data->resolveFB); 

        glFramebufferTexture2DEXT(	GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, 
									m_eGLTarget, p_data->colorTex, 0);

        ret &= checkStatus(__FILE__, __LINE__, true);

        //now handle the rendering FBO
        glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, p_data->fb);

        // initialize color renderbuffer
        glBindRenderbufferEXT(GL_RENDERBUFFER_EXT, p_data->colorRB);
    }

	// Step #2
	{
		glRenderbufferStorageMultisampleCoverageNV( GL_RENDERBUFFER_EXT, 
													p_config->coverageSamples, 
													p_config->depthSamples, 
													p_config->colorFormat,
													width, height);

		glGetRenderbufferParameterivEXT( GL_RENDERBUFFER_EXT, GL_RENDERBUFFER_COVERAGE_SAMPLES_NV, &query);

		if ( query < p_config->coverageSamples) {
		   ret = false;
		}
		else if ( query > p_config->coverageSamples) {
			// report back the actual number
			p_config->coverageSamples = query;
		}

		glGetRenderbufferParameterivEXT( GL_RENDERBUFFER_EXT, GL_RENDERBUFFER_COLOR_SAMPLES_NV, &query);

		if ( query < p_config->depthSamples) {
		   ret = false;
		}
		else if ( query > p_config->depthSamples) {
			// report back the actual number
			p_config->depthSamples = query;
		}
	}

	// Step #3
	{
        // attach the multisampled color buffer
        glFramebufferRenderbufferEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_RENDERBUFFER_EXT, p_data->colorRB);

        ret &= checkStatus(__FILE__, __LINE__, true);

        // bind the multisampled depth buffer
        glBindRenderbufferEXT(GL_RENDERBUFFER_EXT, p_data->depthRB);
	}

	// Step #4 create the multisampled depth buffer (with coverage sampling)
	{
        // create a coverage sampled MSAA depth buffer
        glRenderbufferStorageMultisampleCoverageNV( GL_RENDERBUFFER_EXT, 
													p_config->coverageSamples, 
													p_config->depthSamples, 
													p_config->depthFormat,
                                                    width, height);

        // check the number of coverage samples
        glGetRenderbufferParameterivEXT( GL_RENDERBUFFER_EXT, GL_RENDERBUFFER_COVERAGE_SAMPLES_NV, &query);

        if ( query < p_config->coverageSamples) {
            ret = false;
        }
        else if ( query > p_config->coverageSamples) {
            // set the coverage samples value to return the actual value
            p_config->coverageSamples = query;
        } 

        // cehck the number of stored color samples (same as depth samples)
        glGetRenderbufferParameterivEXT( GL_RENDERBUFFER_EXT, GL_RENDERBUFFER_COLOR_SAMPLES_NV, &query);

        if ( query < p_config->depthSamples) {
            ret = false;
        }
        else if ( query > p_config->depthSamples) {
            // set the depth samples value to return the actual value
            p_config->depthSamples = query;
        }
	}
	return ret;
}

bool CFrameBufferObject::createMSAA( GLuint width, GLuint height, fboConfig *p_config, fboData *p_data )
{
	GLint query;
	bool ret = false;

	// Step #1
	{
        glGenRenderbuffersEXT( 1, &p_data->depthRB   );
        glGenRenderbuffersEXT( 1, &p_data->colorRB   );
        glGenFramebuffersEXT ( 1, &p_data->resolveFB );
        p_data->depthTex = 0; //no resolve of depth buffer for now
        
        //multisample, so we need to resolve from the FBO, bind the texture to the resolve FBO
        glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, p_data->resolveFB); 

        glFramebufferTexture2DEXT(	GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, 
									m_eGLTarget, p_data->colorTex, 0);
        ret &= checkStatus(__FILE__, __LINE__, true);

        //now handle the rendering FBO
        glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, p_data->fb);

        // initialize color renderbuffer
        glBindRenderbufferEXT(GL_RENDERBUFFER_EXT, p_data->colorRB);
	}

	// Step #2
	{
		// create a regular MSAA color buffer
		glRenderbufferStorageMultisampleEXT( GL_RENDERBUFFER_EXT, 
											p_config->depthSamples, 
											p_config->colorFormat, 
											width, height);
		// check the number of samples
		glGetRenderbufferParameterivEXT( GL_RENDERBUFFER_EXT, GL_RENDERBUFFER_SAMPLES_EXT, &query);

		if ( query < p_config->depthSamples) {
			ret = false;
		}
		else if ( query > p_config->depthSamples) {
			p_config->depthSamples = query;
		}
	}

	// Step #3
	{
        // attach the multisampled color buffer
        glFramebufferRenderbufferEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_RENDERBUFFER_EXT, p_data->colorRB);
        ret &= checkStatus(__FILE__, __LINE__, true);

        // bind the multisampled depth buffer
        glBindRenderbufferEXT(GL_RENDERBUFFER_EXT, p_data->depthRB);
	}

	// Step #4 - create the multisampled depth buffer (without coverage sampling)
	{
        // create a regular (not coverage sampled) MSAA depth buffer
        glRenderbufferStorageMultisampleEXT(	GL_RENDERBUFFER_EXT, 
												p_config->depthSamples, 
												p_config->depthFormat, 
												width, height);

        // check the number of depth samples
        glGetRenderbufferParameterivEXT( GL_RENDERBUFFER_EXT, GL_RENDERBUFFER_SAMPLES_EXT, &query);

        if ( query < p_config->depthSamples) {
            ret = false;
        }
        else if ( query < p_config->depthSamples) {
            p_config->depthSamples = query;
        }
	}

	return ret;
}

// End of class for managing CFrameBufferOBject




// Class for verify and rendering to/from FrameBufferObjects
CheckFBO::CheckFBO(unsigned int width, unsigned int height, unsigned int Bpp) :
			CheckRender(width, height, Bpp, false, false, true),
			m_pFrameBufferObject(NULL)
{
}

CheckFBO::CheckFBO(unsigned int width, unsigned int height, unsigned int Bpp, CFrameBufferObject *pFrameBufferObject) :
			CheckRender(width, height, Bpp, false, true, true),
			m_pFrameBufferObject(pFrameBufferObject)
{
}

CheckFBO::~CheckFBO() 
{
}

//////////////////////////////////////////////////////////////////////
//  checkStatus of FBO
//
//   Check the framebuffer status to ensure it is a supported
//    config.
//////////////////////////////////////////////////////////////////////
bool CheckFBO::checkStatus(const char *zfile, int line, bool silent)
{
    GLenum status;
    status = (GLenum) glCheckFramebufferStatusEXT(GL_FRAMEBUFFER_EXT);
	if (status != GL_FRAMEBUFFER_COMPLETE_EXT) {
		printf("<%s : %d> - ", zfile, line);
	}

	switch(status) {
        case GL_FRAMEBUFFER_COMPLETE_EXT:
            break;
        case GL_FRAMEBUFFER_UNSUPPORTED_EXT:
			if (!silent) printf("Unsupported framebuffer format\n");
            return false;
        case GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT_EXT:
            if (!silent) printf("Framebuffer incomplete, missing attachment\n");
            return false;
        case GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT_EXT:
            if (!silent) printf("Framebuffer incomplete, duplicate attachment\n");
            return false;
        case GL_FRAMEBUFFER_INCOMPLETE_DIMENSIONS_EXT:
            if (!silent) printf("Framebuffer incomplete, attached images must have same dimensions\n");
            return false;
        case GL_FRAMEBUFFER_INCOMPLETE_FORMATS_EXT:
            if (!silent) printf("Framebuffer incomplete, attached images must have same format\n");
            return false;
        case GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER_EXT:
            if (!silent) printf("Framebuffer incomplete, missing draw buffer\n");
            return false;
        case GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER_EXT:
            if (!silent) printf("Framebuffer incomplete, missing read buffer\n");
            return false;
        default:
            assert(0);
            return false;
    }
    return true;
}

//////////////////////////////////////////////////////////////////////
//  readback 
//
//   Code to handle reading back of the FBO data
//
//////////////////////////////////////////////////////////////////////
bool CheckFBO::readback( GLuint width, GLuint height )
{
    bool ret = false;

    if (m_bUsePBO) {
        // binds the PBO for readback
        bindReadback();

        // bind FBO buffer (we want to transfer FBO -> PBO)
        glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, m_pFrameBufferObject->getFbo());

        // Now initiate the readback to PBO
	    glReadPixels(0, 0, width, height, getPixelFormat(),      GL_UNSIGNED_BYTE, BUFFER_OFFSET(0));
        ret = checkStatus(__FILE__, __LINE__, true);
        if (!ret) printf("CheckFBO::readback() FBO->PBO checkStatus = %d\n", ret);

	    // map - unmap simulates readback without the copy
	    void *ioMem = glMapBufferARB(GL_PIXEL_PACK_BUFFER_ARB, GL_READ_ONLY_ARB);
        memcpy(m_pImageData,    ioMem, width*height*m_Bpp);

		glUnmapBufferARB(GL_PIXEL_PACK_BUFFER_ARB);

        // release the FBO
		glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0); 

        // release the PBO
        unbindReadback();
    } else {
        // Reading back from FBO using glReadPixels
        glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, m_pFrameBufferObject->getFbo());
        ret = checkStatus(__FILE__, __LINE__, true);
        if (!ret) printf("CheckFBO::readback::glBindFramebufferEXT() checkStatus = %d\n", ret);

        glReadBuffer(static_cast<GLenum>(GL_COLOR_ATTACHMENT0_EXT));
        ret &= checkStatus(__FILE__, __LINE__, true);
        if (!ret) printf("CheckFBO::readback::glReadBuffer() checkStatus = %d\n", ret);

	    glReadPixels(0, 0, width, height, getPixelFormat(), GL_UNSIGNED_BYTE, m_pImageData);

	    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
    }

    return CHECK_FBO;
}

//////////////////////////////////////////////////////////////////////
//  readback 
//
//   Code to handle reading back of the FBO data (but with a specified FBO pointer)
//
//////////////////////////////////////////////////////////////////////
bool CheckFBO::readback( GLuint width, GLuint height, GLuint bufObject )
{
    bool ret = false;

    if (m_bUseFBO) {
        if (m_bUsePBO) 
        {
            printf("CheckFBO::readback() FBO->PBO->m_pImageData\n");
            // binds the PBO for readback
            bindReadback();

            // bind FBO buffer (we want to transfer FBO -> PBO)
            glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, bufObject );

            // Now initiate the readback to PBO
	        glReadPixels(0, 0, width, height, getPixelFormat(),      GL_UNSIGNED_BYTE, BUFFER_OFFSET(0));
            ret = checkStatus(__FILE__, __LINE__, true);
            if (!ret) printf("CheckFBO::readback() FBO->PBO checkStatus = %d\n", ret);

	        // map - unmap simulates readback without the copy
	        void *ioMem = glMapBufferARB(GL_PIXEL_PACK_BUFFER_ARB, GL_READ_ONLY_ARB);
            memcpy(m_pImageData,    ioMem, width*height*m_Bpp);

		    glUnmapBufferARB(GL_PIXEL_PACK_BUFFER_ARB);

            // release the FBO
		    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0); 

            // release the PBO
            unbindReadback();
        } else {
            printf("CheckFBO::readback() FBO->m_pImageData\n");
            // Reading direct to FBO using glReadPixels
            glBindFramebufferEXT( GL_FRAMEBUFFER_EXT, bufObject );
            ret = checkStatus(__FILE__, __LINE__, true);
            if (!ret) printf("CheckFBO::readback::glBindFramebufferEXT() fbo=%d checkStatus = %d\n", (int)bufObject, (int)ret);

            glReadBuffer(static_cast<GLenum>(GL_COLOR_ATTACHMENT0_EXT));
            ret &= checkStatus(__FILE__, __LINE__, true);
            if (!ret) printf("CheckFBO::readback::glReadBuffer() fbo=%d checkStatus = %d\n", (int)bufObject, (int)ret);

            glReadPixels(0, 0, width, height, getPixelFormat(), GL_UNSIGNED_BYTE, m_pImageData);

            glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
        }
    } else {
        printf("CheckFBO::readback() PBO->m_pImageData\n");
        // read from bufObject (PBO) to system memorys image
		glBindBufferARB(GL_PIXEL_PACK_BUFFER_ARB, bufObject);	// Bind the PBO

        // map - unmap simulates readback without the copy
        void *ioMem = glMapBufferARB(GL_PIXEL_PACK_BUFFER_ARB, GL_READ_ONLY_ARB);
        memcpy(m_pImageData,    ioMem, width*height*m_Bpp);

	    glUnmapBufferARB(GL_PIXEL_PACK_BUFFER_ARB);

        // read from bufObject (PBO) to system memory image
		glBindBufferARB(GL_PIXEL_PACK_BUFFER_ARB, 0);	// unBind the PBO
    }
	return CHECK_FBO;
}

//////////////////////////////////////////////////////////////////////
//  readback 
//
//   Code to handle reading back of a system memory buffer
//
//////////////////////////////////////////////////////////////////////
bool CheckFBO::readback( GLuint width, GLuint height, unsigned char *memBuf )
{
    // let's flip the image as we copy
    for (unsigned int y = 0; y < height; y++) {
        memcpy( (void *)&(m_pImageData[(height-y)*width*m_Bpp]), (void *)&(memBuf[y*width*m_Bpp]), width*m_Bpp);
    }
	return true;
}
