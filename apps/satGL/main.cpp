// -------------------------------------------------------------
// cuDPP -- CUDA Data Parallel Primitives library
// -------------------------------------------------------------
// $Revision$
// $Date$
// ------------------------------------------------------------- 
// This source code is distributed under the terms of license.txt in
// the root directory of this source distribution.
// ------------------------------------------------------------- 

/* 
    This example demonstrates the use of CUDA/OpenGL interoperability
    to post-process an image of a 3D scene generated in OpenGL.

    The basic steps are:
    1 - render the scene to the framebuffer
    2 - copy the image to a PBO (pixel buffer object)
    3 - map this PBO so that its memory is accessible from CUDA
    4 - run CUDA to process the image, writing to memory mapped from a second PBO
    6 - copy from result PBO to a texture
    7 - display the texture

    Press space to toggle the CUDA processing on/off.
    Press 'a' to toggle animation.
    Press '+' and '-' to increment and decrement blur radius
*/

#if defined(_WIN32) || defined(WIN32)
#pragma warning(disable:4996)
#  define WINDOWS_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#endif

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>


// includes, GL
#include <GL/glew.h>

#if defined(__APPLE__) || defined(MACOSX)
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif

#include "cutil.h"
#include "cutil_gl_error.h"

////////////////////////////////////////////////////////////////////////////////
// constants / global variables
unsigned int window_width = 512;
unsigned int window_height = 512;
unsigned int image_width = 512;
unsigned int image_height = 512;

// pbo variables
GLuint pbo_source;
GLuint pbo_dest;

unsigned int vsSAT;
unsigned int psSAT;
unsigned int glprogramSAT;

unsigned int vsColorAndDistance;
unsigned int psColorAndDistance;
unsigned int glprogramColorAndDistance;

GLuint texWood;
unsigned int dlTeapot;

// (offscreen) render target
GLuint fbo;
//GLuint tex_fbo;
GLuint tex_distance;
GLuint tex_screen;
GLuint depth_rb;

GLenum singleChannelFloatIntFormat = GL_LUMINANCE32F_ARB;
GLenum singleChannelFloatFormat    = GL_LUMINANCE;

unsigned int timer;

float rotate[3] = {0, 135, 0};

bool enable_cuda = true;
bool animate = true;
int blur_radius = 8;

float zBias = 0.926f;
float zScale = 116.0f;

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
extern "C" void runTest( int argc, char** argv);
extern "C" void process( int pbo_in, int pbo_out, int width, int height, int radius);
extern "C" void pboRegister( unsigned int pbo );
extern "C" void pboUnregister( unsigned int pbo );
extern "C" void initialize(int width, int height);
extern "C" void finalize();

// GL functionality
CUTBoolean initGL();
void createPBO( GLuint* pbo, bool bUseFloat);
void deletePBO( GLuint* pbo);
void createFBO( GLuint* fbo, GLuint* tex, GLuint* depth_rb);
void deleteFBO( GLuint* fbo, GLuint* tex);
void createTexture(GLuint* tex_name, unsigned int size_x, unsigned int size_y, 
                   bool bUseFloat, bool singleChannel);
void deleteTexture( GLuint* tex);

// rendering callbacks
void display();
void idle();
void keyboard( unsigned char key, int x, int y);
void reshape(int w, int h);
void mainMenu(int i);

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main( int argc, char** argv) 
{
    CUT_DEVICE_INIT(argc, argv);

    runTest( argc, argv);

    return EXIT_SUCCESS;
}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
void
runTest( int argc, char** argv) 
{
    // Create GL context
    glutInit( &argc, argv);
    if (argc > 2)
    {
        image_width = atoi(argv[1]);
        image_height = atoi(argv[2]);
        window_width = image_width;
        window_height = image_height;
    }
    glutInitDisplayMode( GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);
    glutInitWindowSize( window_width, window_height);
    glutCreateWindow( "CUDA OpenGL post-processing");

    // initialize GL
    if( CUTFalse == initGL()) {
        return;
    }

    // register callbacks
    glutDisplayFunc( display);
    glutKeyboardFunc( keyboard);
    glutReshapeFunc( reshape);
    glutIdleFunc( idle);

    // create menu
    glutCreateMenu(mainMenu);
    glutAddMenuEntry("Toggle CUDA processing [ ]", ' ');
    glutAddMenuEntry("Toggle animation [r]", 'r');
    glutAddMenuEntry("Increment blur radius [=]", '=');
    glutAddMenuEntry("Decrement blur radius [-]", '-');
    glutAddMenuEntry("Increment Z bias [a]", 'a');
    glutAddMenuEntry("Decrement Z bias [z]", '-');
    glutAddMenuEntry("Quit (esc)", '\033');
    glutAttachMenu(GLUT_RIGHT_BUTTON);

    // create pbo
    createPBO( &pbo_source, false);
    createPBO( &pbo_dest, true);

    // create fbo
    createFBO( &fbo, &tex_distance, &depth_rb);
    // create texture for blitting onto the screen
    createTexture( &tex_screen, image_width, image_height, true, false);
   

    CUT_CHECK_ERROR_GL();
    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);

    CUT_SAFE_CALL(cutCreateTimer(&timer));

    initialize(image_width, image_height);

    // start rendering mainloop
    glutMainLoop();

    finalize();
}

const char vsSATSrc[] = 
{   
    "void main()                                                            \n"
    "{                                                                      \n"
    "    gl_TexCoord[0].st = gl_MultiTexCoord0.ts;                          \n"
    "    gl_Position = ftransform();                                        \n"
    "}                                                                      \n"
};

const char psSATSrc[] =
{
    "uniform sampler2D sat;                                                 \n"
    "uniform sampler2D distance;                                            \n"
    "uniform float radius;                                                  \n"
    "uniform float resolution;                                              \n"
    "uniform vec2 zScaleBias;                                               \n"

    "vec4 blurSAT(vec2 coords, float radius)                                \n"
    "{                                                                      \n"
    "    float offset = radius / resolution;                                \n"
    "    vec4 color;                                                        \n"
    "    color = texture2D(sat, coords + vec2(offset, offset));             \n"
    "    color -= texture2D(sat, gl_TexCoord[0].st + vec2(offset, -offset));\n"
    "    color -= texture2D(sat, gl_TexCoord[0].st + vec2(-offset, offset));\n"
    "    color += texture2D(sat, gl_TexCoord[0].st + vec2(-offset, -offset));\n"
    "    return color / (4.0 * radius * radius);                            \n"
    "}                                                                      \n"
    
    "void main()                                                            \n"
    "{                                                                      \n"
    "    float z = texture2D(distance, gl_TexCoord[0].ts).x;                \n"
    "    float blur = 1.0 + floor(abs(z - zScaleBias.y) * zScaleBias.x);    \n"
    "    gl_FragColor = blurSAT(gl_TexCoord[0].st, blur);                   \n"
    "}                                                                      \n"
};

const char vsColorAndDistanceSrc[] = 
{   
    "void main()                                                            \n"
    "{                                                                      \n"
    "    gl_TexCoord[0].st = gl_MultiTexCoord0.ts;                          \n"
    "    gl_Position = ftransform();                                        \n"
    "    gl_TexCoord[1].st  = gl_Position.zw;                               \n"
    "}                                                                      \n"
};

const char psColorAndDistanceSrc[] =
{
    "void main()                                                            \n"
    "{                                                                      \n"
    "    gl_FragData[0].r = 0.5 + 0.5 * gl_TexCoord[1].s/gl_TexCoord[1].t;  \n"
    "}                                                                      \n"
};

////////////////////////////////////////////////////////////////////////////////
//! Initialize GL
////////////////////////////////////////////////////////////////////////////////
CUTBoolean
initGL() {

    // initialize necessary OpenGL extensions
    glewInit();
    if (! glewIsSupported(
        "GL_VERSION_2_0 " 
        "GL_ARB_pixel_buffer_object "
        "GL_EXT_framebuffer_object "
        "GL_ARB_texture_float ")) 
    {
        fprintf( stderr, "ERROR: Support for necessary OpenGL extensions missing.");
        fflush( stderr);
        return CUTFalse;
    }
    
    if (! glewIsSupported("GL_ARB_framebuffer_object"))
    {
        singleChannelFloatIntFormat = GL_RGBA32F_ARB;
        singleChannelFloatFormat    = GL_RGBA;
    }
    
    // world display program
    vsColorAndDistance = glCreateShader(GL_VERTEX_SHADER);
    psColorAndDistance = glCreateShader(GL_FRAGMENT_SHADER);

    const char* v = vsColorAndDistanceSrc;
    const char* p = psColorAndDistanceSrc;
    glShaderSource(vsColorAndDistance, 1, &v, 0);
    glShaderSource(psColorAndDistance, 1, &p, 0);
    
    glCompileShader(vsColorAndDistance);
    glCompileShader(psColorAndDistance);

    glprogramColorAndDistance = glCreateProgram();

    glAttachShader(glprogramColorAndDistance, vsColorAndDistance);
    glAttachShader(glprogramColorAndDistance, psColorAndDistance);

    glLinkProgram(glprogramColorAndDistance);

    CUT_CHECK_ERROR_GL();

    // blur program
    vsSAT = glCreateShader(GL_VERTEX_SHADER);
    psSAT = glCreateShader(GL_FRAGMENT_SHADER);

    v = vsSATSrc;
    p = psSATSrc;
    glShaderSource(vsSAT, 1, &v, 0);
    glShaderSource(psSAT, 1, &p, 0);
    
    glCompileShader(vsSAT);
    glCompileShader(psSAT);

    glprogramSAT = glCreateProgram();

    glAttachShader(glprogramSAT, vsSAT);
    glAttachShader(glprogramSAT, psSAT);

    glLinkProgram(glprogramSAT);

    CUT_CHECK_ERROR_GL();

    // default initialization
    glClearColor( 0, 0, 0, 1.0f);//0.5, 0.5, 0.5, 1.0);
    glDisable( GL_DEPTH_TEST);

    // viewport
    glViewport( 0, 0, window_width, window_height);

    // projection
    glMatrixMode( GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60.0, (GLfloat)window_width / (GLfloat) window_height, 0.5, 50.0);
    
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glPolygonMode( GL_FRONT_AND_BACK, GL_FILL);

    unsigned char* img = NULL;
    unsigned int w = 512, h = 512;
    char path[100];
    CUT_SAFE_CALL(cutFindFile(path, "apps", "cedfence.ppm"));
    CUT_SAFE_CALL(cutLoadPPMub( path, &img, &w, &h));

    glGenTextures(1, &texWood);
    glBindTexture(GL_TEXTURE_2D, texWood);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY_EXT, 8);
    gluBuild2DMipmaps(GL_TEXTURE_2D, 3, w, h, GL_RGB, GL_UNSIGNED_BYTE, img);

    //delete [] img;

    dlTeapot = glGenLists(1);
    glNewList(dlTeapot, GL_COMPILE);
    glPushAttrib(GL_ALL_ATTRIB_BITS);
    glutSolidTeapot(1.0f);
    glPopAttrib();
    glEndList();

    glEnable(GL_LIGHT0);
    float red[] = { 1.0, 0.1, 0.1, 1.0 };
    float white[] = { 1.0, 1.0, 1.0, 1.0 };
    glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, red);
    glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, white);
    glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 60.0);

    glEnable(GL_DEPTH_TEST);

    CUT_CHECK_ERROR_GL();



    return CUTTrue;
}

////////////////////////////////////////////////////////////////////////////////
//! Create PBO
////////////////////////////////////////////////////////////////////////////////
void
createPBO( GLuint* pbo, bool bUseFloat) 
{
    unsigned int size_tex_data;
    unsigned int num_texels;
    unsigned int num_values;

    // set up vertex data parameter
    num_texels = image_width * image_height;
    num_values = num_texels * 4;
    if (bUseFloat)
    {
        size_tex_data = sizeof(GLfloat) * num_values;
    }
    else
    {
        size_tex_data = sizeof(GLubyte) * num_values;
    }
    void *data = malloc(size_tex_data);

    // create buffer object
    glGenBuffers( 1, pbo);
    glBindBuffer( GL_ARRAY_BUFFER, *pbo);

    // buffer data
    glBufferData( GL_ARRAY_BUFFER, size_tex_data, data, GL_DYNAMIC_DRAW);
    free(data);

    glBindBuffer( GL_ARRAY_BUFFER, 0);

    // attach this Buffer Object to CUDA
    pboRegister(*pbo);

    CUT_CHECK_ERROR_GL();
}

////////////////////////////////////////////////////////////////////////////////
//! Delete PBO
////////////////////////////////////////////////////////////////////////////////
void
deletePBO( GLuint* pbo) {

    pboUnregister(*pbo);

    glBindBuffer( GL_ARRAY_BUFFER, *pbo);
    glDeleteBuffers( 1, pbo);
    CUT_CHECK_ERROR_GL();

    glBindBuffer( GL_ARRAY_BUFFER, 0);

    *pbo = 0;
}

// render a simple 3D scene
void renderScene()
{
    glMatrixMode( GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();

    gluLookAt(0, 1.0, -0.5, 0, 0, -3, 0, 1, 0);

    glActiveTextureARB(GL_TEXTURE0_ARB);
    glBindTexture(GL_TEXTURE_2D, texWood);
    glEnable(GL_TEXTURE_2D);
    glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);

    glViewport(0, 0, window_width, window_height);
    glEnable(GL_LIGHTING);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_COLOR_MATERIAL);

    glLightModeli(GL_LIGHT_MODEL_COLOR_CONTROL, GL_SINGLE_COLOR);

    glColor3f(1.0f, 1.0f, 1.0f);
    glBegin(GL_QUADS);
    glTexCoord2f(0.0f, 0.0f); glVertex3f(-50.0f, -0.7f, -50.0f);
    glTexCoord2f(10.0f, 0.0f); glVertex3f(50.0f, -0.7f, -50.0f);
    glTexCoord2f(10.0f, 10.0f); glVertex3f(50.0f, -0.7f, 50.0f);
    glTexCoord2f(0.0f, 10.0f); glVertex3f(-50.0f, -0.7f, 50.0f);
    glEnd();

    glLightModeli(GL_LIGHT_MODEL_COLOR_CONTROL, GL_SEPARATE_SPECULAR_COLOR);

    int numTeapots = 4;
    float x = -1.0f;
    float z = -3.5f;
    float r = 0.0f;

    for (int i = 0; i < numTeapots; ++i)
    {
        glPushMatrix();
        glColor3f(r, 0.6f, 1.0f);
        glTranslatef(x, 0.0, z);
        glRotatef(rotate[0], 1.0, 0.0, 0.0);
        glRotatef(rotate[1], 0.0, 1.0, 0.0);
        glRotatef(rotate[2], 0.0, 0.0, 1.0);
        glCallList(dlTeapot);
        glPopMatrix();
        x += 6.0f / (numTeapots - 1);
        z -= 3.5f;
        r += 1.0f / (numTeapots - 1);
    }

    glDisable(GL_TEXTURE_2D);
    
    glPopMatrix();

    CUT_CHECK_ERROR_GL();
}

// copy image and process using CUDA
void processImage()
{
    // read data into pbo
    // activate destination buffer
    glBindBuffer( GL_PIXEL_PACK_BUFFER_ARB, pbo_source);
    // read
    glReadPixels( 0, 0, image_width, image_height, GL_RGBA, GL_UNSIGNED_BYTE, NULL); 

    glBindBuffer(GL_PIXEL_PACK_BUFFER_ARB, 0);

    //cutResetTimer(timer);
    //CUT_SAFE_CALL(cutStartTimer(timer));

    // run the Cuda kernel    
    process( pbo_source, pbo_dest, image_width, image_height, blur_radius);
    
    //CUT_SAFE_CALL(cutStopTimer(timer));

    //printf("Process: %0.2f\n", cutGetTimerValue(timer));


    // download texture from PBO
    glBindBuffer( GL_PIXEL_UNPACK_BUFFER_ARB, pbo_dest);
    glBindTexture( GL_TEXTURE_2D, tex_screen);
    glTexSubImage2D( GL_TEXTURE_2D, 0, 0, 0, image_width, image_height, GL_RGBA, GL_FLOAT, NULL);
    glBindBuffer( GL_PIXEL_UNPACK_BUFFER_ARB, 0);


    CUT_CHECK_ERROR_GL();
}

void displayImage()
{
    // render a screen sized quad
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_LIGHTING);
    
    glUseProgram(glprogramSAT);
    GLuint satTexLoc = glGetUniformLocation(glprogramSAT, "sat");
    glUniform1i(satTexLoc, 0);
    GLuint distTexLoc = glGetUniformLocation(glprogramSAT, "distance");
    glUniform1i(distTexLoc, 1);
    GLuint radiusLoc = glGetUniformLocation(glprogramSAT, "radius");
    glUniform1f(radiusLoc, blur_radius);
    GLuint widthLoc = glGetUniformLocation(glprogramSAT, "resolution");
    glUniform1f(widthLoc, image_width);
    GLuint zScaleBiasLoc = glGetUniformLocation(glprogramSAT, "zScaleBias");
    glUniform2f(zScaleBiasLoc, zScale, zBias);


    glActiveTextureARB(GL_TEXTURE0_ARB);
    glBindTexture(GL_TEXTURE_2D, tex_screen);
    glActiveTextureARB(GL_TEXTURE1_ARB);
    glBindTexture(GL_TEXTURE_2D, tex_distance);

    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glOrtho(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0);

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();

    glViewport(0, 0, window_width, window_height);

    glBegin(GL_QUADS);

    glTexCoord2f( 0.0, 0.0);
    glVertex3f( -1.0, -1.0, 0.5);

    glTexCoord2f( 1.0, 0.0);
    glVertex3f(  1.0, -1.0, 0.5);

    glTexCoord2f( 1.0, 1.0);
    glVertex3f(  1.0,  1.0, 0.5);

    glTexCoord2f( 0.0, 1.0);
    glVertex3f( -1.0,  1.0, 0.5);

    glEnd();

    glPopMatrix();
    glMatrixMode( GL_PROJECTION);
    glPopMatrix();

    glDisable( GL_TEXTURE_2D);
    glBindBuffer( GL_PIXEL_PACK_BUFFER_ARB, 0);
    glBindBuffer( GL_PIXEL_UNPACK_BUFFER_ARB, 0);

    glUseProgram(0);

    // re-attach to CUDA
    //pboRegister(pbo_dest);

    CUT_CHECK_ERROR_GL();
}

////////////////////////////////////////////////////////////////////////////////
//! Display callback
////////////////////////////////////////////////////////////////////////////////
void
display() 
{
    {
        // render color to back buffer
        glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);    
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();
        renderScene();

        // render distance from y=0 to distance texture
        glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, fbo);
        glUseProgram(glprogramColorAndDistance);
        //GLenum buffers[] = { GL_COLOR_ATTACHMENT0_EXT, GL_COLOR_ATTACHMENT1_EXT };
        //glDrawBuffers(2, buffers);

        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);    
        renderScene();
            
        glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
        glUseProgram(0);       
        
        CUT_CHECK_ERROR_GL();
    }
    
    if (enable_cuda) {
        processImage();
        displayImage();
    }

    glutSwapBuffers();
    CUT_CHECK_ERROR_GL();
}

void idle()
{
    if (animate) {
        rotate[0] += 0.2;
        rotate[1] += 0.6;
        //rotate[2] += 1.0;
    }
    glutPostRedisplay();
}

////////////////////////////////////////////////////////////////////////////////
//! Keyboard events handler
////////////////////////////////////////////////////////////////////////////////
void
keyboard( unsigned char key, int /*x*/, int /*y*/) {

    switch( key) {
    case( 27) :
        deletePBO( &pbo_source);
        deletePBO( &pbo_dest);
        deleteFBO( &fbo, &tex_distance);
        deleteTexture( &tex_screen);
        exit( 0);
    case ' ':
        enable_cuda ^= 1;
        break;
    case 'r':
        animate ^= 1;
        break;
    case '=':
    case '+':
        if (zScale < 1000) zScale+=1.0f;
        printf("zScale = %f\n", zScale);
        break;
    case '-':
        if (zScale > 1) zScale-=1.0f;
        printf("zScale = %f\n", zScale);
        break;
    case 'a':
        if (zBias < 1) zBias += 0.001;
        printf("zBias = %f\n", zBias);
        break;
    case 'z':
        if (zBias > 0) zBias -= 0.001;
        printf("zBias = %f\n", zBias);
        break;
    }
}

void reshape(int w, int /*h*/)
{
    window_width = w;
    window_height = w;
}

void mainMenu(int i)
{
  keyboard((unsigned char) i, 0, 0);
}

////////////////////////////////////////////////////////////////////////////////
//! Create offscreen render target
////////////////////////////////////////////////////////////////////////////////
void
createFBO( GLuint* fbo, GLuint* tex, GLuint* depth_rb = 0) {

    // create a new fbo
    glGenFramebuffersEXT( 1, fbo);
    glBindFramebufferEXT( GL_FRAMEBUFFER_EXT, *fbo);
    CUT_CHECK_ERROR_GL();

    // check if the fbo is valid
    if( ! glIsFramebufferEXT( *fbo)) {
        fprintf( stderr, "Framebuffer object creation failed.\n");
        fflush( stderr);

        return;
    }

    // create attachment
    createTexture( tex, image_width, image_height, true, false);

    CUT_CHECK_ERROR_GL();

    if (depth_rb != 0)
    {
        glGenRenderbuffersEXT(1, depth_rb);
        glBindRenderbufferEXT(GL_RENDERBUFFER_EXT, *depth_rb);
        glRenderbufferStorageEXT(GL_RENDERBUFFER_EXT,
                                 GL_DEPTH_COMPONENT24, image_width, image_height);
        glFramebufferRenderbufferEXT(GL_FRAMEBUFFER_EXT,
                                     GL_DEPTH_ATTACHMENT_EXT,
                                     GL_RENDERBUFFER_EXT, *depth_rb);
    }

    
    
    // attach texture to fbo
    glFramebufferTexture2DEXT( GL_FRAMEBUFFER_EXT,
        GL_COLOR_ATTACHMENT0_EXT,
        GL_TEXTURE_2D, 
        *tex, 
        0);

    CUT_CHECK_ERROR_GL();

    // deactivate offsreen render target
    glBindFramebufferEXT( GL_FRAMEBUFFER_EXT, 0);

    CUT_CHECK_ERROR_GL();
}

////////////////////////////////////////////////////////////////////////////////
//! Cleanup offscreen render target
////////////////////////////////////////////////////////////////////////////////
void
deleteFBO( GLuint* fbo, GLuint* tex) {

    glDeleteFramebuffersEXT( 1, fbo);
    CUT_CHECK_ERROR_GL();

    deleteTexture( tex);

    *fbo = 0;  
}

////////////////////////////////////////////////////////////////////////////////
//! 
////////////////////////////////////////////////////////////////////////////////
void
deleteTexture( GLuint* tex) {

    glDeleteTextures( 1, tex);
    CUT_CHECK_ERROR_GL();

    *tex = 0;
}

////////////////////////////////////////////////////////////////////////////////
//! 
////////////////////////////////////////////////////////////////////////////////
void
createTexture( GLuint* tex_name, unsigned int size_x, unsigned int size_y, 
               bool bUseFloat, bool singleChannel) 
{
    // create a tex as attachment
    glGenTextures( 1, tex_name);
    glBindTexture( GL_TEXTURE_2D, *tex_name);

    // set basic parameters
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    // buffer data
    if (bUseFloat)
        glTexImage2D( GL_TEXTURE_2D, 0, 
                      singleChannel ? singleChannelFloatIntFormat : GL_RGBA32F_ARB, 
                      size_x, size_y, 0, 
                      singleChannel ? singleChannelFloatFormat : GL_RGBA, 
                      GL_FLOAT, NULL);
    else
        glTexImage2D( GL_TEXTURE_2D, 0, 
                      singleChannel ? GL_LUMINANCE8 : GL_RGBA8, 
                      size_x, size_y, 0, 
                      singleChannel ? GL_LUMINANCE : GL_RGBA, 
                      GL_UNSIGNED_BYTE, NULL);

    CUT_CHECK_ERROR_GL();
}
