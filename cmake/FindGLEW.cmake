#
# Try to find GLEW library and include path.
# Once done this will define
#
# GLEW_FOUND
# GLEW_INCLUDE_PATH
# GLEW_LIBRARY
# 
IF (WIN32)
	FIND_PATH( GLEW_INCLUDE_PATH GL/glew.h
		${GLEW_ROOT_DIR}/include
		DOC "The directory where GL/glew.h resides")
	FIND_LIBRARY( GLEW_LIBRARY
		NAMES glew GLEW glew32 glew32s
		PATHS
		${GLEW_ROOT_DIR}/bin
		${GLEW_ROOT_DIR}/lib
		DOC "The GLEW library")
ELSE (WIN32)
        FIND_PATH( GLEW_INCLUDE_PATH GL/glew.h
		/usr/include
		/usr/local/include
		/sw/include
		/opt/local/include
        ${GLEW_ROOT_DIR}/include
		DOC "The directory where GL/glew.h resides")
	FIND_LIBRARY( GLEW_LIBRARY
		NAMES GLEW libGLEW
		PATHS
		/usr/lib64
		/usr/lib
        /usr/local/lib64
		/usr/local/lib
		/sw/lib
		/opt/local/lib
        ${GLEW_ROOT_DIR}/lib
		DOC "The GLEW library")
SET(GLEW_LIBRARY "/usr/lib/libGLEW.so")
ENDIF (WIN32)

IF (GLEW_INCLUDE_PATH)
  SET( FOUND_GLEW 1)
ELSE (GLEW_INCLUDE_PATH)
  SET( FOUND_GLEW 0)
ENDIF (GLEW_INCLUDE_PATH)

MARK_AS_ADVANCED( FOUND_GLEW )