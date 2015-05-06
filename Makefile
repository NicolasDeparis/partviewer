
ARCH = GPU

C_OBJS = 	main.o\
					Part.o\
					freeflycamera.o\
					scene.o\
					vector3d.o

WORKDIR = `pwd`

CC = gcc
CXX = g++
NVCC = nvcc

AR = ar
LD = g++
WINDRES = windres

INC =
<<<<<<< HEAD
CFLAGS = -fopenmp -DGL_GLEXT_PROTOTYPES -I/Library/Frameworks/SDL.framework/Headers
RESINC =
LIBDIR =
# LIB = -lSDL `sdl-config --cflags --libs` -framework Cocoa -lGL -lGLU -lgomp
LIB = -framework SDL -framework OpenGl -framework GLUT -framework Cocoa -v -lc++
# -macosx_version_min 10.9.0
LDFLAGS = 
=======

CFLAGS = -DGL_GLEXT_PROTOTYPES

ifeq ($(ARCH),GPU)
	$(CFLAGS) = $(CFLAGS) -DCUDA
else
	$(CFLAGS) = $(CFLAGS) -fopenmp
endif

RESINC =
LIBDIR =
LIB = -lSDL -lGL -lGLU -lgomp
LDFLAGS =
>>>>>>> 5d00892ff44d7d34e7ffb8c45e3d5f537263202a

INC_DEFAULT = $(INC)
CFLAGS_DEFAULT = $(CFLAGS)
RESINC_DEFAULT = $(RESINC)
RCFLAGS_DEFAULT = $(RCFLAGS)
LIBDIR_DEFAULT = $(LIBDIR)
LIB_DEFAULT = $(LIB)
LDFLAGS_DEFAULT = $(LDFLAGS)
OBJDIR_DEFAULT = obj
SRCDIR_DEFAULT = src
DEP_DEFAULT =
OUT_DEFAULT = viewer

OBJ_DEFAULT =$(patsubst %,$(OBJDIR_DEFAULT)/%,$(C_OBJS))

all: default

clean: clean_default

before_default:
	test -d $(OBJDIR_DEFAULT) || mkdir -p $(OBJDIR_DEFAULT)
ifeq ($(ARCH),GPU)
	echo "GPU"
else
	echo "CPU"
endif
ifeq ($(ARCH),GPU)
	$(CFLAGS) = $(CFLAGS) -DCUDA
else
	$(CFLAGS) = $(CFLAGS) -fopenmp
endif


after_default:

default: before_default out_default after_default

out_default: before_default $(OBJ_DEFAULT) $(DEP_DEFAULT)
	$(LD) $(LIBDIR_DEFULT) -o $(OUT_DEFAULT) $(OBJ_DEFAULT)  $(LDFLAGS_DEFAULT) $(LIB_DEFAULT)


$(OBJDIR_DEFAULT)/%.o: $(SRCDIR_DEFAULT)/%.cpp
	$(CXX) $(CFLAGS_DEFAULT) $(INC_DEFAULT) -c $< -o $@


ifeq ($(ARCH),GPU)
$(OBJDIR_DEFAULT)/%.o: $(SRCDIR_DEFAULT)/%.cu
	$(NVCC) $(CFLAGS_DEFAULT) $(INC_DEFAULT) -c $< -o $@
endif

clean_default:
	rm -f $(OBJ_DEFAULT) $(OUT_DEFAULT)
	rm -rf $(OBJDIR_DEFAULT)

virtual_all: default

.PHONY: before_default after_default clean_default

