
C_OBJS = 	main.o\
					Part.o\
					freeflycamera.o\
					scene.o\
					vector3d.o\
					physic.o\
					kernel.o

WORKDIR = `pwd`

CC = gcc
CXX = g++
NVCC = nvcc

AR = ar
LD = nvcc
WINDRES = windres

INC =

CFLAGS = -DGL_GLEXT_PROTOTYPES -DCUDA

RESINC =
LIBDIR =
LIB = -lSDL -lGL -lGLU -lgomp
LDFLAGS =

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

after_default:

default: before_default out_default after_default

out_default: before_default $(OBJ_DEFAULT) $(DEP_DEFAULT)
	$(LD) $(LIBDIR_DEFULT) -o $(OUT_DEFAULT) $(OBJ_DEFAULT)  $(LDFLAGS_DEFAULT) $(LIB_DEFAULT)


$(OBJDIR_DEFAULT)/%.o: $(SRCDIR_DEFAULT)/%.cpp
	$(CXX) $(CFLAGS_DEFAULT) $(INC_DEFAULT) -c $< -o $@

$(OBJDIR_DEFAULT)/%.o: $(SRCDIR_DEFAULT)/kernel.cu
	$(NVCC) $(CFLAGS_DEFAULT) $(INC_DEFAULT) -c $< -o $@


clean_default:
	rm -f $(OBJ_DEFAULT) $(OUT_DEFAULT)
	rm -rf $(OBJDIR_DEFAULT)

virtual_all: default

.PHONY: before_default after_default clean_default

