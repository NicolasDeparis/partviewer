SOURCES =	main.cpp \
		vector3d.cpp \
		freeflycamera.cpp \
		scene.cpp\
		Part.cpp
OBJECTS	=	$(SOURCES:.cpp=.o)
TARGET	=	viewer
LIBS	=	$(shell sdl-config --libs) -lSDL -lGL -lGLU

all: $(OBJECTS)
	g++  -o $(TARGET) $(OBJECTS) $(LIBS)

%o: %cpp
	g++ -o $@ -c $<

x: all
	./$(TARGET)

clean:
	rm -rf $(OBJECTS)

superclean : clean
	rm -rf $(TARGET)

